"""
TAPE Framework - Complete End-to-End Pipeline

This script implements the complete TAPE (Text Augmentation with Pre-trained Encoders)
framework for rumor detection with the following stages:

Stage 1: Initial Feature Extraction (baseline TF-IDF features)
Stage 2: Adaptive Node Selection (identify nodes to augment)
Stage 3: Selective Augmentation & Encoding (LLM + LM)
Stage 4: Feature Fusion & GNN Classification

Usage:
    python tape_pipeline.py --dataset Twitter15 --enable_augmentation
"""

import os
import argparse
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm
import pickle

from config import Config
from data_preprocessing import TwitterDataProcessor, WeiboDataProcessor, split_data
from node_selector import NodeSelector
from node_augmentor import LanguageModelEncoder, NodeAugmentor
from model_tape import get_tape_model
from torch_geometric.loader import DataLoader
# Note: utils module not needed - functions not used in this script


class TAPEPipeline:
    """
    Complete TAPE Framework Pipeline
    """
    
    def __init__(
        self,
        config: Config = None,
        enable_augmentation: bool = True,
        node_selection_strategy: str = "hybrid",
        fusion_strategy: str = "concat",
        augmentation_ratio: float = 0.3,
        use_llm: bool = False,
        gnn_backbone: str = "gcn"
    ):
        """
        Initialize TAPE Pipeline
        
        Args:
            config: Configuration object
            enable_augmentation: Whether to enable node augmentation
            node_selection_strategy: Strategy for selecting nodes
            fusion_strategy: Strategy for feature fusion
            augmentation_ratio: Ratio of nodes to augment per graph
            use_llm: Whether to use LLM for augmentation
            gnn_backbone: GNN backbone type ('gcn' or 'gat')
        """
        self.config = config or Config()
        self.enable_augmentation = enable_augmentation
        self.node_selection_strategy = node_selection_strategy
        self.fusion_strategy = fusion_strategy
        self.augmentation_ratio = augmentation_ratio
        self.use_llm = use_llm
        self.gnn_backbone = gnn_backbone
        
        # Components
        self.node_selector = None
        self.lm_encoder = None
        self.node_augmentor = None
        self.model = None
        
        # Statistics
        self.stats = {
            'total_graphs': 0,
            'total_nodes': 0,
            'augmented_nodes': 0,
            'augmentation_time': 0.0
        }
        
        print("="*70)
        print("TAPE Framework Pipeline Initialized")
        print("="*70)
        print(f"  GNN Backbone: {gnn_backbone.upper()}")
        print(f"  Augmentation enabled: {enable_augmentation}")
        print(f"  Node selection strategy: {node_selection_strategy}")
        print(f"  Fusion strategy: {fusion_strategy}")
        print(f"  Augmentation ratio: {augmentation_ratio}")
        print(f"  Use LLM: {use_llm}")
        print("="*70)
    
    def setup_components(self):
        """
        Setup pipeline components
        """
        print("\n" + "="*70)
        print("Setting up TAPE Components")
        print("="*70)
        
        if self.enable_augmentation:
            # Stage 2: Node Selector
            print("\n[Stage 2] Initializing Node Selector...")
            self.node_selector = NodeSelector(
                strategy=self.node_selection_strategy,
                top_k_ratio=self.augmentation_ratio,
                min_nodes=1,
                max_nodes=10
            )
            print("✓ Node Selector initialized")
            
            # Stage 3: LM Encoder and Node Augmentor
            print("\n[Stage 3] Initializing LM Encoder and Augmentor...")
            self.lm_encoder = LanguageModelEncoder(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.node_augmentor = NodeAugmentor(
                lm_encoder=self.lm_encoder,
                use_llm=self.use_llm
            )
            print("✓ LM Encoder and Augmentor initialized")
        
        print("\n✓ All components initialized")
    
    def process_data(
        self,
        dataset_name: str,
        sample_ratio: float = 1.0
    ):
        """
        Stage 1: Load and preprocess data (baseline features)
        
        Args:
            dataset_name: Name of dataset
            sample_ratio: Data sampling ratio
        
        Returns:
            train_list, val_list, test_list
        """
        print("\n" + "="*70)
        print("[Stage 1] Initial Feature Extraction")
        print("="*70)
        
        # Choose processor
        use_bert = getattr(self.config, 'USE_BERT_FEATURES', False)
        
        if dataset_name == 'Weibo':
            processor = WeiboDataProcessor(
                dataname=dataset_name,
                feature_dim=self.config.FEATURE_DIM,
                sample_ratio=sample_ratio,
                use_bert=use_bert
            )
        else:
            processor = TwitterDataProcessor(
                dataname=dataset_name,
                feature_dim=self.config.FEATURE_DIM,
                sample_ratio=sample_ratio,
                use_bert=use_bert
            )
        
        # Load processed data
        feature_type = "bert" if use_bert else "tfidf"
        if sample_ratio == 1.0:
            processed_filename = f'{dataset_name}_processed_{feature_type}_full.pkl'
        else:
            processed_filename = f'{dataset_name}_processed_{feature_type}_sample{sample_ratio}.pkl'
        
        processed_path = os.path.join(self.config.PROCESSED_DIR, processed_filename)
        
        if os.path.exists(processed_path):
            print(f"Loading processed data: {processed_path}")
            graph_list = processor.load_processed_data(processed_path)
        else:
            print("Processing raw data...")
            graph_list = processor.process_data()
            processor.save_processed_data(graph_list)
        
        # Split data
        train_list, val_list, test_list = split_data(
            graph_list,
            train_ratio=self.config.TRAIN_RATIO,
            val_ratio=self.config.VAL_RATIO,
            test_ratio=self.config.TEST_RATIO,
            seed=self.config.SEED
        )
        
        self.stats['total_graphs'] = len(graph_list)
        self.stats['total_nodes'] = sum(data.x.shape[0] for data in graph_list)
        
        print(f"\n✓ Data loaded:")
        print(f"  Train: {len(train_list)} graphs")
        print(f"  Val: {len(val_list)} graphs")
        print(f"  Test: {len(test_list)} graphs")
        print(f"  Total nodes: {self.stats['total_nodes']:,}")
        
        return train_list, val_list, test_list
    
    def augment_data(
        self,
        data_list,
        split_name: str = "train"
    ):
        """
        Stage 2 & 3: Node selection and augmentation
        
        Args:
            data_list: List of PyG Data objects
            split_name: Name of data split
        
        Returns:
            Augmented data list
        """
        if not self.enable_augmentation:
            print(f"\n⚠️  Augmentation disabled, skipping...")
            return data_list
        
        print("\n" + "="*70)
        print(f"[Stage 2 & 3] Node Selection and Augmentation ({split_name})")
        print("="*70)
        
        import time
        start_time = time.time()
        
        # Stage 2: Select nodes
        print("\n[Stage 2] Selecting nodes for augmentation...")
        
        if split_name == "train" and not self.node_selector.is_fitted:
            self.node_selector.fit(data_list)
        
        selected_nodes_list = self.node_selector.select_batch(data_list)
        
        total_selected = sum(len(nodes) for nodes in selected_nodes_list)
        total_nodes = sum(data.x.shape[0] for data in data_list)
        
        print(f"✓ Node selection completed:")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Selected nodes: {total_selected:,} ({total_selected/total_nodes*100:.1f}%)")
        
        # Get statistics
        stats = self.node_selector.get_selection_stats(data_list)
        print(f"  Avg selected per graph: {stats['avg_selected_per_graph']:.1f}")
        
        # Stage 3: Augment selected nodes
        print("\n[Stage 3] Augmenting selected nodes...")
        
        augmented_list = self.node_augmentor.augment_batch(
            data_list,
            selected_nodes_list,
            texts_list=None,  # Would need actual texts here
            verbose=True
        )
        
        elapsed = time.time() - start_time
        self.stats['augmentation_time'] += elapsed
        self.stats['augmented_nodes'] += total_selected
        
        print(f"\n✓ Augmentation completed:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Augmented embedding dim: {self.lm_encoder.embedding_dim}")
        
        return augmented_list
    
    def train_model(
        self,
        train_list,
        val_list,
        test_list,
        dataset_name: str
    ):
        """
        Stage 4: Train GCN with fused features
        
        Args:
            train_list: Training data
            val_list: Validation data
            test_list: Test data
            dataset_name: Dataset name
        
        Returns:
            Training results
        """
        print("\n" + "="*70)
        print("[Stage 4] Feature Fusion & GNN Training")
        print("="*70)
        
        # Create model
        augmented_dim = self.lm_encoder.embedding_dim if self.enable_augmentation else 0
        
        self.model = get_tape_model(
            model_type="tape" if self.enable_augmentation else "baseline",
            gnn_backbone=self.gnn_backbone,
            baseline_dim=self.config.FEATURE_DIM,
            augmented_dim=augmented_dim,
            hidden_dim=self.config.HIDDEN_DIM,
            num_classes=self.config.NUM_CLASSES,
            dropout=self.config.DROPOUT,
            fusion_strategy=self.fusion_strategy,
            gat_heads=self.config.GAT_HEADS if hasattr(self.config, 'GAT_HEADS') else 4
        )
        
        self.model.to(self.config.DEVICE)
        
        # Create data loaders
        train_loader = DataLoader(
            train_list,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS
        )
        
        val_loader = DataLoader(
            val_list,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        test_loader = DataLoader(
            test_list,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS
        )
        
        # Training
        print("\nStarting training...")
        # Using built-in training loop
        results = self._simple_training_loop(
            train_loader, val_loader, test_loader, dataset_name
        )
        
        return results
    
    def _simple_training_loop(
        self,
        train_loader,
        val_loader,
        test_loader,
        dataset_name
    ):
        """
        Simple training loop
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        criterion = torch.nn.NLLLoss()
        
        best_val_acc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [], 'val_loss': [], 
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
        
        print(f"\nTraining for {self.config.NUM_EPOCHS} epochs...")
        print(f"  Device: {self.config.DEVICE}")
        print(f"  Batch size: {self.config.BATCH_SIZE}")
        print(f"  Learning rate: {self.config.LEARNING_RATE}")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []
            
            for batch in train_loader:
                batch = batch.to(self.config.DEVICE)
                
                optimizer.zero_grad()
                output = self.model(batch)
                loss = criterion(output, batch.y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Collect predictions for metrics
                pred = output.argmax(dim=1)
                train_preds.extend(pred.cpu().numpy())
                train_labels.extend(batch.y.cpu().numpy())
            
            train_loss /= len(train_loader)
            
            # Calculate training metrics
            from sklearn.metrics import accuracy_score, f1_score
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='binary', zero_division=0)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.config.DEVICE)
                    output = self.model(batch)
                    loss = criterion(output, batch.y)
                    val_loss += loss.item()
                    
                    pred = output.argmax(dim=1)
                    val_preds.extend(pred.cpu().numpy())
                    val_labels.extend(batch.y.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # Calculate validation metrics
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.config.NUM_EPOCHS}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 
                          os.path.join(self.config.SAVE_DIR, f'{dataset_name}_tape_best.pt'))
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\n✓ Training completed")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model and evaluate on test set
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.SAVE_DIR, f'{dataset_name}_tape_best.pt'))
        )
        
        test_results = self._evaluate(test_loader, return_predictions=True)
        
        return {
            'history': history,
            'best_val_acc': best_val_acc,
            'test_results': {
                'accuracy': test_results['accuracy'],
                'precision': test_results['precision'],
                'recall': test_results['recall'],
                'f1': test_results['f1']
            },
            'test_predictions': {
                'predictions': test_results['predictions'],
                'labels': test_results['labels'],
                'probabilities': test_results['probabilities']
            }
        }
    
    def _evaluate(self, data_loader, return_predictions=False):
        """Evaluate model
        
        Args:
            data_loader: DataLoader for evaluation
            return_predictions: If True, return predictions and labels for visualization
        
        Returns:
            Dictionary with metrics, optionally includes predictions
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        import torch.nn.functional as F
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.config.DEVICE)
                output = self.model(batch)
                
                # Get probabilities
                probs = F.softmax(output, dim=1)
                
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        if return_predictions:
            results['predictions'] = all_preds
            results['labels'] = all_labels
            results['probabilities'] = all_probs
        
        return results
    
    def run(self, dataset_name: str, sample_ratio: float = 1.0):
        """
        Run complete TAPE pipeline
        
        Args:
            dataset_name: Dataset name
            sample_ratio: Sampling ratio
        
        Returns:
            Results dictionary
        """
        print("\n" + "="*70)
        print(f"Running TAPE Pipeline on {dataset_name}")
        print("="*70)
        
        # Setup
        self.config.create_dirs()
        self.setup_components()
        
        # Stage 1: Load data
        train_list, val_list, test_list = self.process_data(dataset_name, sample_ratio)
        
        # Stage 2 & 3: Augment data
        train_list = self.augment_data(train_list, "train")
        val_list = self.augment_data(val_list, "val")
        test_list = self.augment_data(test_list, "test")
        
        # Stage 4: Train model
        results = self.train_model(train_list, val_list, test_list, dataset_name)
        
        # Print final results
        print("\n" + "="*70)
        print("TAPE Pipeline Completed!")
        print("="*70)
        print(f"\nFinal Test Results:")
        print(f"  Accuracy:  {results['test_results']['accuracy']:.4f}")
        print(f"  Precision: {results['test_results']['precision']:.4f}")
        print(f"  Recall:    {results['test_results']['recall']:.4f}")
        print(f"  F1-Score:  {results['test_results']['f1']:.4f}")
        
        print(f"\nPipeline Statistics:")
        print(f"  Total graphs: {self.stats['total_graphs']}")
        print(f"  Total nodes: {self.stats['total_nodes']:,}")
        if self.enable_augmentation:
            print(f"  Augmented nodes: {self.stats['augmented_nodes']:,} "
                  f"({self.stats['augmented_nodes']/self.stats['total_nodes']*100:.1f}%)")
            print(f"  Augmentation time: {self.stats['augmentation_time']:.2f}s")
        
        # Generate visualizations
        if results.get('test_predictions') is not None:
            print("\n" + "="*70)
            print("Generating Visualizations...")
            print("="*70)
            
            from utils.visualization import (
                plot_training_history,
                plot_confusion_matrix,
                plot_prediction_analysis,
                save_results_summary,
                create_log_directory
            )
            
            # Create log directory
            log_dir, timestamp = create_log_directory(self.config.LOG_DIR, dataset_name)
            print(f"\nSaving results to: {log_dir}")
            
            # Plot training history
            history_path = os.path.join(log_dir, f'training_history_{timestamp}.png')
            plot_training_history(results['history'], history_path)
            
            # Plot confusion matrix
            class_names = ['Non-rumor', 'Rumor']
            cm_path = os.path.join(log_dir, f'confusion_matrix_{timestamp}.png')
            plot_confusion_matrix(
                results['test_predictions']['labels'],
                results['test_predictions']['predictions'],
                class_names,
                cm_path
            )
            
            # Plot prediction analysis
            pred_path = os.path.join(log_dir, f'prediction_analysis_{timestamp}.png')
            plot_prediction_analysis(
                results['test_predictions']['labels'],
                results['test_predictions']['predictions'],
                results['test_predictions']['probabilities'],
                pred_path
            )
            
            # Save results summary
            config_info = {
                'Enable Augmentation': self.enable_augmentation,
                'Node Strategy': self.node_selection_strategy,
                'Fusion Strategy': self.fusion_strategy,
                'Augmentation Ratio': self.augmentation_ratio,
                'GNN Backbone': self.gnn_backbone,
                'Sample Ratio': sample_ratio,
            }
            
            summary_path = os.path.join(log_dir, f'results_summary_{timestamp}.txt')
            save_results_summary(
                results['test_results'],
                results['history'],
                summary_path,
                dataset_name,
                config_info
            )
            
            print("\n✓ All visualizations saved successfully!")
            print("="*70)
        
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TAPE Framework for Rumor Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='Twitter15',
                       choices=['Twitter15', 'Twitter16', 'Weibo'],
                       help='Dataset name')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Data sampling ratio')
    
    # TAPE parameters
    parser.add_argument('--enable_augmentation', action='store_true',
                       help='Enable node-level augmentation')
    parser.add_argument('--node_strategy', type=str, default='hybrid',
                       choices=['uncertainty', 'importance', 'hybrid'],
                       help='Node selection strategy')
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                       choices=['concat', 'weighted', 'gated', 'attention'],
                       help='Feature fusion strategy')
    parser.add_argument('--augmentation_ratio', type=float, default=0.3,
                       help='Ratio of nodes to augment per graph')
    parser.add_argument('--use_llm', action='store_true',
                       help='Use LLM for text augmentation')
    parser.add_argument('--gnn_backbone', type=str, default='gcn',
                       choices=['gcn', 'gat'],
                       help='GNN backbone type')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = TAPEPipeline(
        enable_augmentation=args.enable_augmentation,
        node_selection_strategy=args.node_strategy,
        fusion_strategy=args.fusion_strategy,
        augmentation_ratio=args.augmentation_ratio,
        use_llm=args.use_llm,
        gnn_backbone=args.gnn_backbone
    )
    
    # Run pipeline
    results = pipeline.run(
        dataset_name=args.dataset,
        sample_ratio=args.sample_ratio
    )
    
    return results


if __name__ == '__main__':
    main()

