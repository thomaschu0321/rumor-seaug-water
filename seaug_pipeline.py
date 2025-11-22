"""
SeAug Framework - Complete End-to-End Pipeline

This script implements the complete SeAug (Selective LLM Augmentation Pipeline)
framework for rumor detection with the following stages:

Stage 1: Initial Feature Extraction (baseline TF-IDF features)
Stage 2: Adaptive Node Selection (identify nodes to augment)
Stage 3: Selective Augmentation & Encoding (LLM + LM)
Stage 4: Feature Fusion & GNN Classification

Usage:
    python seaug_pipeline.py --dataset Twitter15 --enable_augmentation
"""

import os
import csv
import argparse
import numpy as np
import torch
from tqdm import tqdm

from config import Config
from data_preprocessing import TwitterDataProcessor, WeiboDataProcessor, split_data
from node_selector import NodeSelector
from node_augmentor import LanguageModelEncoder, NodeAugmentor
from model_seaug import get_seaug_model
from torch_geometric.loader import DataLoader


class SeAugPipeline:
    """
    Complete SeAug Framework Pipeline
    """
    
    def __init__(
        self,
        config: Config = None,
        enable_augmentation: bool = True,
        node_selection_strategy: str = "hybrid",
        fusion_strategy: str = "concat",
        augmentation_ratio: float = 0.3,
        gnn_backbone: str = "gcn",
        batch_size: int = 20
    ):
        """
        Initialize SeAug Pipeline
        
        Args:
            config: Configuration object
            enable_augmentation: Whether to enable node augmentation
            node_selection_strategy: Strategy for selecting nodes
            fusion_strategy: Strategy for feature fusion
            augmentation_ratio: Ratio of nodes to augment per graph
            gnn_backbone: GNN backbone type ('gcn' or 'gat')
            batch_size: Number of nodes per API call for batched processing (default: 20)
        """
        self.config = config or Config()
        
        self.enable_augmentation = enable_augmentation
        self.node_selection_strategy = node_selection_strategy
        self.fusion_strategy = fusion_strategy
        self.augmentation_ratio = augmentation_ratio
        self.gnn_backbone = gnn_backbone
        self.batch_size = batch_size
        
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
        print("SeAug Framework Pipeline Initialized")
        print("="*70)
        print(f"  GNN Backbone: {gnn_backbone.upper()}")
        print(f"  Augmentation enabled: {enable_augmentation}")
        print(f"  Node selection strategy: {node_selection_strategy}")
        print(f"  Fusion strategy: {fusion_strategy}")
        print(f"  Augmentation ratio: {augmentation_ratio}")
        print(f"  LLM augmentation: {enable_augmentation}")
        if enable_augmentation:
            print(f"  Batch size: {batch_size} nodes/call (Token optimized!)")
        print("="*70)
    
    def setup_components(self):
        """
        Setup pipeline components
        """
        print("\n" + "="*70)
        print("Setting up SeAug Components")
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
            print("Node Selector initialized")
            
            # Stage 3: LM Encoder and Node Augmentor
            print("\n[Stage 3] Initializing LM Encoder and Augmentor...")
            self.lm_encoder = LanguageModelEncoder(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            self.node_augmentor = NodeAugmentor(
                lm_encoder=self.lm_encoder,
            )
            print("LM Encoder and Augmentor initialized")
        
        print("\nAll components initialized")
    
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
        
        # Choose processor (always uses BERT)
        if dataset_name == 'Weibo':
            processor = WeiboDataProcessor(
                dataname=dataset_name,
                feature_dim=self.config.FEATURE_DIM,
                sample_ratio=sample_ratio
            )
        else:
            processor = TwitterDataProcessor(
                dataname=dataset_name,
                feature_dim=self.config.FEATURE_DIM,
                sample_ratio=sample_ratio
            )
        
        # Load processed data (always BERT)
        if sample_ratio == 1.0:
            processed_filename = f'{dataset_name}_processed_bert_full.pkl'
        else:
            processed_filename = f'{dataset_name}_processed_bert_sample{sample_ratio}.pkl'
        
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
        
        print(f"\nData loaded:")
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
            print(f"\nAugmentation disabled, skipping...")
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
        
        selected_nodes_list = [self.node_selector.select_nodes(data) for data in data_list]
        
        total_selected = sum(len(nodes) for nodes in selected_nodes_list)
        total_nodes = sum(data.x.shape[0] for data in data_list)
        
        print(f"Node selection completed:")
        print(f"  Total nodes: {total_nodes:,}")
        print(f"  Selected nodes: {total_selected:,} ({total_selected/total_nodes*100:.1f}%)")
        
        # Get statistics
        stats = self.node_selector.get_selection_stats(data_list)
        print(f"  Avg selected per graph: {stats['avg_selected_per_graph']:.1f}")
        
        # Stage 3: Augment selected nodes
        print("\n[Stage 3] Augmenting selected nodes...")
        print(f"  Using batched API calls: {self.batch_size} nodes per call")
        
        augmented_list = self.node_augmentor.augment_batch(
            data_list,
            selected_nodes_list,
            texts_list=None,  # Would need actual texts here
            verbose=True,
            batch_size=self.batch_size
        )
        
        elapsed = time.time() - start_time
        self.stats['augmentation_time'] += elapsed
        self.stats['augmented_nodes'] += total_selected
        
        print(f"\nAugmentation completed:")
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
        
        self.model = get_seaug_model(
            model_type="seaug" if self.enable_augmentation else "baseline",
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
                          os.path.join(self.config.SAVE_DIR, f'{dataset_name}_seaug_best.pt'))
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\nTraining completed")
        print(f"  Best validation accuracy: {best_val_acc:.4f}")
        
        # Load best model and evaluate on test set
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.SAVE_DIR, f'{dataset_name}_seaug_best.pt'))
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
    
    def output_results(
        self,
        results: dict,
        dataset_name: str,
        sample_ratio: float = 1.0,
        save_json: bool = True,
        save_csv: bool = True
    ):
        """
        Output and save results after training is complete
        
        Args:
            results: Results dictionary from train_model()
            dataset_name: Dataset name
            sample_ratio: Sampling ratio used
            save_json: Whether to save results to JSON file
            save_csv: Whether to append results to CSV file
        
        Returns:
            Dictionary with output file paths
        """
        import json
        from datetime import datetime
        
        print("\n" + "="*70)
        print("SeAug Pipeline Completed!")
        print("="*70)
        
        # Print test results
        print(f"\nFinal Test Results:")
        print(f"  Accuracy:  {results['test_results']['accuracy']:.4f}")
        print(f"  Precision: {results['test_results']['precision']:.4f}")
        print(f"  Recall:    {results['test_results']['recall']:.4f}")
        print(f"  F1-Score:  {results['test_results']['f1']:.4f}")
        
        # Print pipeline statistics
        print(f"\nPipeline Statistics:")
        print(f"  Total graphs: {self.stats['total_graphs']}")
        print(f"  Total nodes: {self.stats['total_nodes']:,}")
        if self.enable_augmentation:
            print(f"  Augmented nodes: {self.stats['augmented_nodes']:,} "
                  f"({self.stats['augmented_nodes']/self.stats['total_nodes']*100:.1f}%)")
            print(f"  Augmentation time: {self.stats['augmentation_time']:.2f}s")
        
        # Print training history summary
        if 'history' in results and results['history']:
            history = results['history']
            print(f"\nTraining History Summary:")
            print(f"  Best validation accuracy: {results.get('best_val_acc', 0):.4f}")
            if history.get('train_acc'):
                print(f"  Final train accuracy: {history['train_acc'][-1]:.4f}")
            if history.get('val_acc'):
                print(f"  Final val accuracy: {history['val_acc'][-1]:.4f}")
        
        output_files = {}
        
        # Save to JSON
        if save_json:
            json_path = os.path.join(
                self.config.SAVE_DIR,
                f'{dataset_name}_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            json_results = {
                'dataset': dataset_name,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'enable_augmentation': self.enable_augmentation,
                    'node_selection_strategy': self.node_selection_strategy,
                    'fusion_strategy': self.fusion_strategy,
                    'augmentation_ratio': self.augmentation_ratio,
                    'gnn_backbone': self.gnn_backbone,
                    'sample_ratio': sample_ratio,
                    'batch_size': self.batch_size
                },
                'statistics': {
                    'total_graphs': self.stats['total_graphs'],
                    'total_nodes': self.stats['total_nodes'],
                    'augmented_nodes': self.stats['augmented_nodes'],
                    'augmentation_time': self.stats['augmentation_time']
                },
                'test_results': results['test_results'],
                'best_val_acc': results.get('best_val_acc', 0),
                'history': {
                    'train_loss': [float(x) for x in results.get('history', {}).get('train_loss', [])],
                    'val_loss': [float(x) for x in results.get('history', {}).get('val_loss', [])],
                    'train_acc': [float(x) for x in results.get('history', {}).get('train_acc', [])],
                    'val_acc': [float(x) for x in results.get('history', {}).get('val_acc', [])],
                    'train_f1': [float(x) for x in results.get('history', {}).get('train_f1', [])],
                    'val_f1': [float(x) for x in results.get('history', {}).get('val_f1', [])]
                }
            }
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            output_files['json'] = json_path
            print(f"\nResults saved to JSON: {json_path}")
        
        # Save to CSV
        if save_csv:
            csv_path = os.path.join(self.config.PROJECT_ROOT, "results_summary.csv")
            file_exists = os.path.isfile(csv_path)
            fieldnames = [
                "dataset",
                "model_type",
                "enable_augmentation",
                "node_strategy",
                "fusion_strategy",
                "augmentation_ratio",
                "gnn_backbone",
                "sample_ratio",
                "total_graphs",
                "total_nodes",
                "augmented_nodes",
                "augmentation_time",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
            
            # Determine high-level model description
            if self.enable_augmentation:
                model_desc = "SeAug with LLM"
            else:
                model_desc = "Baseline (No Augmentation)"
            
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "model_type": model_desc,
                        "enable_augmentation": self.enable_augmentation,
                        "node_strategy": self.node_selection_strategy,
                        "fusion_strategy": self.fusion_strategy,
                        "augmentation_ratio": self.augmentation_ratio,
                        "gnn_backbone": self.gnn_backbone,
                        "sample_ratio": sample_ratio,
                        "total_graphs": self.stats["total_graphs"],
                        "total_nodes": self.stats["total_nodes"],
                        "augmented_nodes": self.stats["augmented_nodes"],
                        "augmentation_time": round(self.stats["augmentation_time"], 2),
                        "accuracy": results["test_results"]["accuracy"],
                        "precision": results["test_results"]["precision"],
                        "recall": results["test_results"]["recall"],
                        "f1": results["test_results"]["f1"],
                    }
                )
            
            output_files['csv'] = csv_path
            print(f"Results appended to CSV: {csv_path}")
        
        print("="*70)
        
        return output_files
    
    def run(self, dataset_name: str, sample_ratio: float = 1.0):
        """
        Run complete SeAug pipeline
        
        Args:
            dataset_name: Dataset name
            sample_ratio: Sampling ratio
        
        Returns:
            Results dictionary
        """
        print("\n" + "="*70)
        print(f"Running SeAug Pipeline on {dataset_name}")
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
        
        # Output results
        output_files = self.output_results(results, dataset_name, sample_ratio)
        
        # Results and history are returned so that external scripts / notebooks
        # can decide how to present or store them (tables, plots, etc.).
        return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="SeAug Framework for Rumor Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='Twitter15',
                       choices=['Twitter15', 'Twitter16', 'Weibo'],
                       help='Dataset name')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Data sampling ratio')
    
    # SeAug parameters
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
    parser.add_argument('--gnn_backbone', type=str, default='gcn',
                       choices=['gcn', 'gat'],
                       help='GNN backbone type')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Number of nodes per API call for batched processing (default: 20, recommended: 10-20)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = SeAugPipeline(
        enable_augmentation=args.enable_augmentation,
        node_selection_strategy=args.node_strategy,
        fusion_strategy=args.fusion_strategy,
        augmentation_ratio=args.augmentation_ratio,
        gnn_backbone=args.gnn_backbone,
        batch_size=args.batch_size
    )
    
    # Run pipeline
    results = pipeline.run(
        dataset_name=args.dataset,
        sample_ratio=args.sample_ratio
    )
    
    return results


if __name__ == '__main__':
    main()

