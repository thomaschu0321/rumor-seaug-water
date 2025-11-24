"""
SeAug Framework - Complete End-to-End Pipeline

Usage:
    python seaug_pipeline.py --dataset Twitter15 --enable_augmentation
"""

import os
import csv
import argparse
import numpy as np
import torch
from config import Config
from data_preprocessing import TwitterDataProcessor, WeiboDataProcessor, PhemeDataProcessor, split_data
from node_selector import NodeSelector
from node_augmentor import LanguageModelEncoder, NodeAugmentor, QuotaExceededError
from model_seaug import get_seaug_model
from torch_geometric.loader import DataLoader

try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; fall back to plain logging.
    tqdm = None


class SeAugPipeline:
    def __init__(
        self,
        config: Config = None,
        enable_augmentation: bool = True,
        node_selection_strategy: str = "hybrid",
        fusion_strategy: str = "concat",
        augmentation_ratio: float = 0.3,
        gnn_backbone: str = "gcn",
        batch_size: int = 50
    ):
        self.config = config or Config()
        self.enable_augmentation = enable_augmentation
        self.node_selection_strategy = node_selection_strategy
        self.fusion_strategy = fusion_strategy
        self.augmentation_ratio = augmentation_ratio
        self.gnn_backbone = gnn_backbone
        self.batch_size = batch_size
        
        self.node_selector = None
        self.lm_encoder = None
        self.node_augmentor = None
        self.model = None
        
        self.stats = {
            'total_graphs': 0,
            'total_nodes': 0,
            'augmented_nodes': 0,
            'augmentation_time': 0.0
        }

        # Track metadata for dataset-specific artifacts (e.g., PHEME events)
        self.current_events = None
        self.current_event_suffix = None
        
        print(f"SeAug Pipeline: {gnn_backbone.upper()}, Augmentation: {enable_augmentation}, Strategy: {node_selection_strategy}")

    @staticmethod
    def _format_sample_ratio(sample_ratio: float) -> str:
        """Return a compact string for the given sample ratio."""
        ratio_str = f"{sample_ratio}".rstrip('0').rstrip('.')
        return ratio_str or "0"

    def _build_run_tag(self, dataset_name: str, sample_ratio: float) -> str:
        """Compose a consistent file prefix for artifacts produced by a run."""
        sample_tag = self._format_sample_ratio(sample_ratio)
        aug_tag = "aug" if self.enable_augmentation else "noaug"
        backbone_tag = self.gnn_backbone
        parts = [dataset_name]
        if dataset_name.lower() == 'pheme' and self.current_event_suffix:
            parts.append(self.current_event_suffix)
        parts.extend([backbone_tag, aug_tag, f"sample{sample_tag}"])
        return "_".join(parts)
    
    def setup_components(self):
        if self.enable_augmentation:
            self.node_selector = NodeSelector(
                strategy=self.node_selection_strategy,
                top_k_ratio=self.augmentation_ratio,
                min_nodes=1,
                max_nodes=10
            )
            self.lm_encoder = LanguageModelEncoder(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.node_augmentor = NodeAugmentor(lm_encoder=self.lm_encoder)
    
    def process_data(self, dataset_name: str, sample_ratio: float = 1.0):
        print(f"[Stage 1] Loading data: {dataset_name}")
        
        # Reset dataset-specific metadata before loading a new dataset
        self.current_events = None
        self.current_event_suffix = None
        
        if dataset_name == 'Weibo':
            processor = WeiboDataProcessor(
                dataname=dataset_name,
                feature_dim=self.config.FEATURE_DIM,
                sample_ratio=sample_ratio
            )
        elif dataset_name == 'PHEME':
            processor = PhemeDataProcessor(
                dataname=dataset_name,
                feature_dim=self.config.FEATURE_DIM,
                sample_ratio=sample_ratio,
                events=self.config.PHEME_EVENTS or None
            )
            self.current_events = getattr(processor, 'events', None)
            self.current_event_suffix = getattr(processor, 'event_suffix', None)
        else:
            processor = TwitterDataProcessor(
                dataname=dataset_name,
                feature_dim=self.config.FEATURE_DIM,
                sample_ratio=sample_ratio
            )
        
        processed_filename = getattr(
            processor,
            'processed_filename',
            f'{dataset_name}_processed_bert_full.pkl' if sample_ratio == 1.0
            else f'{dataset_name}_processed_bert_sample{sample_ratio}.pkl'
        )
        processed_path = os.path.join(self.config.PROCESSED_DIR, processed_filename)
        
        if os.path.exists(processed_path):
            graph_list = processor.load_processed_data(processed_path)
        else:
            graph_list = processor.process_data()
            processor.save_processed_data(graph_list)
        
        train_list, val_list, test_list = split_data(
            graph_list,
            train_ratio=self.config.TRAIN_RATIO,
            val_ratio=self.config.VAL_RATIO,
            test_ratio=self.config.TEST_RATIO,
            seed=self.config.SEED
        )
        
        self.stats['total_graphs'] = len(graph_list)
        self.stats['total_nodes'] = sum(data.x.shape[0] for data in graph_list)
        
        print(f"Data loaded: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
        
        return train_list, val_list, test_list
    
    def augment_data(self, data_list, split_name: str = "train"):
        if not self.enable_augmentation:
            return data_list
        
        print(f"\n[Stage 2 & 3] Augmenting {split_name} data ({len(data_list)} graphs)")
        
        import time
        start_time = time.time()
        
        try:
            # Stage 2: Node Selection
            print(f"  [Stage 2] Selecting nodes to augment...")
            if split_name == "train" and not self.node_selector.is_fitted:
                print(f"    Fitting node selector on {len(data_list)} graphs...")
                self.node_selector.fit(data_list)
                print(f"    Node selector fitted")
            
            selected_nodes_list = [self.node_selector.select_nodes(data) for data in data_list]
            total_selected = sum(len(nodes) for nodes in selected_nodes_list)
            total_nodes = sum(data.x.shape[0] for data in data_list)
            
            print(f"    Selected {total_selected:,}/{total_nodes:,} nodes ({total_selected/total_nodes*100:.1f}%)")
            
            if total_selected == 0:
                print(f"    No nodes selected for augmentation, skipping...")
                return data_list
            
            # Stage 3: Node Augmentation
            print(f"  [Stage 3] Augmenting selected nodes...")
            print(f"    Configuration: LLM_BATCH_SIZE={self.batch_size}")
            initial_stats = self.node_augmentor.get_statistics()
            
            augmented_list = self.node_augmentor.augment_batch(
                data_list,
                selected_nodes_list,
                texts_list=None,
                verbose=True,
                batch_size=self.batch_size
            )
            
            # Print detailed statistics
            final_stats = self.node_augmentor.get_statistics()
            stats_diff = {k: final_stats[k] - initial_stats[k] for k in final_stats}
            
            elapsed = time.time() - start_time
            self.stats['augmentation_time'] += elapsed
            self.stats['augmented_nodes'] += total_selected
            
            print(f"\n  Augmentation Statistics:")
            print(f"    Nodes augmented: {stats_diff['nodes_augmented']:,}")
            print(f"    LLM API calls: {stats_diff['llm_calls']:,}")
            print(f"    Cache hits: {stats_diff['cache_hits']:,}")
            if stats_diff['llm_calls'] > 0:
                cache_rate = stats_diff['cache_hits'] / (stats_diff['llm_calls'] + stats_diff['cache_hits']) * 100
                print(f"    Cache hit rate: {cache_rate:.1f}%")
            if stats_diff['rate_limited'] > 0:
                print(f"    Rate limit events: {stats_diff['rate_limited']}")
            if stats_diff['network_errors'] > 0:
                print(f"    Network errors: {stats_diff['network_errors']}")
            if stats_diff['retries'] > 0:
                print(f"    Retries: {stats_diff['retries']}")
            print(f"    Time elapsed: {elapsed:.2f}s")
            if total_selected > 0:
                print(f"    Avg time per node: {elapsed/total_selected:.3f}s")
            
            return augmented_list
            
        except QuotaExceededError as e:
            print(f"\n❌ ERROR: {e}")
            print("Stopping execution due to API quota exceeded.")
            print("Please check your API quota and try again later.")
            raise SystemExit(1)
    
    def train_model(self, train_list, val_list, test_list, dataset_name: str, sample_ratio: float = 1.0):
        print(f"[Stage 4] Training model")
        
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
        
        train_loader = DataLoader(train_list, batch_size=self.config.BATCH_SIZE, shuffle=True, num_workers=self.config.NUM_WORKERS)
        val_loader = DataLoader(val_list, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS)
        test_loader = DataLoader(test_list, batch_size=self.config.BATCH_SIZE, shuffle=False, num_workers=self.config.NUM_WORKERS)
        
        results = self._simple_training_loop(train_loader, val_loader, test_loader, dataset_name, sample_ratio)
        
        return results
    
    def _simple_training_loop(self, train_loader, val_loader, test_loader, dataset_name, sample_ratio):
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.LEARNING_RATE, weight_decay=self.config.WEIGHT_DECAY)
        criterion = torch.nn.NLLLoss()
        
        best_val_acc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': []
        }
        
        if tqdm is None:
            print("Tip: install tqdm (`pip install tqdm`) to see live epoch progress bars.")

        progress_label = f"{self.gnn_backbone.upper()} {'SeAug' if self.enable_augmentation else 'Baseline'}"
        epoch_iter = range(self.config.NUM_EPOCHS)
        if tqdm is not None:
            epoch_iter = tqdm(
                epoch_iter,
                desc=f"Training {progress_label}",
                leave=False,
                dynamic_ncols=True
            )

        run_tag = self._build_run_tag(dataset_name, sample_ratio)
        checkpoint_path = os.path.join(self.config.SAVE_DIR, f'{run_tag}_best.pt')

        for epoch in epoch_iter:
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
                pred = output.argmax(dim=1)
                train_preds.extend(pred.cpu().numpy())
                train_labels.extend(batch.y.cpu().numpy())
            
            train_loss /= len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_precision = precision_score(train_labels, train_preds, average='binary', zero_division=0)
            train_recall = recall_score(train_labels, train_preds, average='binary', zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, average='binary', zero_division=0)
            
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
            val_acc = accuracy_score(val_labels, val_preds)
            val_precision = precision_score(val_labels, val_preds, average='binary', zero_division=0)
            val_recall = recall_score(val_labels, val_preds, average='binary', zero_division=0)
            val_f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_precision'].append(train_precision)
            history['val_precision'].append(val_precision)
            history['train_recall'].append(train_recall)
            history['val_recall'].append(val_recall)
            history['train_f1'].append(train_f1)
            history['val_f1'].append(val_f1)
            
            if tqdm is not None:
                epoch_iter.set_postfix({
                    'train_loss': f"{train_loss:.3f}",
                    'val_loss': f"{val_loss:.3f}",
                    'val_acc': f"{val_acc:.3f}",
                    'val_prec': f"{val_precision:.3f}",
                    'val_rec': f"{val_recall:.3f}",
                    'val_f1': f"{val_f1:.3f}"
                })

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{self.config.NUM_EPOCHS}: "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Rec: {train_recall:.4f}, F1: {train_f1:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}, F1: {val_f1:.4f}")
            
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training completed. Best val acc: {best_val_acc:.4f}")
        
        self.model.load_state_dict(torch.load(checkpoint_path))
        test_results = self._evaluate(test_loader, return_predictions=True)
        print("Test metrics: "
              f"Acc={test_results['accuracy']:.4f}, "
              f"Prec={test_results['precision']:.4f}, "
              f"Rec={test_results['recall']:.4f}, "
              f"F1={test_results['f1']:.4f}")
        
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
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
        
        results = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        
        if return_predictions:
            results['predictions'] = all_preds
            results['labels'] = all_labels
            results['probabilities'] = all_probs
        
        return results
    
    def _save_training_curves(self, history: dict, run_tag: str):
        if not history:
            return None
        
        num_epochs = len(history.get('train_loss', []))
        if num_epochs == 0:
            return None
        epochs = range(1, num_epochs + 1)
        
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed; skipping training curve plot.")
            return None
        
        plot_path = os.path.join(self.config.SAVE_DIR, f"{run_tag}_training_curves.png")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(epochs, history.get('train_loss', []), label='Train')
        axes[0].plot(epochs, history.get('val_loss', []), label='Val')
        axes[0].set_title('Loss vs Epoch')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        axes[1].plot(epochs, history.get('train_acc', []), label='Train')
        axes[1].plot(epochs, history.get('val_acc', []), label='Val')
        axes[1].set_title('Accuracy vs Epoch')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        fig.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Training curves saved to: {plot_path}")
        return plot_path
    
    def output_results(self, results: dict, dataset_name: str, sample_ratio: float = 1.0, save_json: bool = True, save_csv: bool = True):
        import json
        from datetime import datetime
        
        run_tag = self._build_run_tag(dataset_name, sample_ratio)
        
        print(f"\nTest Results: Acc={results['test_results']['accuracy']:.4f}, F1={results['test_results']['f1']:.4f}")
        if self.enable_augmentation:
            print(f"Augmented {self.stats['augmented_nodes']:,} nodes in {self.stats['augmentation_time']:.2f}s")
        if dataset_name.lower() == 'pheme' and self.current_events:
            print(f"Events: {', '.join(self.current_events)}")
        
        # Get augmentation statistics if available
        augmentation_stats = {}
        if self.enable_augmentation and self.node_augmentor:
            aug_stats = self.node_augmentor.get_statistics()
            augmentation_stats = {
                'llm_calls': aug_stats.get('llm_calls', 0),
                'cache_hits': aug_stats.get('cache_hits', 0),
                'quota_exceeded': aug_stats.get('quota_exceeded', 0),
                'rate_limited': aug_stats.get('rate_limited', 0),
                'network_errors': aug_stats.get('network_errors', 0),
                'retries': aug_stats.get('retries', 0)
            }
            # Calculate cache hit rate
            total_requests = augmentation_stats['llm_calls'] + augmentation_stats['cache_hits']
            if total_requests > 0:
                augmentation_stats['cache_hit_rate'] = augmentation_stats['cache_hits'] / total_requests * 100
            else:
                augmentation_stats['cache_hit_rate'] = 0.0
            
            # Print augmentation statistics summary
            if augmentation_stats['llm_calls'] > 0 or augmentation_stats['cache_hits'] > 0:
                print(f"\nAugmentation Statistics:")
                print(f"  LLM API calls: {augmentation_stats['llm_calls']:,}")
                print(f"  Cache hits: {augmentation_stats['cache_hits']:,}")
                print(f"  Cache hit rate: {augmentation_stats['cache_hit_rate']:.1f}%")
                if augmentation_stats['retries'] > 0:
                    print(f"  Retries: {augmentation_stats['retries']:,}")
                if augmentation_stats['rate_limited'] > 0:
                    print(f"  Rate limit events: {augmentation_stats['rate_limited']:,}")
                if augmentation_stats['network_errors'] > 0:
                    print(f"  Network errors: {augmentation_stats['network_errors']:,}")
                if augmentation_stats['quota_exceeded'] > 0:
                    print(f"  Quota exceeded: {augmentation_stats['quota_exceeded']:,}")
        
        output_files = {}
        history_plot = self._save_training_curves(results.get('history'), run_tag)
        if history_plot:
            output_files['training_plot'] = history_plot
        
        if save_json:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_path = os.path.join(self.config.SAVE_DIR, f'{run_tag}_results_{timestamp}.json')
            json_results = {
                'dataset': dataset_name,
                'dataset_metadata': {
                    'events': self.current_events if dataset_name.lower() == 'pheme' else None,
                    'event_suffix': self.current_event_suffix if dataset_name.lower() == 'pheme' else None
                },
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
                'augmentation_statistics': augmentation_stats if augmentation_stats else None,
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
            print(f"Results saved to: {json_path}")
        
        if save_csv:
            csv_path = os.path.join(self.config.SAVE_DIR, "results_summary.csv")
            file_exists = os.path.isfile(csv_path)
            fieldnames = ["dataset", "model_type", "enable_augmentation", "node_strategy", "fusion_strategy",
                         "augmentation_ratio", "gnn_backbone", "sample_ratio", "total_graphs", "total_nodes",
                         "augmented_nodes", "augmentation_time", "llm_calls", "cache_hits", "cache_hit_rate",
                         "quota_exceeded", "rate_limited", "network_errors", "retries", "accuracy", "precision", "recall", "f1",
                         "pheme_events"]
            
            model_desc = "SeAug with LLM" if self.enable_augmentation else "Baseline (No Augmentation)"
            
            existing_rows = []
            needs_header_upgrade = False
            if file_exists:
                with open(csv_path, mode="r", newline="") as existing_file:
                    reader = csv.DictReader(existing_file)
                    header = reader.fieldnames or []
                    # Check if new augmentation stats columns are missing
                    new_columns = ["llm_calls", "cache_hits", "cache_hit_rate", "quota_exceeded", "rate_limited", "network_errors", "retries"]
                    if any(col not in header for col in new_columns):
                        needs_header_upgrade = True
                        existing_rows = list(reader)
            if needs_header_upgrade:
                with open(csv_path, mode="w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in existing_rows:
                        # Add missing columns with default values
                        for col in fieldnames:
                            if col not in row:
                                row[col] = ""
                        writer.writerow(row)
                file_exists = True  # Header upgraded; treat as existing file for append
            
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                
                row_data = {
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
                    "llm_calls": augmentation_stats.get("llm_calls", 0),
                    "cache_hits": augmentation_stats.get("cache_hits", 0),
                    "cache_hit_rate": round(augmentation_stats.get("cache_hit_rate", 0.0), 2),
                    "quota_exceeded": augmentation_stats.get("quota_exceeded", 0),
                    "rate_limited": augmentation_stats.get("rate_limited", 0),
                    "network_errors": augmentation_stats.get("network_errors", 0),
                    "retries": augmentation_stats.get("retries", 0),
                    "accuracy": results["test_results"]["accuracy"],
                    "precision": results["test_results"]["precision"],
                    "recall": results["test_results"]["recall"],
                    "f1": results["test_results"]["f1"],
                    "pheme_events": ", ".join(self.current_events) if dataset_name.lower() == "pheme" and self.current_events else "",
                }
                writer.writerow(row_data)
            output_files['csv'] = csv_path
        
        return output_files
    
    def run(self, dataset_name: str, sample_ratio: float = 1.0):
        print(f"\nRunning SeAug Pipeline on {dataset_name}")
        
        self.config.create_dirs()
        self.setup_components()
        
        train_list, val_list, test_list = self.process_data(dataset_name, sample_ratio)
        train_list = self.augment_data(train_list, "train")
        val_list = self.augment_data(val_list, "val")
        test_list = self.augment_data(test_list, "test")
        
        results = self.train_model(train_list, val_list, test_list, dataset_name, sample_ratio)
        self.output_results(results, dataset_name, sample_ratio)
        
        return results


def main():
    parser = argparse.ArgumentParser(description="SeAug Framework for Rumor Detection")
    parser.add_argument('--dataset', type=str, default='Twitter15', choices=['Twitter15', 'Twitter16', 'Weibo', 'PHEME'])
    parser.add_argument('--sample_ratio', type=float, default=1.0)
    parser.add_argument('--enable_augmentation', action='store_true')
    parser.add_argument('--node_strategy', type=str, default='hybrid', choices=['uncertainty', 'importance', 'hybrid'])
    parser.add_argument('--fusion_strategy', type=str, default='concat', choices=['concat', 'weighted', 'gated', 'attention'])
    parser.add_argument('--augmentation_ratio', type=float, default=0.3)
    parser.add_argument('--gnn_backbone', type=str, default='gcn', choices=['gcn', 'gat'])
    parser.add_argument('--batch_size', type=int, default=50)
    
    args = parser.parse_args()
    
    pipeline = SeAugPipeline(
        enable_augmentation=args.enable_augmentation,
        node_selection_strategy=args.node_strategy,
        fusion_strategy=args.fusion_strategy,
        augmentation_ratio=args.augmentation_ratio,
        gnn_backbone=args.gnn_backbone,
        batch_size=args.batch_size
    )
    
    return pipeline.run(dataset_name=args.dataset, sample_ratio=args.sample_ratio)


if __name__ == '__main__':
    main()

