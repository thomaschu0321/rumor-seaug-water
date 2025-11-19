"""
Enhanced GNN Model with Feature Fusion for SeAug Framework (Stage 4b)

This model supports multiple GNN backbones (GCN, GAT) with feature fusion
to demonstrate the generalizability of the SeAug framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from typing import Optional

from feature_fusion import FeatureFusion


class SeAugRumorGNN(nn.Module):
    """
    GNN model with integrated feature fusion for SeAug framework
    Supports multiple GNN backbones (GCN, GAT) to demonstrate generalizability
    
    Architecture:
    1. Feature Fusion Layer (optional): Fuse baseline + augmented features
    2. GNN Layers: Process fused features (GCN or GAT)
    3. Graph Pooling: Aggregate to graph-level
    4. Classification: Final prediction
    """
    
    def __init__(
        self,
        baseline_dim: int = 1000,
        augmented_dim: int = 384,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_fusion: bool = True,
        fusion_strategy: str = "concat",
        num_gnn_layers: int = 2,
        gnn_backbone: str = "gcn",
        gat_heads: int = 4
    ):
        """
        Initialize SeAug GNN Model
        
        Args:
            baseline_dim: Dimension of baseline features (TF-IDF)
            augmented_dim: Dimension of augmented features (LM embeddings)
            hidden_dim: Hidden dimension for GNN layers
            num_classes: Number of output classes
            dropout: Dropout rate
            use_fusion: Whether to use feature fusion
            fusion_strategy: Fusion strategy ('concat', 'weighted', 'gated', 'attention')
            num_gnn_layers: Number of GNN layers
            gnn_backbone: GNN backbone type ('gcn' or 'gat')
            gat_heads: Number of attention heads for GAT (only used if backbone='gat')
        """
        super(SeAugRumorGNN, self).__init__()
        
        self.baseline_dim = baseline_dim
        self.augmented_dim = augmented_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_fusion = use_fusion
        self.num_gnn_layers = num_gnn_layers
        self.gnn_backbone = gnn_backbone.lower()
        self.gat_heads = gat_heads
        
        # Validate backbone
        if self.gnn_backbone not in ['gcn', 'gat']:
            raise ValueError(f"Unsupported GNN backbone: {gnn_backbone}. Choose 'gcn' or 'gat'.")
        
        # Feature fusion layer
        if use_fusion:
            self.fusion = FeatureFusion(
                baseline_dim=baseline_dim,
                augmented_dim=augmented_dim,
                output_dim=hidden_dim,  # Project to hidden_dim
                strategy=fusion_strategy,
                dropout=dropout
            )
            input_dim = hidden_dim
        else:
            # No fusion, use only baseline features
            self.fusion = None
            input_dim = baseline_dim
        
        # GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Build GNN layers based on backbone type
        if self.gnn_backbone == 'gcn':
            # GCN layers
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            for _ in range(num_gnn_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        elif self.gnn_backbone == 'gat':
            # GAT layers
            # First layer: multi-head attention with concatenation
            self.convs.append(GATConv(
                input_dim, 
                hidden_dim // gat_heads,  # Each head outputs hidden_dim/heads dimensions
                heads=gat_heads,
                concat=True,
                dropout=dropout
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            # Additional layers
            for _ in range(num_gnn_layers - 1):
                self.convs.append(GATConv(
                    hidden_dim,
                    hidden_dim // gat_heads,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout
                ))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        print(f"SeAug GNN Model initialized")
        print(f"  GNN Backbone: {self.gnn_backbone.upper()}")
        if self.gnn_backbone == 'gat':
            print(f"  GAT Heads: {gat_heads}")
        print(f"  Baseline dim: {baseline_dim}")
        print(f"  Augmented dim: {augmented_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Use fusion: {use_fusion}")
        if use_fusion:
            print(f"  Fusion strategy: {fusion_strategy}")
            print(f"  Fused input dim: {input_dim}")
        print(f"  GNN layers: {num_gnn_layers}")
    
    def forward(self, data):
        """
        Forward pass
        
        Args:
            data: PyG Data object
                Required: x, edge_index, batch
                Optional: x_aug, augmented_node_mask
        
        Returns:
            Log-softmax logits [batch_size, num_classes]
        """
        edge_index = data.edge_index
        batch = data.batch
        
        # Step 1: Feature Fusion (if enabled and augmented features available)
        if self.use_fusion and hasattr(data, 'x_aug'):
            x_base = data.x
            x_aug = data.x_aug
            mask = data.augmented_node_mask if hasattr(data, 'augmented_node_mask') else None
            x = self.fusion(x_base, x_aug, mask)
        else:
            # Use only baseline features
            x = data.x
        
        # Step 2: GCN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Step 3: Graph-level pooling
        x = global_mean_pool(x, batch)
        
        # Step 4: Classification
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)
    
    def get_embeddings(self, data):
        """
        Get graph embeddings (before classification layer)
        
        Args:
            data: PyG Data object
        
        Returns:
            Graph embeddings [batch_size, hidden_dim]
        """
        edge_index = data.edge_index
        batch = data.batch
        
        # Feature fusion
        if self.use_fusion and hasattr(data, 'x_aug'):
            x_base = data.x
            x_aug = data.x_aug
            mask = data.augmented_node_mask if hasattr(data, 'augmented_node_mask') else None
            x = self.fusion(x_base, x_aug, mask)
        else:
            x = data.x
        
        # GCN layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        
        # Graph pooling
        x = global_mean_pool(x, batch)
        
        return x
    
    def predict_with_confidence(self, data):
        """
        Predict with confidence scores
        
        Args:
            data: PyG Data object
        
        Returns:
            predictions: Predicted classes [batch_size]
            confidences: Confidence scores [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probs = torch.exp(logits)
            confidences, predictions = torch.max(probs, dim=1)
        
        return predictions, confidences


class HybridSeAugGCN(nn.Module):
    """
    Hybrid model that can switch between baseline-only and fusion modes
    
    Useful for comparing performance with/without augmentation
    """
    
    def __init__(
        self,
        baseline_dim: int = 1000,
        augmented_dim: int = 384,
        hidden_dim: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        fusion_strategy: str = "concat"
    ):
        super(HybridSeAugGCN, self).__init__()
        
        # Baseline-only branch
        self.baseline_branch = SeAugRumorGNN(
            baseline_dim=baseline_dim,
            augmented_dim=augmented_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_fusion=False
        )
        
        # Fusion branch
        self.fusion_branch = SeAugRumorGNN(
            baseline_dim=baseline_dim,
            augmented_dim=augmented_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_fusion=True,
            fusion_strategy=fusion_strategy
        )
        
        self.mode = "fusion"  # "baseline" or "fusion"
    
    def forward(self, data):
        """Forward pass using current mode"""
        if self.mode == "baseline":
            return self.baseline_branch(data)
        else:
            return self.fusion_branch(data)
    
    def set_mode(self, mode: str):
        """
        Set prediction mode
        
        Args:
            mode: "baseline" or "fusion"
        """
        if mode not in ["baseline", "fusion"]:
            raise ValueError(f"Invalid mode: {mode}. Use 'baseline' or 'fusion'.")
        self.mode = mode
    
    def compare_modes(self, data):
        """
        Compare predictions from both modes
        
        Args:
            data: PyG Data object
        
        Returns:
            dict with results from both modes
        """
        self.eval()
        with torch.no_grad():
            baseline_logits = self.baseline_branch(data)
            fusion_logits = self.fusion_branch(data)
            
            baseline_probs = torch.exp(baseline_logits)
            fusion_probs = torch.exp(fusion_logits)
            
            baseline_conf, baseline_pred = torch.max(baseline_probs, dim=1)
            fusion_conf, fusion_pred = torch.max(fusion_probs, dim=1)
        
        return {
            'baseline': {
                'predictions': baseline_pred,
                'confidences': baseline_conf,
                'logits': baseline_logits
            },
            'fusion': {
                'predictions': fusion_pred,
                'confidences': fusion_conf,
                'logits': fusion_logits
            }
        }


def get_seaug_model(
    model_type: str = "seaug",
    gnn_backbone: str = "gcn",
    baseline_dim: int = 1000,
    augmented_dim: int = 384,
    hidden_dim: int = 64,
    num_classes: int = 2,
    dropout: float = 0.3,
    fusion_strategy: str = "concat",
    gat_heads: int = 4
):
    """
    Factory function to create SeAug models with different GNN backbones
    
    Args:
        model_type: Type of model
            - "seaug": Standard SeAug GNN with fusion
            - "baseline": GNN without fusion (baseline only)
        gnn_backbone: GNN backbone type ('gcn' or 'gat')
        baseline_dim: Baseline feature dimension
        augmented_dim: Augmented feature dimension
        hidden_dim: Hidden dimension
        num_classes: Number of classes
        dropout: Dropout rate
        fusion_strategy: Fusion strategy
        gat_heads: Number of attention heads for GAT
    
    Returns:
        Model instance
    """
    if model_type == "seaug":
        return SeAugRumorGNN(
            baseline_dim=baseline_dim,
            augmented_dim=augmented_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_fusion=True,
            fusion_strategy=fusion_strategy,
            gnn_backbone=gnn_backbone,
            gat_heads=gat_heads
        )
    
    elif model_type == "baseline":
        return SeAugRumorGNN(
            baseline_dim=baseline_dim,
            augmented_dim=augmented_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            use_fusion=False,
            gnn_backbone=gnn_backbone,
            gat_heads=gat_heads
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Backward compatibility: keep old names as aliases
class SeAugRumorGCN(SeAugRumorGNN):
    """
    Backward compatibility wrapper for SeAugRumorGNN with GCN backbone
    """
    def __init__(self, **kwargs):
        kwargs['gnn_backbone'] = 'gcn'
        super().__init__(**kwargs)


# Legacy TAPE names for backward compatibility
TAPERumorGNN = SeAugRumorGNN
TAPERumorGCN = SeAugRumorGCN
HybridTAPEGCN = HybridSeAugGCN
get_tape_model = get_seaug_model

