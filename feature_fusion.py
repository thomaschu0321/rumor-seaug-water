"""
Feature Fusion Module for TAPE Framework (Stage 4a)

This module fuses baseline features (TF-IDF) with augmented features (LM embeddings)
to create hybrid node representations for GNN training.

Fusion Strategies:
1. Concatenation: [x_base || x_aug]
2. Weighted Sum: alpha * x_base + beta * x_aug
3. Attention-based: Learn adaptive weights
4. Gated Fusion: Use gates to control information flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from typing import List, Optional


class FeatureFusion(nn.Module):
    """
    Feature fusion layer for combining baseline and augmented features
    """
    
    def __init__(
        self,
        baseline_dim: int,
        augmented_dim: int,
        output_dim: int = None,
        strategy: str = "concat",
        dropout: float = 0.1
    ):
        """
        Initialize Feature Fusion
        
        Args:
            baseline_dim: Dimension of baseline features (e.g., 1000 for TF-IDF)
            augmented_dim: Dimension of augmented features (e.g., 384/768 for BERT)
            output_dim: Output dimension (if None, determined by strategy)
            strategy: Fusion strategy
                - "concat": Simple concatenation
                - "weighted": Learnable weighted sum (requires projection)
                - "gated": Gated fusion with learnable gates
                - "attention": Attention-based fusion
            dropout: Dropout rate
        """
        super(FeatureFusion, self).__init__()
        
        self.baseline_dim = baseline_dim
        self.augmented_dim = augmented_dim
        self.strategy = strategy
        self.dropout = dropout
        
        # Determine output dimension
        if output_dim is None:
            if strategy == "concat":
                output_dim = baseline_dim + augmented_dim
            else:
                output_dim = max(baseline_dim, augmented_dim)
        
        self.output_dim = output_dim
        
        # Build fusion layers based on strategy
        if strategy == "concat":
            # Simple concatenation + optional projection
            if output_dim != baseline_dim + augmented_dim:
                self.projection = nn.Linear(baseline_dim + augmented_dim, output_dim)
            else:
                self.projection = None
        
        elif strategy == "weighted":
            # Project both to same dimension, then weighted sum
            self.base_proj = nn.Linear(baseline_dim, output_dim)
            self.aug_proj = nn.Linear(augmented_dim, output_dim)
            self.weight_alpha = nn.Parameter(torch.tensor(0.5))  # Learnable weight
        
        elif strategy == "gated":
            # Gated fusion
            self.base_proj = nn.Linear(baseline_dim, output_dim)
            self.aug_proj = nn.Linear(augmented_dim, output_dim)
            self.gate = nn.Sequential(
                nn.Linear(baseline_dim + augmented_dim, output_dim),
                nn.Sigmoid()
            )
        
        elif strategy == "attention":
            # Attention-based fusion
            self.base_proj = nn.Linear(baseline_dim, output_dim)
            self.aug_proj = nn.Linear(augmented_dim, output_dim)
            self.attention = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Tanh(),
                nn.Linear(output_dim, 2),  # 2 attention scores
                nn.Softmax(dim=-1)
            )
        
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        
        self.dropout_layer = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(output_dim)
    
    def forward(
        self,
        x_base: torch.Tensor,
        x_aug: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Fuse baseline and augmented features
        
        Args:
            x_base: Baseline features [num_nodes, baseline_dim]
            x_aug: Augmented features [num_nodes, augmented_dim]
            mask: Boolean mask [num_nodes] indicating which nodes have augmented features
                  If None, assumes all nodes have augmented features
        
        Returns:
            Fused features [num_nodes, output_dim]
        """
        if mask is None:
            mask = torch.ones(x_base.shape[0], dtype=torch.bool, device=x_base.device)
        
        # Handle nodes without augmented features
        # For these nodes, use only baseline features (or zero augmented features)
        if x_aug.shape[0] < x_base.shape[0]:
            # Pad x_aug with zeros
            padding = torch.zeros(
                x_base.shape[0] - x_aug.shape[0],
                x_aug.shape[1],
                device=x_aug.device,
                dtype=x_aug.dtype
            )
            x_aug = torch.cat([x_aug, padding], dim=0)
        
        # Apply fusion strategy
        if self.strategy == "concat":
            x_fused = torch.cat([x_base, x_aug], dim=1)
            if self.projection is not None:
                x_fused = self.projection(x_fused)
        
        elif self.strategy == "weighted":
            x_base_proj = self.base_proj(x_base)
            x_aug_proj = self.aug_proj(x_aug)
            
            # Learnable weighted sum
            alpha = torch.sigmoid(self.weight_alpha)  # Ensure in [0, 1]
            x_fused = alpha * x_base_proj + (1 - alpha) * x_aug_proj
            
            # For non-augmented nodes, use only baseline
            x_fused[~mask] = x_base_proj[~mask]
        
        elif self.strategy == "gated":
            x_base_proj = self.base_proj(x_base)
            x_aug_proj = self.aug_proj(x_aug)
            
            # Compute gate
            gate_input = torch.cat([x_base, x_aug], dim=1)
            gate_values = self.gate(gate_input)
            
            # Apply gate
            x_fused = gate_values * x_aug_proj + (1 - gate_values) * x_base_proj
            
            # For non-augmented nodes, use only baseline
            x_fused[~mask] = x_base_proj[~mask]
        
        elif self.strategy == "attention":
            x_base_proj = self.base_proj(x_base)
            x_aug_proj = self.aug_proj(x_aug)
            
            # Compute attention weights
            combined = torch.cat([x_base_proj, x_aug_proj], dim=1)
            att_weights = self.attention(combined)  # [num_nodes, 2]
            
            # Apply attention
            x_fused = att_weights[:, 0:1] * x_base_proj + att_weights[:, 1:2] * x_aug_proj
            
            # For non-augmented nodes, use only baseline
            x_fused[~mask] = x_base_proj[~mask]
        
        # Apply dropout and batch norm
        x_fused = self.dropout_layer(x_fused)
        x_fused = self.batch_norm(x_fused)
        
        return x_fused


def prepare_fused_data(
    data: Data,
    fusion_layer: FeatureFusion = None,
    strategy: str = "concat"
) -> Data:
    """
    Prepare a Data object with fused features
    
    Args:
        data: PyG Data object with x (baseline) and x_aug (augmented) fields
        fusion_layer: Fusion layer instance (if None, use simple concatenation)
        strategy: Fusion strategy if fusion_layer is None
    
    Returns:
        Data object with fused features in x field
    """
    if not hasattr(data, 'x_aug'):
        # No augmented features, return original
        return data
    
    x_base = data.x
    x_aug = data.x_aug
    mask = data.augmented_node_mask if hasattr(data, 'augmented_node_mask') else None
    
    if fusion_layer is None:
        # Simple concatenation fallback
        if strategy == "concat":
            x_fused = torch.cat([x_base, x_aug], dim=1)
        else:
            raise ValueError("Fusion layer required for non-concat strategies")
    else:
        # Use fusion layer
        x_fused = fusion_layer(x_base, x_aug, mask)
    
    # Create new data object
    data_fused = Data(
        x=x_fused,
        edge_index=data.edge_index,
        y=data.y if hasattr(data, 'y') else None,
        batch=data.batch if hasattr(data, 'batch') else None
    )
    
    # Copy other attributes
    for key in data.keys():
        if key not in ['x', 'x_aug', 'edge_index', 'y', 'batch']:
            data_fused[key] = data[key]
    
    return data_fused


def prepare_fused_batch(
    data_list: List[Data],
    fusion_layer: FeatureFusion = None,
    strategy: str = "concat"
) -> List[Data]:
    """
    Prepare a batch of Data objects with fused features
    
    Args:
        data_list: List of PyG Data objects
        fusion_layer: Fusion layer instance
        strategy: Fusion strategy
    
    Returns:
        List of Data objects with fused features
    """
    fused_list = []
    
    for data in data_list:
        data_fused = prepare_fused_data(data, fusion_layer, strategy)
        fused_list.append(data_fused)
    
    return fused_list


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns to use augmented features only when beneficial
    """
    
    def __init__(
        self,
        baseline_dim: int,
        augmented_dim: int,
        hidden_dim: int = 128
    ):
        """
        Initialize Adaptive Fusion
        
        Args:
            baseline_dim: Baseline feature dimension
            augmented_dim: Augmented feature dimension
            hidden_dim: Hidden dimension for decision network
        """
        super(AdaptiveFusion, self).__init__()
        
        self.baseline_dim = baseline_dim
        self.augmented_dim = augmented_dim
        self.output_dim = baseline_dim + augmented_dim
        
        # Decision network: decides whether to use augmented features
        self.decision_net = nn.Sequential(
            nn.Linear(baseline_dim + augmented_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x_base: torch.Tensor,
        x_aug: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Adaptive fusion
        
        Args:
            x_base: Baseline features
            x_aug: Augmented features
            mask: Boolean mask for augmented nodes
        
        Returns:
            Fused features
        """
        # Concatenate for decision
        x_concat = torch.cat([x_base, x_aug], dim=1)
        
        # Compute confidence scores
        confidence = self.decision_net(x_concat)  # [num_nodes, 1]
        
        # Apply confidence as gate
        # High confidence -> use augmented; Low confidence -> use baseline only
        x_base_repeated = x_base.repeat(1, 2) if x_base.shape[1] * 2 == self.output_dim else x_base
        x_fused = confidence * x_concat + (1 - confidence) * x_base_repeated
        
        return x_fused


if __name__ == '__main__':
    print("="*70)
    print("Testing Feature Fusion")
    print("="*70)
    
    # Test parameters
    batch_size = 32
    num_nodes = 10
    baseline_dim = 1000
    augmented_dim = 384
    
    # Create dummy data
    x_base = torch.randn(batch_size * num_nodes, baseline_dim)
    x_aug = torch.randn(batch_size * num_nodes, augmented_dim)
    mask = torch.rand(batch_size * num_nodes) > 0.5  # 50% augmented
    
    # Test different fusion strategies
    strategies = ["concat", "weighted", "gated", "attention"]
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Testing Strategy: {strategy}")
        print(f"{'='*70}")
        
        # Create fusion layer
        fusion = FeatureFusion(
            baseline_dim=baseline_dim,
            augmented_dim=augmented_dim,
            strategy=strategy
        )
        
        print(f"Fusion layer created:")
        print(f"  Input: baseline({baseline_dim}) + augmented({augmented_dim})")
        print(f"  Output: {fusion.output_dim}")
        print(f"  Strategy: {strategy}")
        
        # Apply fusion
        x_fused = fusion(x_base, x_aug, mask)
        
        print(f"\nFusion results:")
        print(f"  Baseline shape: {x_base.shape}")
        print(f"  Augmented shape: {x_aug.shape}")
        print(f"  Fused shape: {x_fused.shape}")
        print(f"  Augmented nodes: {mask.sum().item()}/{len(mask)}")
        
        # Check gradients
        loss = x_fused.mean()
        loss.backward()
        print(f"  ✓ Gradients computed successfully")
    
    # Test with PyG Data objects
    print(f"\n{'='*70}")
    print("Testing with PyG Data")
    print(f"{'='*70}")
    
    from torch_geometric.data import Data
    
    # Create data with augmented features
    data = Data(
        x=torch.randn(10, 1000),
        x_aug=torch.randn(10, 384),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        y=torch.tensor([1]),
        augmented_node_mask=torch.tensor([True, True, False, True, False, 
                                         False, True, False, False, True])
    )
    
    print(f"Original data:")
    print(f"  x: {data.x.shape}")
    print(f"  x_aug: {data.x_aug.shape}")
    print(f"  Augmented nodes: {data.augmented_node_mask.sum()}/{len(data.augmented_node_mask)}")
    
    # Apply fusion
    data_fused = prepare_fused_data(data, strategy="concat")
    
    print(f"\nFused data:")
    print(f"  x: {data_fused.x.shape}")
    print(f"  Feature dimension: {data_fused.x.shape[1]}")
    
    print("\n" + "="*70)
    print("✓ Feature Fusion test completed!")
    print("="*70)



