"""
Quick Test Script for GAT Implementation

This script verifies that GAT backbone works correctly with SeAug framework.
"""

import torch
from torch_geometric.data import Data, Batch

from model_seaug import get_seaug_model


def test_gat_model():
    """Test GAT model creation and forward pass"""
    
    print("="*70)
    print("Testing GAT Implementation")
    print("="*70)
    
    # Create dummy graph data
    num_nodes = 10
    baseline_dim = 768
    augmented_dim = 384
    
    data1 = Data(
        x=torch.randn(num_nodes, baseline_dim),
        x_aug=torch.randn(num_nodes, augmented_dim),
        edge_index=torch.tensor([[0, 0, 1, 2, 3, 4], 
                                 [1, 2, 3, 4, 5, 6]], dtype=torch.long),
        y=torch.tensor([1]),
        augmented_node_mask=torch.tensor([True, True, False, True, False, 
                                         False, True, False, False, True])
    )
    
    data2 = Data(
        x=torch.randn(8, baseline_dim),
        x_aug=torch.randn(8, augmented_dim),
        edge_index=torch.tensor([[0, 0, 1, 2], 
                                 [1, 2, 3, 4]], dtype=torch.long),
        y=torch.tensor([0]),
        augmented_node_mask=torch.tensor([False, True, True, False, 
                                         True, False, False, True])
    )
    
    batch = Batch.from_data_list([data1, data2])
    
    print(f"\nTest Data:")
    print(f"  Batch size: 2")
    print(f"  Total nodes: {batch.x.shape[0]}")
    print(f"  Baseline features: {batch.x.shape}")
    print(f"  Augmented features: {batch.x_aug.shape}")
    
    # Test GCN backbone
    print(f"\n{'='*70}")
    print("Testing GCN Backbone")
    print(f"{'='*70}")
    
    model_gcn = get_seaug_model(
        model_type="seaug",
        gnn_backbone="gcn",
        baseline_dim=baseline_dim,
        augmented_dim=augmented_dim,
        hidden_dim=64,
        num_classes=2,
        fusion_strategy="concat"
    )
    
    output_gcn = model_gcn(batch)
    print(f"\nGCN Output shape: {output_gcn.shape}")
    print(f"Expected: [2, 2] (batch_size, num_classes)")
    assert output_gcn.shape == (2, 2), "GCN output shape mismatch!"
    print("✓ GCN test passed")
    
    # Test GAT backbone
    print(f"\n{'='*70}")
    print("Testing GAT Backbone")
    print(f"{'='*70}")
    
    model_gat = get_seaug_model(
        model_type="seaug",
        gnn_backbone="gat",
        baseline_dim=baseline_dim,
        augmented_dim=augmented_dim,
        hidden_dim=64,
        num_classes=2,
        fusion_strategy="concat",
        gat_heads=4
    )
    
    output_gat = model_gat(batch)
    print(f"\nGAT Output shape: {output_gat.shape}")
    print(f"Expected: [2, 2] (batch_size, num_classes)")
    assert output_gat.shape == (2, 2), "GAT output shape mismatch!"
    print("✓ GAT test passed")
    
    # Compare parameter counts
    gcn_params = sum(p.numel() for p in model_gcn.parameters())
    gat_params = sum(p.numel() for p in model_gat.parameters())
    
    print(f"\n{'='*70}")
    print("Model Comparison")
    print(f"{'='*70}")
    print(f"GCN parameters: {gcn_params:,}")
    print(f"GAT parameters: {gat_params:,}")
    print(f"Difference: {abs(gat_params - gcn_params):,} parameters")
    
    print(f"\n{'='*70}")
    print("✓ All tests passed!")
    print("GAT implementation is working correctly")
    print(f"{'='*70}")


if __name__ == '__main__':
    test_gat_model()

