"""SeAug Model Implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from feature_fusion import FeatureFusion


class SeAugRumorGNN(nn.Module):
    
    def __init__(
        self,
        baseline_dim: int = 768,
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
        
        if self.gnn_backbone not in ['gcn', 'gat']:
            raise ValueError(f"Unsupported GNN backbone: {gnn_backbone}. Choose 'gcn' or 'gat'.")
        
        if use_fusion:
            self.fusion = FeatureFusion(
                baseline_dim=baseline_dim,
                augmented_dim=augmented_dim,
                output_dim=hidden_dim,
                strategy=fusion_strategy,
                dropout=dropout
            )
            input_dim = hidden_dim
        else:
            self.fusion = None
            input_dim = baseline_dim
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if self.gnn_backbone == 'gcn':
            self.convs.append(GCNConv(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            for _ in range(num_gnn_layers - 1):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        elif self.gnn_backbone == 'gat':
            self.convs.append(GATConv(
                input_dim, 
                hidden_dim // gat_heads,
                heads=gat_heads,
                concat=True,
                dropout=dropout
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            for _ in range(num_gnn_layers - 1):
                self.convs.append(GATConv(
                    hidden_dim,
                    hidden_dim // gat_heads,
                    heads=gat_heads,
                    concat=True,
                    dropout=dropout
                ))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        edge_index = data.edge_index
        batch = data.batch
        
        if self.use_fusion and hasattr(data, 'x_aug'):
            x_base = data.x
            x_aug = data.x_aug
            mask = data.augmented_node_mask if hasattr(data, 'augmented_node_mask') else None
            x = self.fusion(x_base, x_aug, mask)
        else:
            x = data.x
        
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)


def get_seaug_model(
    model_type: str = "seaug",
    gnn_backbone: str = "gcn",
    baseline_dim: int = 768,
    augmented_dim: int = 384,
    hidden_dim: int = 64,
    num_classes: int = 2,
    dropout: float = 0.3,
    fusion_strategy: str = "concat",
    gat_heads: int = 4
):
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
