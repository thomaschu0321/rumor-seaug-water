import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Feature fusion layer for combining baseline and augmented features"""
    
    def __init__(
        self,
        baseline_dim: int,
        augmented_dim: int,
        output_dim: int = None,
        strategy: str = "concat",
        dropout: float = 0.1
    ):
        super(FeatureFusion, self).__init__()
        
        self.baseline_dim = baseline_dim
        self.augmented_dim = augmented_dim
        self.strategy = strategy
        self.dropout = dropout
        
        if output_dim is None:
            if strategy == "concat":
                output_dim = baseline_dim + augmented_dim
            else:
                output_dim = max(baseline_dim, augmented_dim)
        
        self.output_dim = output_dim
        
        if strategy == "concat":
            if output_dim != baseline_dim + augmented_dim:
                self.projection = nn.Linear(baseline_dim + augmented_dim, output_dim)
            else:
                self.projection = None
        
        elif strategy == "weighted":
            self.base_proj = nn.Linear(baseline_dim, output_dim)
            self.aug_proj = nn.Linear(augmented_dim, output_dim)
            self.weight_alpha = nn.Parameter(torch.tensor(0.5))
        
        elif strategy == "gated":
            self.base_proj = nn.Linear(baseline_dim, output_dim)
            self.aug_proj = nn.Linear(augmented_dim, output_dim)
            self.gate = nn.Sequential(
                nn.Linear(baseline_dim + augmented_dim, output_dim),
                nn.Sigmoid()
            )
        
        elif strategy == "attention":
            self.base_proj = nn.Linear(baseline_dim, output_dim)
            self.aug_proj = nn.Linear(augmented_dim, output_dim)
            self.attention = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Tanh(),
                nn.Linear(output_dim, 2),
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
        if mask is None:
            mask = torch.ones(x_base.shape[0], dtype=torch.bool, device=x_base.device)
        
        if x_aug.shape[0] < x_base.shape[0]:
            padding = torch.zeros(
                x_base.shape[0] - x_aug.shape[0],
                x_aug.shape[1],
                device=x_aug.device,
                dtype=x_aug.dtype
            )
            x_aug = torch.cat([x_aug, padding], dim=0)
        
        if self.strategy == "concat":
            x_fused = torch.cat([x_base, x_aug], dim=1)
            if self.projection is not None:
                x_fused = self.projection(x_fused)
        
        elif self.strategy == "weighted":
            x_base_proj = self.base_proj(x_base)
            x_aug_proj = self.aug_proj(x_aug)
            
            alpha = torch.sigmoid(self.weight_alpha)
            x_fused = alpha * x_base_proj + (1 - alpha) * x_aug_proj
            x_fused[~mask] = x_base_proj[~mask]
        
        elif self.strategy == "gated":
            x_base_proj = self.base_proj(x_base)
            x_aug_proj = self.aug_proj(x_aug)
            gate_input = torch.cat([x_base, x_aug], dim=1)
            gate_values = self.gate(gate_input)
            x_fused = gate_values * x_aug_proj + (1 - gate_values) * x_base_proj
            x_fused[~mask] = x_base_proj[~mask]
        
        elif self.strategy == "attention":
            x_base_proj = self.base_proj(x_base)
            x_aug_proj = self.aug_proj(x_aug)
            combined = torch.cat([x_base_proj, x_aug_proj], dim=1)
            att_weights = self.attention(combined)
            x_fused = att_weights[:, 0:1] * x_base_proj + att_weights[:, 1:2] * x_aug_proj
            x_fused[~mask] = x_base_proj[~mask]
        
        x_fused = self.dropout_layer(x_fused)
        x_fused = self.batch_norm(x_fused)
        
        return x_fused
