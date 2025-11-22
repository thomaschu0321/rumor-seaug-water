"""
Node Selection Module for SeAug Framework (Phase 2)

This module implements adaptive node selection using DBSCAN clustering
to identify outlier/uncertain nodes that should be augmented with LLM.

Method (matching paper):
- Apply DBSCAN (sklearn.cluster.DBSCAN) to X_initial (BERT features)
- Identify outlier nodes (DBSCAN label = -1) as uncertain nodes
- Hypothesis: Semantically outlying tweets (e.g., sarcastic replies, 
  misinformation injections) are critical to graph classification
"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


def _to_numpy(x):
    """Convert tensor to numpy array if needed"""
    return x.numpy() if isinstance(x, torch.Tensor) else x


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range using min-max scaling"""
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


class NodeSelector:
    """
    Select nodes for augmentation based on various criteria
    """
    
    def __init__(
        self,
        strategy: str = "uncertainty",
        top_k_ratio: float = 0.3,
        min_nodes: int = 1,
        max_nodes: int = 10
    ):
        """
        Initialize Node Selector
        
        Args:
            strategy: Selection strategy
                - "uncertainty": Select nodes with uncertain features
                - "importance": Select structurally important nodes
                - "hybrid": Combine multiple strategies
            top_k_ratio: Ratio of nodes to select per graph
            min_nodes: Minimum nodes to select per graph
            max_nodes: Maximum nodes to select per graph
        """
        self.strategy = strategy
        self.top_k_ratio = top_k_ratio
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, graph_list: List[Data]):
        """
        Fit the selector on training data
        
        Args:
            graph_list: List of PyG Data objects
        """
        all_features = np.vstack([_to_numpy(data.x) for data in graph_list])
        self.scaler.fit(all_features)
        self.is_fitted = True
        print(f"✓ Node Selector fitted on {len(graph_list)} graphs")
    
    def select_nodes(
        self,
        data: Data
    ) -> np.ndarray:
        """
        Select nodes for a single graph
        
        Args:
            data: PyG Data object
        
        Returns:
            selected_indices: Array of selected node indices
        """
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before use")
        
        node_features = _to_numpy(data.x)
        num_nodes = node_features.shape[0]
        
        # Compute node importance scores based on strategy
        if self.strategy == "uncertainty":
            scores = self._compute_uncertainty_scores(node_features)
        elif self.strategy == "importance":
            scores = self._compute_structural_importance(data)
        elif self.strategy == "hybrid":
            uncertainty_scores = _normalize_scores(self._compute_uncertainty_scores(node_features))
            structural_scores = _normalize_scores(self._compute_structural_importance(data))
            scores = 0.5 * uncertainty_scores + 0.5 * structural_scores
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Determine number of nodes to select
        k = max(self.min_nodes, min(int(num_nodes * self.top_k_ratio), self.max_nodes, num_nodes))
        
        # Select top-k nodes
        selected_indices = np.argsort(scores)[-k:][::-1]  # Descending order
        
        return selected_indices
    
    def _compute_uncertainty_scores(self, node_features: np.ndarray) -> np.ndarray:
        """
        Compute uncertainty scores based on feature clustering
        
        Intuition: Nodes far from clusters are uncertain and need augmentation
        """
        if node_features.shape[0] < 2:
            return np.ones(node_features.shape[0])
        
        # Normalize features
        features_norm = self.scaler.transform(node_features)
        
        # Use DBSCAN to find outliers
        try:
            # Adaptive eps based on data scale
            k = min(5, features_norm.shape[0])
            nbrs = NearestNeighbors(n_neighbors=k).fit(features_norm)
            distances, _ = nbrs.kneighbors(features_norm)
            eps = np.median(distances[:, -1])
            
            dbscan = DBSCAN(eps=eps, min_samples=max(2, k//2))
            labels = dbscan.fit_predict(features_norm)
            
            # Compute scores: outliers (-1) get high scores, core points get low scores
            scores = np.zeros(len(labels))
            outlier_mask = (labels == -1)
            scores[outlier_mask] = 1.0
            
            # For cluster members, compute distance to cluster center
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster_mask = (labels == label)
                cluster_center = features_norm[cluster_mask].mean(axis=0)
                distances_to_center = np.linalg.norm(features_norm[cluster_mask] - cluster_center, axis=1)
                scores[cluster_mask] = distances_to_center
            
            scores = _normalize_scores(scores)
            
        except Exception:
            # Fallback: use feature variance as uncertainty
            scores = _normalize_scores(np.var(node_features, axis=1))
        
        return scores
    
    def _compute_structural_importance(self, data: Data) -> np.ndarray:
        """
        Compute structural importance based on graph topology
        
        Metrics:
        - Degree centrality
        - Position in propagation tree (root is most important)
        """
        num_nodes = data.x.shape[0]
        edge_index = _to_numpy(data.edge_index)
        
        # Compute degree centrality using vectorized operations
        degrees = np.zeros(num_nodes)
        np.add.at(degrees, edge_index[0], 1)  # Out-degree
        np.add.at(degrees, edge_index[1], 1)  # In-degree
        
        # Normalize
        if degrees.max() > 0:
            degrees = degrees / degrees.max()
        
        # Root node (node 0) is always important
        root_bonus = np.zeros(num_nodes)
        root_bonus[0] = 1.0
        
        # Combine metrics
        return 0.7 * degrees + 0.3 * root_bonus
    
    def get_selection_stats(
        self,
        graph_list: List[Data]
    ) -> Dict:
        """
        Get statistics about node selection
        
        Args:
            graph_list: List of PyG Data objects
        
        Returns:
            Dictionary with statistics
        """
        selected_counts = [len(self.select_nodes(data)) for data in graph_list]
        total_nodes = [data.x.shape[0] for data in graph_list]
        
        return {
            'total_graphs': len(graph_list),
            'avg_nodes_per_graph': np.mean(total_nodes),
            'avg_selected_per_graph': np.mean(selected_counts),
            'selection_ratio': np.mean(selected_counts) / np.mean(total_nodes),
            'min_selected': np.min(selected_counts),
            'max_selected': np.max(selected_counts)
        }

