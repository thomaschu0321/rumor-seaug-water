"""Node Selection Module for SeAug Framework"""

import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict


def _to_numpy(x):
    return x.numpy() if isinstance(x, torch.Tensor) else x


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.max() > scores.min():
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores


class NodeSelector:
    def __init__(
        self,
        strategy: str = "uncertainty",
        top_k_ratio: float = 0.3,
        min_nodes: int = 1,
        max_nodes: int = 10
    ):
        self.strategy = strategy
        self.top_k_ratio = top_k_ratio
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, graph_list: List[Data]):
        all_features = np.vstack([_to_numpy(data.x) for data in graph_list])
        self.scaler.fit(all_features)
        self.is_fitted = True
        print(f"Node Selector fitted on {len(graph_list)} graphs")
    
    def select_nodes(self, data: Data) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before use")
        
        node_features = _to_numpy(data.x)
        num_nodes = node_features.shape[0]
        
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
        
        k = max(self.min_nodes, min(int(num_nodes * self.top_k_ratio), self.max_nodes, num_nodes))
        selected_indices = np.argsort(scores)[-k:][::-1]
        
        return selected_indices
    
    def _compute_uncertainty_scores(self, node_features: np.ndarray) -> np.ndarray:
        if node_features.shape[0] < 2:
            return np.ones(node_features.shape[0])
        
        features_norm = self.scaler.transform(node_features)
        
        try:
            k = min(5, features_norm.shape[0])
            nbrs = NearestNeighbors(n_neighbors=k).fit(features_norm)
            distances, _ = nbrs.kneighbors(features_norm)
            eps = np.median(distances[:, -1])
            
            dbscan = DBSCAN(eps=eps, min_samples=max(2, k//2))
            labels = dbscan.fit_predict(features_norm)
            
            scores = np.zeros(len(labels))
            outlier_mask = (labels == -1)
            scores[outlier_mask] = 1.0
            
            for label in np.unique(labels):
                if label == -1:
                    continue
                cluster_mask = (labels == label)
                cluster_center = features_norm[cluster_mask].mean(axis=0)
                distances_to_center = np.linalg.norm(features_norm[cluster_mask] - cluster_center, axis=1)
                scores[cluster_mask] = distances_to_center
            
            scores = _normalize_scores(scores)
        except Exception:
            scores = _normalize_scores(np.var(node_features, axis=1))
        
        return scores
    
    def _compute_structural_importance(self, data: Data) -> np.ndarray:
        num_nodes = data.x.shape[0]
        edge_index = _to_numpy(data.edge_index)
        
        degrees = np.zeros(num_nodes)
        np.add.at(degrees, edge_index[0], 1)
        np.add.at(degrees, edge_index[1], 1)
        
        if degrees.max() > 0:
            degrees = degrees / degrees.max()
        
        root_bonus = np.zeros(num_nodes)
        root_bonus[0] = 1.0
        
        return 0.7 * degrees + 0.3 * root_bonus
    
    def get_selection_stats(self, graph_list: List[Data]) -> Dict:
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

