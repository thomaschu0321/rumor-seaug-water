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
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


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
        # Collect all node features for normalization
        all_features = []
        for data in graph_list:
            node_features = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x
            all_features.append(node_features)
        
        all_features = np.vstack(all_features)
        self.scaler.fit(all_features)
        self.is_fitted = True
        
        print(f"✓ Node Selector fitted on {len(graph_list)} graphs")
    
    def select_nodes(
        self,
        data: Data,
        return_scores: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select nodes for a single graph
        
        Args:
            data: PyG Data object
            return_scores: Whether to return importance scores
        
        Returns:
            selected_indices: Array of selected node indices
            scores: Importance scores (if return_scores=True)
        """
        if not self.is_fitted:
            raise ValueError("Selector must be fitted before use")
        
        node_features = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x
        num_nodes = node_features.shape[0]
        
        # Compute node importance scores based on strategy
        if self.strategy == "uncertainty":
            scores = self._compute_uncertainty_scores(node_features)
        elif self.strategy == "importance":
            scores = self._compute_structural_importance(data)
        elif self.strategy == "hybrid":
            uncertainty_scores = self._compute_uncertainty_scores(node_features)
            structural_scores = self._compute_structural_importance(data)
            # Combine scores (normalize first)
            uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / (uncertainty_scores.max() - uncertainty_scores.min() + 1e-8)
            structural_scores = (structural_scores - structural_scores.min()) / (structural_scores.max() - structural_scores.min() + 1e-8)
            scores = 0.5 * uncertainty_scores + 0.5 * structural_scores
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Determine number of nodes to select
        k = int(num_nodes * self.top_k_ratio)
        k = max(self.min_nodes, min(k, self.max_nodes, num_nodes))
        
        # Select top-k nodes
        selected_indices = np.argsort(scores)[-k:][::-1]  # Descending order
        
        if return_scores:
            return selected_indices, scores
        return selected_indices
    
    def select_batch(
        self,
        graph_list: List[Data]
    ) -> List[np.ndarray]:
        """
        Select nodes for a batch of graphs
        
        Args:
            graph_list: List of PyG Data objects
        
        Returns:
            List of selected node indices for each graph
        """
        selected_nodes_list = []
        
        for data in graph_list:
            selected_indices = self.select_nodes(data)
            selected_nodes_list.append(selected_indices)
        
        return selected_nodes_list
    
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
            from sklearn.neighbors import NearestNeighbors
            k = min(5, features_norm.shape[0])
            nbrs = NearestNeighbors(n_neighbors=k).fit(features_norm)
            distances, _ = nbrs.kneighbors(features_norm)
            eps = np.median(distances[:, -1])
            
            dbscan = DBSCAN(eps=eps, min_samples=max(2, k//2))
            labels = dbscan.fit_predict(features_norm)
            
            # Compute scores: outliers (-1) get high scores, core points get low scores
            scores = np.zeros(len(labels))
            for i, label in enumerate(labels):
                if label == -1:
                    # Outlier: high uncertainty
                    scores[i] = 1.0
                else:
                    # Cluster member: compute distance to cluster center
                    cluster_mask = (labels == label)
                    cluster_center = features_norm[cluster_mask].mean(axis=0)
                    distance = np.linalg.norm(features_norm[i] - cluster_center)
                    scores[i] = distance
            
            # Normalize scores to [0, 1]
            if scores.max() > scores.min():
                scores = (scores - scores.min()) / (scores.max() - scores.min())
            
        except Exception as e:
            # Fallback: use feature variance as uncertainty
            scores = np.var(node_features, axis=1)
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        return scores
    
    def _compute_structural_importance(self, data: Data) -> np.ndarray:
        """
        Compute structural importance based on graph topology
        
        Metrics:
        - Degree centrality
        - Position in propagation tree (root is most important)
        """
        num_nodes = data.x.shape[0]
        edge_index = data.edge_index.numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index
        
        # Compute degree centrality
        degrees = np.zeros(num_nodes)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            degrees[src] += 1  # Out-degree
            degrees[dst] += 1  # In-degree
        
        # Normalize
        max_degree = degrees.max()
        if max_degree > 0:
            degrees = degrees / max_degree
        
        # Root node (node 0) is always important
        root_bonus = np.zeros(num_nodes)
        root_bonus[0] = 1.0
        
        # Combine metrics
        scores = 0.7 * degrees + 0.3 * root_bonus
        
        return scores
    
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
        selected_counts = []
        total_nodes = []
        
        for data in graph_list:
            selected = self.select_nodes(data)
            selected_counts.append(len(selected))
            total_nodes.append(data.x.shape[0])
        
        stats = {
            'total_graphs': len(graph_list),
            'avg_nodes_per_graph': np.mean(total_nodes),
            'avg_selected_per_graph': np.mean(selected_counts),
            'selection_ratio': np.mean(selected_counts) / np.mean(total_nodes),
            'min_selected': np.min(selected_counts),
            'max_selected': np.max(selected_counts)
        }
        
        return stats


def visualize_node_selection(
    data: Data,
    selected_indices: np.ndarray,
    scores: np.ndarray = None,
    save_path: str = None
):
    """
    Visualize which nodes were selected
    
    Args:
        data: PyG Data object
        selected_indices: Selected node indices
        scores: Node importance scores
        save_path: Path to save visualization
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    
    # Convert to NetworkX graph
    edge_index = data.edge_index.numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index
    G = nx.DiGraph()
    
    num_nodes = data.x.shape[0]
    G.add_nodes_from(range(num_nodes))
    
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Colors: selected nodes in red, others in blue
    colors = ['red' if i in selected_indices else 'lightblue' for i in range(num_nodes)]
    sizes = [500 if i in selected_indices else 200 for i in range(num_nodes)]
    
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f'Node Selection Visualization\n{len(selected_indices)}/{num_nodes} nodes selected')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


if __name__ == '__main__':
    # Test the node selector
    print("="*70)
    print("Testing Node Selector")
    print("="*70)
    
    # Create dummy data
    from torch_geometric.data import Data
    
    # Graph 1: 10 nodes
    x1 = torch.randn(10, 1000)
    edge_index1 = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 3, 4, 5, 5]], dtype=torch.long)
    data1 = Data(x=x1, edge_index=edge_index1)
    
    # Graph 2: 15 nodes
    x2 = torch.randn(15, 1000)
    edge_index2 = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    data2 = Data(x=x2, edge_index=edge_index2)
    
    graph_list = [data1, data2]
    
    # Test different strategies
    for strategy in ['uncertainty', 'importance', 'hybrid']:
        print(f"\n{'='*70}")
        print(f"Testing Strategy: {strategy}")
        print(f"{'='*70}")
        
        selector = NodeSelector(strategy=strategy, top_k_ratio=0.3)
        selector.fit(graph_list)
        
        for i, data in enumerate(graph_list):
            selected, scores = selector.select_nodes(data, return_scores=True)
            print(f"\nGraph {i+1}:")
            print(f"  Total nodes: {data.x.shape[0]}")
            print(f"  Selected nodes: {len(selected)}")
            print(f"  Indices: {selected}")
            print(f"  Scores (selected): {scores[selected]}")
        
        # Get statistics
        stats = selector.get_selection_stats(graph_list)
        print(f"\nSelection Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("✓ Node Selector test completed!")
    print("="*70)

