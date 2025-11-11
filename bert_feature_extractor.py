"""
BERT Feature Extractor for Initial Node Features

This module extracts 768-dimensional BERT features for each node (tweet) in the graph.
These features serve as the X_initial (Initial Feature Matrix) for the TAPE framework.

Usage:
    - Extract BERT features from raw tweet text
    - Replace TF-IDF features with BERT features
    - Use as input to Phase 2 (Node Selection with DBSCAN)
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Dict, Optional
from tqdm import tqdm
import pickle

try:
    from transformers import BertTokenizer, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Warning: transformers not installed. Run: pip install transformers")


class BERTFeatureExtractor:
    """
    Extract 768-dimensional BERT features for tweet nodes
    
    This replaces TF-IDF features with BERT embeddings as the initial node features (x)
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = None,
        max_length: int = 128
    ):
        """
        Initialize BERT Feature Extractor
        
        Args:
            model_name: BERT model name (default: bert-base-uncased for 768-dim)
            device: Device to use ('cuda' or 'cpu')
            max_length: Maximum sequence length for BERT
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required. Run: pip install transformers")
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        print(f"Loading BERT model: {model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size  # 768 for bert-base
        
        print(f"✓ BERT Feature Extractor initialized")
        print(f"  Model: {model_name}")
        print(f"  Device: {self.device}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Max sequence length: {max_length}")
    
    def extract_single(self, text: str) -> np.ndarray:
        """
        Extract BERT features for a single text
        
        Args:
            text: Input text (tweet)
        
        Returns:
            768-dimensional BERT embedding
        """
        if not text or len(text.strip()) == 0:
            # Return zero vector for empty text
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as sentence representation
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        
        return cls_embedding.cpu().numpy()
    
    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract BERT features for a batch of texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            show_progress: Show progress bar
        
        Returns:
            Array of shape [num_texts, 768]
        """
        embeddings = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting BERT features")
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def process_graph(
        self,
        data: Data,
        node_texts: List[str]
    ) -> Data:
        """
        Process a single graph and replace x with BERT features
        
        Args:
            data: Original PyG Data object (with TF-IDF features)
            node_texts: List of text for each node
        
        Returns:
            Data object with BERT features as x
        """
        # Extract BERT features for all nodes
        bert_features = self.extract_batch(node_texts, show_progress=False)
        
        # Create new data object with BERT features
        data_bert = Data(
            x=torch.FloatTensor(bert_features),  # Replace with BERT features
            edge_index=data.edge_index,
            y=data.y if hasattr(data, 'y') else None,
            num_nodes=len(node_texts)
        )
        
        # Copy other attributes
        for key in data.keys():
            if key not in ['x', 'edge_index', 'y', 'num_nodes']:
                data_bert[key] = data[key]
        
        # Store original TF-IDF features for reference (optional)
        data_bert.x_tfidf = data.x
        
        return data_bert
    
    def process_graph_list(
        self,
        graph_list: List[Data],
        texts_list: List[List[str]],
        save_path: str = None
    ) -> List[Data]:
        """
        Process a list of graphs and replace all x with BERT features
        
        Args:
            graph_list: List of PyG Data objects
            texts_list: List of node texts for each graph
            save_path: Path to save processed graphs
        
        Returns:
            List of Data objects with BERT features
        """
        if len(graph_list) != len(texts_list):
            raise ValueError(f"Mismatch: {len(graph_list)} graphs but {len(texts_list)} text lists")
        
        print(f"\nProcessing {len(graph_list)} graphs with BERT...")
        
        bert_graphs = []
        for data, node_texts in tqdm(zip(graph_list, texts_list), 
                                     total=len(graph_list),
                                     desc="Processing graphs"):
            data_bert = self.process_graph(data, node_texts)
            bert_graphs.append(data_bert)
        
        print(f"✓ Processed {len(bert_graphs)} graphs")
        print(f"  Feature dimension: {bert_graphs[0].x.shape[1]}")
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(bert_graphs, f)
            print(f"✓ Saved to: {save_path}")
        
        return bert_graphs


def extract_node_texts_from_data(data: Data, dataset_name: str = "Twitter15") -> List[str]:
    """
    Extract text for each node in a graph
    
    For now, this is a placeholder. In actual implementation, you need to:
    1. Load raw tweet text from dataset
    2. Match tweet IDs to node indices
    3. Return list of texts
    
    Args:
        data: PyG Data object
        dataset_name: Dataset name
    
    Returns:
        List of text strings for each node
    """
    # TODO: Implement actual text extraction from raw data
    num_nodes = data.x.shape[0]
    
    # Placeholder: return dummy texts
    # In real implementation, load actual tweet texts
    texts = [f"Node {i} placeholder tweet text for {dataset_name}" for i in range(num_nodes)]
    
    return texts


def convert_tfidf_graphs_to_bert(
    tfidf_graph_list: List[Data],
    dataset_name: str = "Twitter15",
    save_path: str = None
) -> List[Data]:
    """
    Convert graphs with TF-IDF features to BERT features
    
    Args:
        tfidf_graph_list: List of graphs with TF-IDF features
        dataset_name: Dataset name
        save_path: Path to save BERT graphs
    
    Returns:
        List of graphs with BERT features
    """
    extractor = BERTFeatureExtractor()
    
    # Extract texts for each graph
    print("Extracting node texts...")
    texts_list = []
    for data in tqdm(tfidf_graph_list, desc="Preparing texts"):
        node_texts = extract_node_texts_from_data(data, dataset_name)
        texts_list.append(node_texts)
    
    # Process graphs
    bert_graphs = extractor.process_graph_list(
        tfidf_graph_list,
        texts_list,
        save_path=save_path
    )
    
    return bert_graphs


if __name__ == '__main__':
    print("="*70)
    print("Testing BERT Feature Extractor")
    print("="*70)
    
    # Test with sample texts
    sample_texts = [
        "Breaking news: Major earthquake hits California",
        "Just saw something interesting today",
        "This is fake news about celebrities",
        "Weather update: sunny tomorrow",
        "Important announcement from government"
    ]
    
    print("\n1. Testing single text extraction...")
    extractor = BERTFeatureExtractor()
    
    single_embedding = extractor.extract_single(sample_texts[0])
    print(f"✓ Single embedding shape: {single_embedding.shape}")
    print(f"  Expected: (768,)")
    
    print("\n2. Testing batch extraction...")
    batch_embeddings = extractor.extract_batch(sample_texts, batch_size=2)
    print(f"✓ Batch embeddings shape: {batch_embeddings.shape}")
    print(f"  Expected: ({len(sample_texts)}, 768)")
    
    print("\n3. Testing graph processing...")
    # Create dummy graph with TF-IDF features
    from torch_geometric.data import Data
    
    dummy_graph = Data(
        x=torch.randn(5, 1000),  # TF-IDF features
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long),
        y=torch.tensor([1])
    )
    
    bert_graph = extractor.process_graph(dummy_graph, sample_texts)
    print(f"✓ Graph processed:")
    print(f"  Original x shape: (5, 1000) [TF-IDF]")
    print(f"  New x shape: {bert_graph.x.shape} [BERT]")
    print(f"  Expected: (5, 768)")
    
    print("\n" + "="*70)
    print("✓ BERT Feature Extractor test completed!")
    print("="*70)

