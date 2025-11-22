"""
BERT Feature Extractor for Initial Node Features

This module extracts 768-dimensional BERT features for each node (tweet) in the graph.
These features serve as the X_initial (Initial Feature Matrix) for the SeAug framework.

Usage:
    - Extract BERT features from raw tweet text
    - Replace TF-IDF features with BERT features
    - Use as input to Phase 2 (Node Selection with DBSCAN)
"""

import numpy as np
import torch
from typing import List
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


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

