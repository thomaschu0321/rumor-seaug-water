import numpy as np
import torch
from typing import List
from tqdm import tqdm
from transformers import BertTokenizer, BertModel


class BERTFeatureExtractor:
    """Extract BERT features for tweet nodes"""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        device: str = None,
        max_length: int = 128
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
    
    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Extract BERT features for a batch of texts"""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting BERT features", total=total_batches)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
            
            embeddings.append(cls_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

