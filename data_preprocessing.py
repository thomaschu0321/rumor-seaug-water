"""
Data Preprocessing Module
Convert from original data format to simplified graph structure
"""
import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
from config import Config


class Node_tweet:
    """Tweet node class"""
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.parent = None


def construct_tree(tree_dict):
    """
    Build tree structure and extract graph adjacency relationships
    Returns: edge_index, num_nodes, root_index
    """
    # Create nodes
    index2node = {}
    for i in tree_dict:
        node = Node_tweet(idx=i)
        index2node[i] = node
    
    # Establish parent-child relationships
    root_index = None
    for j in tree_dict:
        indexC = j
        indexP = tree_dict[j]['parent']
        nodeC = index2node[indexC]
        
        # Not root node
        if indexP != 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        else:
            # Root node
            root_index = indexC - 1  # Convert to 0-based
    
    # Build adjacency matrix (edge_index format)
    edge_index = [[], []]
    
    num_nodes = len(index2node)
    for i in range(num_nodes):
        node = index2node[i + 1]
        
        # Collect edges (parent to child, Top-Down)
        for child in node.children:
            edge_index[0].append(i)  # Source node
            edge_index[1].append(child.idx - 1)  # Target node
    
    return edge_index, num_nodes, root_index




class TwitterDataProcessor:
    """Twitter data processor"""
    
    def __init__(self, dataname='Twitter15', feature_dim=768, sample_ratio=1.0):
        """
        dataname: 'Twitter15' or 'Twitter16'
        feature_dim: Feature dimension (768 for BERT)
        sample_ratio: Sampling ratio (1.0 means use all data)
        """
        self.dataname = dataname
        self.feature_dim = feature_dim
        self.sample_ratio = sample_ratio
        
        # Use new path structure: data/Twitter/{dataname}/
        self.data_path = os.path.join(Config.DATA_DIR, 'Twitter', dataname)
        
        # Check if path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data path does not exist: {self.data_path}\n"
                f"Please ensure Twitter data is in correct location"
            )
        
        # Initialize BERT extractor (always required)
        from bert_feature_extractor import BERTFeatureExtractor
        print(f"Initializing BERT feature extractor...")
        self.bert_extractor = BERTFeatureExtractor(model_name="bert-base-uncased")
    
    def load_texts(self):
        """Load original tweet texts (needed for BERT)"""
        # Use new path structure: data/Twitter/{dataname}/{dataname}_source_tweets.txt
        text_file = os.path.join(self.data_path, f'{self.dataname}_source_tweets.txt')
        
        if not os.path.exists(text_file):
            print(f"⚠️  Text file not found: {text_file}")
            return {}
        
        texts_dict = {}
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eid = parts[0]
                    text = parts[1] if len(parts[1]) > 0 else "empty tweet"
                    texts_dict[eid] = text
        
        print(f"✓ Loaded {len(texts_dict)} texts from {text_file}")
        return texts_dict
    
    def load_raw_data(self):
        """Load raw data"""
        # Load tree structure
        tree_file = os.path.join(self.data_path, 'data.TD_RvNN.vol_5000.txt')
        print(f"Reading tree structure file: {tree_file}")
        
        if not os.path.exists(tree_file):
            raise FileNotFoundError(f"Tree file does not exist: {tree_file}")
        
        treeDic = {}
        for line in open(tree_file, encoding='utf-8'):
            line = line.rstrip()
            parts = line.split('\t')
            if len(parts) < 6:
                continue
            
            eid = parts[0]
            indexP = parts[1]
            indexC = int(parts[2])
            Vec = parts[5]
            
            if eid not in treeDic:
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
        
        print(f"Number of trees: {len(treeDic)}")
        
        # Load labels
        label_file = os.path.join(self.data_path, f'{self.dataname}_label_All.txt')
        print(f"Reading label file: {label_file}")
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file does not exist: {label_file}")
        
        labelDic = {}
        label_counts = {'non-rumor': 0, 'false': 0, 'true': 0, 'unverified': 0}
        
        for line in open(label_file, encoding='utf-8'):
            line = line.rstrip()
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            label = parts[0].lower()
            eid = parts[2]
            
            # Simplify to 2-class
            if label in ['news', 'non-rumor']:
                labelDic[eid] = 0  # Non-rumor
                label_counts['non-rumor'] += 1
            elif label in ['false', 'true', 'unverified']:
                labelDic[eid] = 1  # Rumor
                label_counts[label] += 1
        
        print(f"Label statistics: {label_counts}")
        print(f"Total samples: {len(labelDic)}")
        
        return treeDic, labelDic
    
    def process_data(self):
        """Process data and convert to PyG graph objects"""
        print(f"\n{'='*70}")
        print(f"Processing {self.dataname} with BERT (768-dim) features")
        print(f"{'='*70}")
        
        treeDic, labelDic = self.load_raw_data()
        
        # Load texts (required for BERT)
        texts_dict = self.load_texts()
        if not texts_dict:
            raise ValueError(
                f"No texts loaded for BERT feature extraction. "
                f"Expected text file at: {os.path.join(self.data_path, f'{self.dataname}_source_tweets.txt')}"
            )
        
        # Get valid event IDs (must have text, tree, and label)
        valid_eids = [eid for eid in labelDic.keys() 
                     if eid in treeDic and eid in texts_dict]
        print(f"Valid samples (with text, tree, and label): {len(valid_eids)}")
        
        # Sampling (if needed)
        if self.sample_ratio < 1.0:
            num_samples = int(len(valid_eids) * self.sample_ratio)
            np.random.seed(42)
            valid_eids = np.random.choice(valid_eids, num_samples, replace=False).tolist()
            print(f"Samples after sampling: {len(valid_eids)}")
        else:
            print(f"Using all data: {len(valid_eids)} samples")
        
        # Convert to graph objects
        graph_list = []
        skipped = 0
        
        for eid in tqdm(valid_eids, desc="Extracting BERT features"):
            try:
                tree = treeDic[eid]
                label = labelDic[eid]
                
                # Check tree size
                if len(tree) < 2:
                    skipped += 1
                    continue
                
                # Build graph
                edge_index, num_nodes, root_index = construct_tree(tree)
                
                # Extract BERT features (768-dim)
                root_text = texts_dict[eid]
                # Simplified: use root tweet text for all nodes
                # TODO: In future, extract text for each node individually
                texts = [root_text] * num_nodes
                x = self.bert_extractor.extract_batch(texts)
                
                # Create PyG Data object
                data = Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index),
                    y=torch.LongTensor([label]),
                    num_nodes=num_nodes,
                    eid=eid
                )
                
                graph_list.append(data)
                
            except Exception as e:
                print(f"Error processing event {eid}: {e}")
                skipped += 1
                continue
        
        print(f"\n✓ Successfully processed: {len(graph_list)} graphs")
        if skipped > 0:
            print(f"✗ Skipped: {skipped} graphs")
        
        # Label distribution statistics
        labels = [data.y.item() for data in graph_list]
        print(f"\nLabel distribution:")
        print(f"  Non-rumor (0): {labels.count(0)} ({labels.count(0)/len(labels)*100:.1f}%)")
        print(f"  Rumor (1):     {labels.count(1)} ({labels.count(1)/len(labels)*100:.1f}%)")
        
        return graph_list
    
    def save_processed_data(self, graph_list, save_path=None):
        """Save processed data"""
        if save_path is None:
            save_path = Config.PROCESSED_DIR
        
        os.makedirs(save_path, exist_ok=True)
        
        # Name based on sampling ratio (always BERT)
        if self.sample_ratio == 1.0:
            filename = f'{self.dataname}_processed_bert_full.pkl'
        else:
            filename = f'{self.dataname}_processed_bert_sample{self.sample_ratio}.pkl'
        
        filepath = os.path.join(save_path, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph_list, f)
        
        print(f"\n✓ Data saved to: {filepath}")
        return filepath
    
    def load_processed_data(self, load_path=None):
        """Load processed data"""
        if load_path is None:
            if self.sample_ratio == 1.0:
                filename = f'{self.dataname}_processed_bert_full.pkl'
            else:
                filename = f'{self.dataname}_processed_bert_sample{self.sample_ratio}.pkl'
            load_path = os.path.join(Config.PROCESSED_DIR, filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Processed data not found: {load_path}\n"
                f"Please run processor.process_data() first"
            )
        
        print(f"Loading data from {load_path}...")
        with open(load_path, 'rb') as f:
            graph_list = pickle.load(f)
        
        print(f"✓ Loaded {len(graph_list)} graphs")
        return graph_list


class WeiboDataProcessor:
    """Weibo data processor"""
    
    def __init__(self, dataname='Weibo', feature_dim=768, sample_ratio=1.0):
        """
        dataname: 'Weibo'
        feature_dim: Feature dimension (768 for BERT)
        sample_ratio: Sampling ratio (1.0 means use all data)
        """
        self.dataname = dataname
        self.feature_dim = feature_dim
        self.sample_ratio = sample_ratio
        
        # Use project data path structure: data/Weibo/
        self.data_path = os.path.join(Config.DATA_DIR, 'Weibo')
        
        # Check if path exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data path does not exist: {self.data_path}\n"
                f"Please ensure Weibo data is in correct location"
            )
        
        # Initialize BERT extractor (always required)
        from bert_feature_extractor import BERTFeatureExtractor
        print(f"Initializing BERT feature extractor...")
        self.bert_extractor = BERTFeatureExtractor(model_name="bert-base-uncased")
    
    def load_texts(self):
        """Load original Weibo texts (needed for BERT)"""
        text_file = os.path.join(Config.DATA_DIR, 'raw_text', 'Weibo.txt')
        
        if not os.path.exists(text_file):
            print(f"⚠️  Text file not found: {text_file}")
            return {}
        
        texts_dict = {}
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eid = parts[0]
                    text = parts[1] if len(parts[1]) > 0 else "empty post"
                    texts_dict[eid] = text
        
        print(f"✓ Loaded {len(texts_dict)} texts from {text_file}")
        return texts_dict
    
    def load_raw_data(self):
        """Load raw Weibo data"""
        # Load labels first to know which events we need
        label_file = os.path.join(self.data_path, 'weibo_id_label.txt')
        print(f"Reading label file: {label_file}")
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file does not exist: {label_file}")
        
        labelDic = {}
        label_counts = {'non-rumor': 0, 'rumor': 0}
        
        for line in open(label_file, encoding='utf-8'):
            line = line.rstrip()
            parts = line.split(' ')
            if len(parts) < 2:
                parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            eid = parts[0]  # Event ID
            label = int(parts[1])  # Label: 0 (non-rumor) or 1 (rumor)
            
            labelDic[eid] = label
            if label == 0:
                label_counts['non-rumor'] += 1
            else:
                label_counts['rumor'] += 1
        
        print(f"Label statistics: {label_counts}")
        print(f"Total events: {len(labelDic)}")
        
        # Apply sampling to event IDs first (before loading large tree file)
        valid_eids = list(labelDic.keys())
        if self.sample_ratio < 1.0:
            num_samples = max(1, int(len(valid_eids) * self.sample_ratio))
            np.random.seed(42)
            valid_eids = np.random.choice(valid_eids, num_samples, replace=False).tolist()
            print(f"Sampled events: {num_samples} ({self.sample_ratio*100:.1f}%)")
        
        # Convert to set for faster lookup
        valid_eids_set = set(valid_eids)
        
        # Load tree structure (only for sampled events)
        tree_file = os.path.join(self.data_path, 'weibotree.txt')
        print(f"Reading Weibo tree structure file: {tree_file}")
        print(f"  (Only loading {len(valid_eids)} events out of 4664 total)")
        
        if not os.path.exists(tree_file):
            raise FileNotFoundError(f"Tree file does not exist: {tree_file}")
        
        treeDic = {}
        lines_read = 0
        lines_used = 0
        
        with open(tree_file, encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading tree data", unit=" lines"):
                lines_read += 1
                line = line.rstrip()
                parts = line.split('\t')
                if len(parts) < 4:
                    continue
                
                eid = parts[0]  # Event ID
                
                # Skip events not in our sample
                if eid not in valid_eids_set:
                    continue
                
                lines_used += 1
                indexP = parts[1]  # Parent node ID
                indexC = int(parts[2])  # Current node ID
                Vec = parts[3]  # Word vector
                
                if eid not in treeDic:
                    treeDic[eid] = {}
                treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
                
                # Early stopping: if we have all events we need, stop reading
                if len(treeDic) >= len(valid_eids):
                    print(f"  ✓ Found all {len(valid_eids)} events, stopping early")
                    break
        
        print(f"  Read {lines_read:,} lines, used {lines_used:,} lines")
        print(f"  Number of trees loaded: {len(treeDic)}")
        
        # Filter labelDic to only include sampled events
        filtered_labelDic = {eid: labelDic[eid] for eid in valid_eids if eid in labelDic}
        
        return treeDic, filtered_labelDic
    
    def process_data(self):
        """Process data and convert to PyG graph objects"""
        print(f"\n{'='*70}")
        print(f"Processing {self.dataname} with BERT (768-dim) features")
        print(f"{'='*70}")
        
        treeDic, labelDic = self.load_raw_data()
        
        # Load texts (required for BERT)
        texts_dict = self.load_texts()
        if not texts_dict:
            raise ValueError(
                f"No texts loaded for BERT feature extraction. "
                f"Expected text file at: {os.path.join(Config.DATA_DIR, 'raw_text', 'Weibo.txt')}"
            )
        
        # Get valid event IDs (must have text, tree, and label)
        valid_eids = [eid for eid in labelDic.keys() 
                     if eid in treeDic and eid in texts_dict]
        
        print(f"Valid samples (with text, tree, and label): {len(valid_eids)}")
        
        # Convert to graph objects
        graph_list = []
        skipped = 0
        
        for eid in tqdm(valid_eids, desc="Extracting BERT features"):
            try:
                tree = treeDic[eid]
                label = labelDic[eid]
                
                # Check tree size
                if len(tree) < 2:
                    skipped += 1
                    continue
                
                # Build graph
                edge_index, num_nodes, root_index = construct_tree(tree)
                
                # Extract BERT features (768-dim)
                root_text = texts_dict[eid]
                # Simplified: use root post text for all nodes
                # TODO: In future, extract text for each node individually
                texts = [root_text] * num_nodes
                x = self.bert_extractor.extract_batch(texts)
                
                # Create PyG Data object
                data = Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index),
                    y=torch.LongTensor([label]),
                    num_nodes=num_nodes,
                    eid=eid
                )
                
                graph_list.append(data)
                
            except Exception as e:
                print(f"Error processing event {eid}: {e}")
                skipped += 1
                continue
        
        print(f"\n✓ Successfully processed: {len(graph_list)} graphs")
        if skipped > 0:
            print(f"✗ Skipped: {skipped} graphs")
        
        # Label distribution statistics
        labels = [data.y.item() for data in graph_list]
        print(f"\nLabel distribution:")
        print(f"  Non-rumor (0): {labels.count(0)} ({labels.count(0)/len(labels)*100:.1f}%)")
        print(f"  Rumor (1):     {labels.count(1)} ({labels.count(1)/len(labels)*100:.1f}%)")
        
        return graph_list
    
    def save_processed_data(self, graph_list, save_path=None):
        """Save processed data"""
        if save_path is None:
            save_path = Config.PROCESSED_DIR
        
        os.makedirs(save_path, exist_ok=True)
        
        # Name based on sampling ratio (always BERT)
        if self.sample_ratio == 1.0:
            filename = f'{self.dataname}_processed_bert_full.pkl'
        else:
            filename = f'{self.dataname}_processed_bert_sample{self.sample_ratio}.pkl'
        
        filepath = os.path.join(save_path, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph_list, f)
        
        print(f"\n✓ Data saved to: {filepath}")
        return filepath
    
    def load_processed_data(self, load_path=None):
        """Load processed data"""
        if load_path is None:
            if self.sample_ratio == 1.0:
                filename = f'{self.dataname}_processed_bert_full.pkl'
            else:
                filename = f'{self.dataname}_processed_bert_sample{self.sample_ratio}.pkl'
            load_path = os.path.join(Config.PROCESSED_DIR, filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Processed data not found: {load_path}\n"
                f"Please run processor.process_data() first"
            )
        
        print(f"Loading data from {load_path}...")
        with open(load_path, 'rb') as f:
            graph_list = pickle.load(f)
        
        print(f"✓ Loaded {len(graph_list)} graphs")
        return graph_list


def split_data(graph_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset
    Returns: train_list, val_list, test_list
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(graph_list))
    
    n_train = int(len(graph_list) * train_ratio)
    n_val = int(len(graph_list) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_list = [graph_list[i] for i in train_indices]
    val_list = [graph_list[i] for i in val_indices]
    test_list = [graph_list[i] for i in test_indices]
    
    print(f"\nData split:")
    print(f"  Train set: {len(train_list)} ({len(train_list)/len(graph_list)*100:.1f}%)")
    print(f"  Val set:   {len(val_list)} ({len(val_list)/len(graph_list)*100:.1f}%)")
    print(f"  Test set:  {len(test_list)} ({len(test_list)/len(graph_list)*100:.1f}%)")
    
    return train_list, val_list, test_list
