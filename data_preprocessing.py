"""Data preprocessing module"""
import os
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pickle
from config import Config


class Node_tweet:
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.parent = None


def construct_tree(tree_dict):
    """Build tree structure and extract graph adjacency relationships"""
    index2node = {}
    for i in tree_dict:
        node = Node_tweet(idx=i)
        index2node[i] = node
    
    for j in tree_dict:
        indexC = j
        indexP = tree_dict[j]['parent']
        nodeC = index2node[indexC]
        
        if indexP != 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
    
    edge_index = [[], []]
    num_nodes = len(index2node)
    
    for i in range(num_nodes):
        node = index2node[i + 1]
        for child in node.children:
            edge_index[0].append(i)
            edge_index[1].append(child.idx - 1)
    
    return edge_index, num_nodes




class TwitterDataProcessor:
    def __init__(self, dataname='Twitter15', feature_dim=768, sample_ratio=1.0):
        self.dataname = dataname
        self.sample_ratio = sample_ratio
        
        self.data_path = os.path.join(Config.DATA_DIR, 'Twitter', dataname)
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        from bert_feature_extractor import BERTFeatureExtractor
        self.bert_extractor = BERTFeatureExtractor(model_name="bert-base-uncased")
    
    def load_texts(self):
        """Load original tweet texts"""
        text_file = os.path.join(self.data_path, f'{self.dataname}_source_tweets.txt')
        
        if not os.path.exists(text_file):
            print(f"Warning: Text file not found: {text_file}")
            return {}
        
        texts_dict = {}
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eid = parts[0]
                    text = parts[1] if len(parts[1]) > 0 else "empty tweet"
                    texts_dict[eid] = text
        
        print(f"Loaded {len(texts_dict)} texts")
        return texts_dict
    
    def load_raw_data(self):
        """Load raw data"""
        tree_file = os.path.join(self.data_path, 'data.TD_RvNN.vol_5000.txt')
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
            
            if eid not in treeDic:
                treeDic[eid] = {}
            treeDic[eid][indexC] = {'parent': indexP}
        
        label_file = os.path.join(self.data_path, f'{self.dataname}_label_All.txt')
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file does not exist: {label_file}")
        
        labelDic = {}
        for line in open(label_file, encoding='utf-8'):
            line = line.rstrip()
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            
            label = parts[0].lower()
            eid = parts[2]
            
            if label in ['news', 'non-rumor']:
                labelDic[eid] = 0
            elif label in ['false', 'true', 'unverified']:
                labelDic[eid] = 1
        
        print(f"Loaded {len(treeDic)} trees, {len(labelDic)} labels")
        return treeDic, labelDic
    
    def process_data(self):
        """Process data and convert to PyG graph objects"""
        print(f"\nProcessing {self.dataname}...")
        
        treeDic, labelDic = self.load_raw_data()
        texts_dict = self.load_texts()
        
        if not texts_dict:
            raise ValueError(f"No texts loaded. Expected: {os.path.join(self.data_path, f'{self.dataname}_source_tweets.txt')}")
        
        valid_eids = [eid for eid in labelDic.keys() if eid in treeDic and eid in texts_dict]
        
        if self.sample_ratio < 1.0:
            num_samples = int(len(valid_eids) * self.sample_ratio)
            np.random.seed(42)
            valid_eids = np.random.choice(valid_eids, num_samples, replace=False).tolist()
        
        graph_list = []
        skipped = 0
        
        for eid in tqdm(valid_eids, desc="Processing"):
            try:
                tree = treeDic[eid]
                label = labelDic[eid]
                
                if len(tree) < 2:
                    skipped += 1
                    continue
                
                edge_index, num_nodes = construct_tree(tree)
                root_text = texts_dict[eid]
                texts = [root_text] * num_nodes
                x = self.bert_extractor.extract_batch(texts, show_progress=False)
                
                data = Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index),
                    y=torch.LongTensor([label]),
                    num_nodes=num_nodes,
                    eid=eid
                )
                
                graph_list.append(data)
            except Exception as e:
                print(f"Error processing {eid}: {e}")
                skipped += 1
        
        print(f"Processed: {len(graph_list)} graphs")
        if skipped > 0:
            print(f"Skipped: {skipped} graphs")
        
        return graph_list
    
    def save_processed_data(self, graph_list, save_path=None):
        """Save processed data"""
        if save_path is None:
            save_path = Config.PROCESSED_DIR
        
        os.makedirs(save_path, exist_ok=True)
        
        if self.sample_ratio == 1.0:
            filename = f'{self.dataname}_processed_bert_full.pkl'
        else:
            filename = f'{self.dataname}_processed_bert_sample{self.sample_ratio}.pkl'
        
        filepath = os.path.join(save_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(graph_list, f)
        
        print(f"Saved: {filepath}")
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
            raise FileNotFoundError(f"Processed data not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            graph_list = pickle.load(f)
        
        print(f"Loaded {len(graph_list)} graphs")
        return graph_list


class WeiboDataProcessor:
    def __init__(self, dataname='Weibo', feature_dim=768, sample_ratio=1.0):
        self.dataname = dataname
        self.sample_ratio = sample_ratio
        
        self.data_path = os.path.join(Config.DATA_DIR, 'Weibo')
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")
        
        from bert_feature_extractor import BERTFeatureExtractor
        self.bert_extractor = BERTFeatureExtractor(model_name="bert-base-uncased")
    
    def load_texts(self):
        """Load original Weibo texts"""
        text_file = os.path.join(Config.DATA_DIR, 'raw_text', 'Weibo.txt')
        
        if not os.path.exists(text_file):
            print(f"Warning: Text file not found: {text_file}")
            return {}
        
        texts_dict = {}
        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    eid = parts[0]
                    text = parts[1] if len(parts[1]) > 0 else "empty post"
                    texts_dict[eid] = text
        
        print(f"Loaded {len(texts_dict)} texts")
        return texts_dict
    
    def load_raw_data(self):
        """Load raw Weibo data"""
        label_file = os.path.join(self.data_path, 'weibo_id_label.txt')
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file does not exist: {label_file}")
        
        labelDic = {}
        for line in open(label_file, encoding='utf-8'):
            line = line.rstrip()
            parts = line.split(' ') if ' ' in line else line.split('\t')
            if len(parts) < 2:
                continue
            
            eid = parts[0]
            label = int(parts[1])
            labelDic[eid] = label
        
        valid_eids = list(labelDic.keys())
        if self.sample_ratio < 1.0:
            num_samples = max(1, int(len(valid_eids) * self.sample_ratio))
            np.random.seed(42)
            valid_eids = np.random.choice(valid_eids, num_samples, replace=False).tolist()
        
        valid_eids_set = set(valid_eids)
        
        tree_file = os.path.join(self.data_path, 'weibotree.txt')
        if not os.path.exists(tree_file):
            raise FileNotFoundError(f"Tree file does not exist: {tree_file}")
        
        treeDic = {}
        with open(tree_file, encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading tree data"):
                line = line.rstrip()
                parts = line.split('\t')
                if len(parts) < 4:
                    continue
                
                eid = parts[0]
                if eid not in valid_eids_set:
                    continue
                
                indexP = parts[1]
                indexC = int(parts[2])
                
                if eid not in treeDic:
                    treeDic[eid] = {}
                treeDic[eid][indexC] = {'parent': indexP}
                
                if len(treeDic) >= len(valid_eids):
                    break
        
        filtered_labelDic = {eid: labelDic[eid] for eid in valid_eids if eid in labelDic}
        return treeDic, filtered_labelDic
    
    def process_data(self):
        """Process data and convert to PyG graph objects"""
        print(f"\nProcessing {self.dataname}...")
        
        treeDic, labelDic = self.load_raw_data()
        texts_dict = self.load_texts()
        
        if not texts_dict:
            raise ValueError(f"No texts loaded. Expected: {os.path.join(Config.DATA_DIR, 'raw_text', 'Weibo.txt')}")
        
        valid_eids = [eid for eid in labelDic.keys() if eid in treeDic and eid in texts_dict]
        
        graph_list = []
        skipped = 0
        
        for eid in tqdm(valid_eids, desc="Processing"):
            try:
                tree = treeDic[eid]
                label = labelDic[eid]
                
                if len(tree) < 2:
                    skipped += 1
                    continue
                
                edge_index, num_nodes = construct_tree(tree)
                root_text = texts_dict[eid]
                texts = [root_text] * num_nodes
                x = self.bert_extractor.extract_batch(texts, show_progress=False)
                
                data = Data(
                    x=torch.FloatTensor(x),
                    edge_index=torch.LongTensor(edge_index),
                    y=torch.LongTensor([label]),
                    num_nodes=num_nodes,
                    eid=eid
                )
                
                graph_list.append(data)
            except Exception as e:
                print(f"Error processing {eid}: {e}")
                skipped += 1
        
        print(f"Processed: {len(graph_list)} graphs")
        if skipped > 0:
            print(f"Skipped: {skipped} graphs")
        
        return graph_list
    
    def save_processed_data(self, graph_list, save_path=None):
        """Save processed data"""
        if save_path is None:
            save_path = Config.PROCESSED_DIR
        
        os.makedirs(save_path, exist_ok=True)
        
        if self.sample_ratio == 1.0:
            filename = f'{self.dataname}_processed_bert_full.pkl'
        else:
            filename = f'{self.dataname}_processed_bert_sample{self.sample_ratio}.pkl'
        
        filepath = os.path.join(save_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(graph_list, f)
        
        print(f"Saved: {filepath}")
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
            raise FileNotFoundError(f"Processed data not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            graph_list = pickle.load(f)
        
        print(f"Loaded {len(graph_list)} graphs")
        return graph_list


def split_data(graph_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split dataset into train/val/test"""
    np.random.seed(seed)
    indices = np.random.permutation(len(graph_list))
    
    n_train = int(len(graph_list) * train_ratio)
    n_val = int(len(graph_list) * val_ratio)
    
    train_list = [graph_list[i] for i in indices[:n_train]]
    val_list = [graph_list[i] for i in indices[n_train:n_train + n_val]]
    test_list = [graph_list[i] for i in indices[n_train + n_val:]]
    
    print(f"Split: Train={len(train_list)}, Val={len(val_list)}, Test={len(test_list)}")
    return train_list, val_list, test_list
