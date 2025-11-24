"""Data preprocessing module"""
import os
import json
from email.utils import parsedate_to_datetime
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




def _extract_node_texts(tree, default_text):
    """Return ordered list of node texts, falling back to default_text"""
    num_nodes = len(tree)
    texts = []
    for idx in range(1, num_nodes + 1):
        node_text = tree.get(idx, {}).get('text')
        if not node_text:
            node_text = default_text
        texts.append(node_text)
    return texts


def _build_processed_filename(dataname, sample_ratio):
    """Return the canonical filename used when caching processed graphs."""
    if sample_ratio == 1.0:
        return f'{dataname}_processed_bert_full.pkl'
    return f'{dataname}_processed_bert_sample{sample_ratio}.pkl'


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
                texts = _extract_node_texts(tree, root_text)
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
        if save_path is None:
            save_path = Config.PROCESSED_DIR
        os.makedirs(save_path, exist_ok=True)
        
        filename = _build_processed_filename(self.dataname, self.sample_ratio)
        filepath = os.path.join(save_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(graph_list, f)
        print(f"Saved: {filepath}")
        return filepath
    
    def load_processed_data(self, load_path=None):
        if load_path is None:
            filename = _build_processed_filename(self.dataname, self.sample_ratio)
            load_path = os.path.join(Config.PROCESSED_DIR, filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Processed data not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            graph_list = pickle.load(f)
        print(f"Loaded {len(graph_list)} graphs")
        return graph_list


class PhemeDataProcessor:
    def __init__(
        self,
        dataname='PHEME',
        feature_dim=768,
        sample_ratio=1.0,
        events=None,
    ):
        self.dataname = dataname
        self.sample_ratio = sample_ratio
        self.feature_dim = feature_dim
        self.data_root = self._resolve_data_root()
        discovered_events = self._discover_events()
        
        if events is None:
            self.events = discovered_events
        else:
            # Explicit selections override discovery.
            self.events = list(events)

        if not self.events:
            raise ValueError(f"No events found under {self.data_root}")

        print(f"PHEME events selected: {', '.join(self.events)}")
        
        from bert_feature_extractor import BERTFeatureExtractor
        self.bert_extractor = BERTFeatureExtractor(model_name="bert-base-uncased")
    
    @staticmethod
    def _read_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except UnicodeDecodeError as exc:
            # Some raw PHEME dumps ship with cp1252/latin-1 bytes; fall back so we do not drop threads.
            print(f"Warning: UTF-8 decode failed for {path}: {exc}; retrying with latin-1.")
            with open(path, 'r', encoding='latin-1') as f:
                return json.load(f)
    
    @staticmethod
    def _clean_text(text):
        text = text or ""
        text = text.replace('\n', ' ').strip()
        return text if text else "empty tweet"
    
    @staticmethod
    def _get_timestamp(created_at):
        if not created_at:
            return None
        try:
            return parsedate_to_datetime(created_at).timestamp()
        except Exception:
            return None
    
    @staticmethod
    def _normalize_subset_name(name):
        lowered = name.lower().replace('_', '-')
        if 'non' in lowered and 'rum' in lowered:
            return 'non-rumours'
        if 'rum' in lowered:
            return 'rumours'
        return None
    
    @staticmethod
    def _find_existing_dir(base_path, candidates):
        for candidate in candidates:
            candidate_path = os.path.join(base_path, candidate)
            if os.path.isdir(candidate_path):
                return candidate_path
        return None
    
    def _resolve_data_root(self):
        """Find whichever PHEME layout is available."""
        candidate_roots = [
            os.path.join(Config.DATA_DIR, 'PHEME', 'all-rnr-annotated-threads'),
            os.path.join(Config.DATA_DIR, 'PHEME'),
        ]
        for root in candidate_roots:
            if os.path.isdir(root):
                return root
        raise FileNotFoundError(
            "Could not locate either legacy or simplified PHEME directories under data/PHEME"
        )
    
    def _discover_events(self):
        events = []
        for entry in sorted(os.listdir(self.data_root)):
            event_path = os.path.join(self.data_root, entry)
            if not os.path.isdir(event_path):
                continue
            subset_dirs = [
                self._normalize_subset_name(d)
                for d in os.listdir(event_path)
                if os.path.isdir(os.path.join(event_path, d))
            ]
            if any(s in ('rumours', 'non-rumours') for s in subset_dirs):
                events.append(entry)
        if not events:
            raise ValueError(f"No PHEME events discovered under {self.data_root}")
        return events
    
    def _build_thread_tree(self, thread_path):
        source_dir = self._find_existing_dir(thread_path, ['source-tweets', 'source-tweet'])
        reaction_dir = self._find_existing_dir(thread_path, ['reactions'])
        
        if not source_dir:
            raise FileNotFoundError(f"Missing source directory: {source_dir}")
        
        source_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith('.json')]
        if not source_files:
            raise FileNotFoundError(f"No source tweet json under {source_dir}")
        
        source_data = self._read_json(source_files[0])
        root_id = str(source_data.get('id_str') or source_data.get('id'))
        root_text = self._clean_text(source_data.get('text', ''))
        root_ts = self._get_timestamp(source_data.get('created_at')) or 0.0
        
        tweets = {
            root_id: {
                'text': root_text,
                'parent': None,
                'timestamp': root_ts
            }
        }
        
        if reaction_dir and os.path.isdir(reaction_dir):
            for fname in os.listdir(reaction_dir):
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(reaction_dir, fname)
                try:
                    tweet = self._read_json(fpath)
                except json.JSONDecodeError:
                    continue
                
                tweet_id = str(tweet.get('id_str') or tweet.get('id'))
                if not tweet_id or tweet_id in tweets:
                    continue
                
                parent_id = tweet.get('in_reply_to_status_id_str') or tweet.get('in_reply_to_status_id')
                parent_id = str(parent_id) if parent_id else root_id
                tweets[tweet_id] = {
                    'text': self._clean_text(tweet.get('text', '')),
                    'parent': parent_id,
                    'timestamp': self._get_timestamp(tweet.get('created_at'))
                }
        
        # Ensure parents exist; fallback to root if missing
        valid_ids = set(tweets.keys())
        for tid, meta in list(tweets.items()):
            parent = meta['parent']
            if parent and parent not in valid_ids:
                meta['parent'] = root_id
        
        # Order tweets: root first, then by timestamp/id for stability
        other_ids = [tid for tid in tweets.keys() if tid != root_id]
        other_ids.sort(key=lambda tid: ((tweets[tid]['timestamp'] if tweets[tid]['timestamp'] is not None else float('inf')), tid))
        ordered_ids = [root_id] + other_ids
        id_map = {tid: idx + 1 for idx, tid in enumerate(ordered_ids)}
        
        tree = {}
        for tid in ordered_ids:
            idx = id_map[tid]
            parent_raw = tweets[tid]['parent']
            parent_idx = 'None'
            if parent_raw and parent_raw in id_map:
                parent_idx = str(id_map[parent_raw])
                if parent_idx == str(idx):  # guard against self loops
                    parent_idx = 'None'
            tree[idx] = {
                'parent': parent_idx,
                'text': tweets[tid]['text']
            }
        return tree
    
    def _get_label(self, thread_path, subset):
        normalized_subset = self._normalize_subset_name(subset)
        if normalized_subset == 'non-rumours':
            return 0
        if normalized_subset == 'rumours':
            return 1
        return None
    
    def load_raw_data(self):
        treeDic = {}
        labelDic = {}
        skipped = 0
        
        for event in self.events:
            event_path = os.path.join(self.data_root, event)
            if not os.path.isdir(event_path):
                continue
            potential_subsets = [
                d for d in os.listdir(event_path)
                if os.path.isdir(os.path.join(event_path, d))
            ]
            for subset in potential_subsets:
                normalized_subset = self._normalize_subset_name(subset)
                if normalized_subset not in ('rumours', 'non-rumours'):
                    continue
                subset_path = os.path.join(event_path, subset)
                
                threads = [
                    d for d in os.listdir(subset_path)
                    if os.path.isdir(os.path.join(subset_path, d))
                ]
                for thread_id in threads:
                    thread_path = os.path.join(subset_path, thread_id)
                    eid = f"{event}-{thread_id}"
                    try:
                        label = self._get_label(thread_path, normalized_subset)
                        if label is None:
                            skipped += 1
                            continue
                        tree = self._build_thread_tree(thread_path)
                        if len(tree) < 2:
                            skipped += 1
                            continue
                        treeDic[eid] = tree
                        labelDic[eid] = label
                    except Exception as exc:
                        print(f"Error parsing {eid}: {exc}")
                        skipped += 1
        
        print(f"PHEME loaded trees: {len(treeDic)}, labels: {len(labelDic)}, skipped: {skipped}")
        return treeDic, labelDic
    
    def process_data(self):
        print(f"\nProcessing {self.dataname}...")
        treeDic, labelDic = self.load_raw_data()
        valid_eids = [eid for eid in labelDic.keys() if eid in treeDic]
        
        if not valid_eids:
            raise ValueError("No valid event threads found for PHEME.")
        
        if self.sample_ratio < 1.0:
            num_samples = max(1, int(len(valid_eids) * self.sample_ratio))
            np.random.seed(42)
            valid_eids = np.random.choice(valid_eids, num_samples, replace=False).tolist()
        
        graph_list = []
        skipped = 0
        
        for eid in tqdm(valid_eids, desc="Processing"):
            try:
                tree = treeDic[eid]
                label = labelDic[eid]
                edge_index, num_nodes = construct_tree(tree)
                root_text = tree.get(1, {}).get('text', "empty tweet")
                texts = _extract_node_texts(tree, root_text)
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
            print(f"Skipped after processing: {skipped}")
        return graph_list
    
    def save_processed_data(self, graph_list, save_path=None):
        if save_path is None:
            save_path = Config.PROCESSED_DIR
        os.makedirs(save_path, exist_ok=True)
        
        filename = _build_processed_filename(self.dataname, self.sample_ratio)
        filepath = os.path.join(save_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(graph_list, f)
        print(f"Saved: {filepath}")
        return filepath
    
    def load_processed_data(self, load_path=None):
        if load_path is None:
            filename = _build_processed_filename(self.dataname, self.sample_ratio)
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
                texts = _extract_node_texts(tree, root_text)
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
        
        filename = _build_processed_filename(self.dataname, self.sample_ratio)
        
        filepath = os.path.join(save_path, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(graph_list, f)
        
        print(f"Saved: {filepath}")
        return filepath
    
    def load_processed_data(self, load_path=None):
        """Load processed data"""
        if load_path is None:
            filename = _build_processed_filename(self.dataname, self.sample_ratio)
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
