# TAPE Framework for Rumor Detection
## Overview

TAPE framework implements a **4-stage node-level augmentation pipeline** for rumor detection on social media:
- ✅ **Phase 1**: BERT extracts 768-dim semantic features (replacing TF-IDF)
- ✅ **Phase 2**: DBSCAN identifies semantic outlier nodes (unsupervised)
- ✅ **Phase 3**: LLM+LM selectively augments key nodes
- ✅ **Phase 4**: Feature fusion + GNN classification

python3 -m venv venv
cd /Users/nerwen/Downloads/RumorDetection_FYP && source venv/bin/activate
pip install --upgrade pip

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install transformers sentence-transformers torch-geometric  #for GNN

# 2. Extract BERT features
python bert_feature_extractor.py  #convert tweet into a 768-dimensional semantic vector. Results in data/processed/ directory.

# 3. Run quick comparison (Baseline vs TAPE)
python compare_tape_vs_baseline.py --mode quick --sample_ratio 0.05
```


### Test Individual Modules

```bash
python bert_feature_extractor.py   # Test Phase 1
python node_selector.py            # Test Phase 2  
python node_augmentor.py           # Test Phase 3
python feature_fusion.py           # Test Phase 4a
python model_tape.py               # Test Phase 4b
python tape_pipeline.py 
```

---

## 🏗️ Architecture

Raw Tweets
    
[Phase 1] BERT Feature Extraction
    → X_initial: 768-dim BERT features per node
    
[Phase 2] DBSCAN Node Selection (Unsupervised)
    → Outlier nodes (label = -1)
    
[Phase 3] LLM + LM Encoding
    → Augmented features: 384-dim per selected node

[Phase 4] Feature Fusion + GNN
    → Fused features (768 + 384 = 1152-dim) → Classification
```

### Why This Works

**Assumption**: Semantic outlier tweets (sarcasm, misinformation injection) are crucial for graph classification.


## 📦 Project Structure

<details open>
<summary><b>Click to expand/collapse project structure</b></summary>

```
RumorDetection_FYP/
│
├── 🎯 Core TAPE Modules (7 files)
│   ├── bert_feature_extractor.py    # Phase 1: BERT feature extraction
│   ├── node_selector.py             # Phase 2: DBSCAN node selection
│   ├── node_augmentor.py            # Phase 3: LLM+LM augmentation
│   ├── feature_fusion.py            # Phase 4a: Feature fusion strategies
│   ├── model_tape.py                # Phase 4b: TAPE GNN model (GCN/GAT)
│   ├── tape_pipeline.py             # End-to-end pipeline orchestrator
│   └── prompts.py                   # LLM prompt templates
│
├── 🧪 Experiments & Testing (3 files)
│   ├── compare_tape_vs_baseline.py  # Baseline vs TAPE comparison
│   ├── compare_gnn_backbones.py     # GCN vs GAT backbone comparison
│   └── test_gat.py                  # GAT implementation tests
│
├── 🔧 Infrastructure (3 files)
│   ├── config.py                    # Project configuration
│   ├── data_preprocessing.py        # Data preprocessing utilities
│   └── rate_limiter.py              # API rate limiting for LLM calls
│
├── 📊 Utilities
│   └── utils/
│       ├── __init__.py              # Package initialization
│       └── visualization.py         # Training & result visualization
│
├── 📚 Documentation
│   ├── README.md                    # Main documentation (this file)
│   ├── VISUALIZATION_GUIDE.md       # Visualization usage guide
│   ├── GAT_IMPLEMENTATION.md        # GAT implementation details
│   └── requirements.txt             # Python dependencies
│
├── 📁 Data Directories
│   ├── data/
│   │   ├── raw/                     # Raw datasets (Twitter15/16, Weibo)
│   │   ├── processed/               # Preprocessed graph data (.pkl)
│   │   └── llm_cache.pkl            # LLM response cache
│   │
│   ├── checkpoints/                 # Saved model checkpoints
│   │   ├── Twitter15_tape_best.pt
│   │   └── Twitter16_tape_best.pt
│   │
│   └── logs/                        # Training logs & visualizations
│       ├── Twitter15/               # Twitter15 experiment logs
│       └── Twitter16/               # Twitter16 experiment logs
│
└── 📖 docs/                         # Additional documentation
    ├── README.md                    # Documentation archive info
    ├── GAT_IMPLEMENTATION_SUMMARY.md
    └── GAT_USAGE_GUIDE.md
```

### 📋 File Categories

| Category | Files | Purpose |
|----------|-------|---------|
| **Core Modules** | 7 files | TAPE framework implementation (Phases 1-4) |
| **Experiments** | 3 files | Comparison scripts and testing |
| **Infrastructure** | 3 files | Configuration and utilities |
| **Documentation** | 4 files | User guides and technical docs |

</details>

# Phase 1: Extract BERT features
extractor = BERTFeatureExtractor(model_name="bert-base-uncased")
bert_graphs = extractor.process_graph_list(graphs, texts)

# Phase 2: Select outlier nodes with DBSCAN
selector = NodeSelector(strategy="uncertainty", use_dbscan=True)
selector.fit(bert_graphs)
outlier_indices = [selector.select_nodes(g) for g in bert_graphs]

# Phase 3: Augment outlier nodes
augmentor = NodeAugmentor(use_llm=False)
augmented_graphs = augmentor.augment_batch(bert_graphs, outlier_indices)

# Phase 4: GNN classification
model = TAPERumorGCN(baseline_dim=768, augmented_dim=384)
output = model(augmented_graphs)
```

---

## ⚙️ Configuration

### Key Parameters in `config.py`

```python
# Phase 1: BERT Features
BERT_MODEL_NAME = "bert-base-uncased"  # 768-dim
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 32

# Phase 2: DBSCAN Node Selection
DBSCAN_EPS = 0.5              # Epsilon for DBSCAN
DBSCAN_MIN_SAMPLES = 5        # Minimum samples for core points

# Phase 3: LM Augmentation
AUGMENTED_DIM = 384           # Sentence-BERT dimension
LM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Phase 4: Feature Fusion & GNN
BASELINE_DIM = 768            # BERT dimension
FUSED_DIM = 1152              # 768 + 384
HIDDEN_DIM = 64               # GNN hidden dimension
NUM_GCN_LAYERS = 2
DROPOUT = 0.3

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
```

### Command Line Arguments

```bash
# Dataset selection
--dataset Twitter15|Twitter16|Weibo

# Sampling (for quick testing)
--sample_ratio 0.05           # Use 5% of data

# Enable augmentation
--enable_augmentation

# Node selection strategy
--node_strategy uncertainty|importance|hybrid

# Feature fusion strategy
--fusion_strategy concat|weighted|gated|attention

# Augmentation ratio
--augmentation_ratio 0.3      # Augment 30% of nodes per graph

# Use LLM (requires API key)
--use_llm
```

---

## 📊 Datasets

### Supported Datasets

| Dataset | Graphs | Nodes (avg) | Classes | Description |
|---------|--------|-------------|---------|-------------|
| **Twitter15** | 795 | ~23 | 4 | True/False/Unverified/Non-rumor |
| **Twitter16** | 818 | ~25 | 4 | Same as Twitter15 |
| **Weibo** | 4,664 | ~18 | 2 | Rumor/Non-rumor |

### Data Format

- **Node features**: BERT 768-dim semantic embeddings
- **Graph structure**: Propagation tree (edge_index)
- **Labels**: Graph-level classification

---


## 🎓 Academic Background

TAPE framework combines advantages from:

- **GCN & GAT**: Graph Neural Networks for structure modeling
- **BERT**: Pre-trained language model for semantic understanding
- **DBSCAN**: Density-based clustering for unsupervised anomaly detection
- **Selective Augmentation**: Node-level targeted enhancement

### Why Multiple GNN Backbones?

We support both GCN and GAT to demonstrate that TAPE is a **generalizable framework** that works across different GNN architectures. This validates that the performance gain comes from selective augmentation, not from a specific GNN choice.

### Key Publications

This framework is inspired by recent advances in:
1. Graph Neural Networks for fake news detection
2. Pre-trained language models for social media analysis
3. Selective data augmentation techniques

-