# Graph-Based Rumor Detection with Selective Large Language Model Augmentation

## Overview

SeAug (Selective LLM Augmentation Pipeline) implements a 4-stage node-level augmentation pipeline for rumor detection on social media:

- **Phase 1**: BERT extracts 768-dim semantic features (replacing TF-IDF)
- **Phase 2**: DBSCAN identifies semantic outlier nodes (unsupervised)
- **Phase 3**: LLM+LM selectively augments key nodes
- **Phase 4**: Feature fusion + GNN classification

### Why This Works

**Assumption**: Semantic outlier tweets (sarcasm, misinformation injection) are crucial for graph classification. By selectively augmenting these nodes, we enhance the model's ability to distinguish between rumor and non-rumor propagation patterns.

---

## Installation

> **📘 Windows Users**: For detailed Windows setup instructions, see [WINDOWS_SETUP.md](WINDOWS_SETUP.md)

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster training)

### Setup

```bash
# 1. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Install additional dependencies for GNN
pip install transformers sentence-transformers torch-geometric
```

---

## Quick Start

### Run Experiments

The easiest way to get started is using the experiment runner:

```bash
# Run all 4 configurations (GAT/GCN baseline + SeAug)
python run_experiments.py --dataset Twitter15

# Run with sampling for quick testing
python run_experiments.py --dataset Twitter15 --sample_ratio 0.1
```

### Run Individual Pipeline

```bash
# Run SeAug pipeline with default settings
python seaug_pipeline.py --dataset Twitter15 --enable_augmentation

# Run with custom configuration
python seaug_pipeline.py \
    --dataset Twitter15 \
    --gnn_backbone gat \
    --enable_augmentation \
    --node_strategy uncertainty \
    --fusion_strategy concat \
    --sample_ratio 1.0
```

### Test Individual Modules

```bash
# Test Phase 1: BERT feature extraction
python bert_feature_extractor.py

# Test Phase 2: Node selection
python node_selector.py

# Test Phase 3: Node augmentation
python node_augmentor.py

# Test Phase 4a: Feature fusion
python feature_fusion.py

# Test Phase 4b: Model architecture
python model_seaug.py
```

---

## Architecture

```
Raw Tweets
    ↓
[Phase 1] BERT Feature Extraction
    → X_initial: 768-dim BERT features per node
    ↓
[Phase 2] DBSCAN Node Selection (Unsupervised)
    → Outlier nodes (label = -1)
    ↓
[Phase 3] LLM + LM Encoding
    → Augmented features: 384-dim per selected node
    ↓
[Phase 4] Feature Fusion + GNN
    → Fused features (768 + 384 = 1152-dim) → Classification
```

### Usage Example

```python
from bert_feature_extractor import BERTFeatureExtractor
from node_selector import NodeSelector
from node_augmentor import NodeAugmentor
from model_seaug import get_seaug_model

# Phase 1: Extract BERT features
extractor = BERTFeatureExtractor(model_name="bert-base-uncased")
bert_graphs = extractor.process_graph_list(graphs, texts)

# Phase 2: Select outlier nodes with DBSCAN
selector = NodeSelector(strategy="uncertainty", use_dbscan=True)
selector.fit(bert_graphs)
outlier_indices = [selector.select_nodes(g) for g in bert_graphs]

# Phase 3: Augment outlier nodes
augmentor = NodeAugmentor()
augmented_graphs = augmentor.augment_batch(bert_graphs, outlier_indices)

# Phase 4: GNN classification
model = get_seaug_model(
    model_type="seaug",
    gnn_backbone="gcn",
    baseline_dim=768,
    augmented_dim=384
)
output = model(augmented_graphs)
```

---

## Project Structure

### Core SeAug Modules

- `seaug_pipeline.py` - End-to-end pipeline orchestrator
- `bert_feature_extractor.py` - Phase 1: BERT feature extraction
- `node_selector.py` - Phase 2: DBSCAN node selection
- `node_augmentor.py` - Phase 3: LLM+LM augmentation
- `feature_fusion.py` - Phase 4a: Feature fusion strategies
- `model_seaug.py` - Phase 4b: SeAug GNN model (GCN/GAT)

### Infrastructure

- `config.py` - Project configuration
- `data_preprocessing.py` - Data preprocessing utilities
- `run_experiments.py` - Experiment runner for multiple configurations

### Utilities

- `utils/`
  - `visualization.py` - Training & result visualization

### Data Directories

- `data/`
  - `raw_text/` - Raw tweet/Weibo text files
  - `processed/` - Preprocessed graph data (.pkl files)
  - `llm_cache.pkl` - LLM response cache (auto-generated)
- `logs/` - Training logs & visualizations
  - `Twitter15/` - Twitter15 experiment logs
  - `Twitter16/` - Twitter16 experiment logs
- `checkpoints/` - Saved model checkpoints (auto-generated)

---

## Configuration

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

# Enable (LLM-powered) augmentation
--enable_augmentation

# GNN backbone selection
--gnn_backbone gcn|gat

# Node selection strategy
--node_strategy uncertainty|importance|hybrid

# Feature fusion strategy
--fusion_strategy concat|weighted|gated|attention

# Augmentation ratio
--augmentation_ratio 0.3      # Augment 30% of nodes per graph

```

---

## Datasets

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

### Data Setup

Ensure your data directory structure matches:

```
data/
  raw_text/
    Twitter15_source_tweets.txt
    Twitter16_source_tweets.txt
    Weibo/
      *.json files
```

---

## Technical Background

The SeAug framework combines several key techniques:

- **GCN & GAT**: Graph Neural Networks for modeling propagation structure
- **BERT**: Pre-trained language model for semantic feature extraction
- **DBSCAN**: Density-based clustering for identifying outlier nodes
- **Selective Augmentation**: Targeted enhancement of uncertain nodes

### Design Rationale

The framework supports both GCN and GAT backbones to demonstrate that the performance improvements come from selective augmentation rather than a specific GNN architecture. This design validates the generalizability of the approach across different neural network architectures.

---

## Results

Training results and visualizations are automatically saved to the `logs/` directory, including:

- Training history plots
- Confusion matrices
- Prediction analysis
- Results summary (accuracy, F1-score, etc.)

Each experiment run creates a timestamped subdirectory with all outputs.

---

## License

This project is part of a Final Year Project (FYP) for rumor detection research.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{seaug2024,
  title={SeAug: Selective LLM Augmentation for Rumor Detection},
  author={Your Name},
  year={2024}
}
```
