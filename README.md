# Graph-Based Rumor Detection with Selective Large Language Model Augmentation

## Overview

SeAug (Selective LLM Augmentation Pipeline) implements a 4-stage node-level augmentation pipeline for rumor detection on social media:

- **Phase 1**: BERT extracts 768-dim semantic features (replacing TF-IDF)
- **Phase 2**: Adaptive node selection (uncertainty/importance/hybrid strategies)
- **Phase 3**: LLM+LM selectively augments key nodes
- **Phase 4**: Feature fusion + GNN classification

### Why This Works

**Assumption**: Semantic outlier tweets (sarcasm, misinformation injection) are crucial for graph classification. By selectively augmenting these nodes, we enhance the model's ability to distinguish between rumor and non-rumor propagation patterns.

---

## Installation

### Prerequisites

- Python 3.8+ (tested with Python 3.11)
- CUDA-capable GPU (recommended for faster training)
- CUDA 12.6 compatible GPU (for CUDA support)

### Setup

```bash
# 1. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install PyTorch with CUDA 12.6 support FIRST
# This is critical for GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 3. Install remaining dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"
```

### Environment Configuration

For LLM augmentation (optional but recommended), create a `.env` file in the project root:

```bash
# Azure OpenAI API Configuration
AZURE_API_KEY=your_api_key_here
AZURE_ENDPOINT=https://cuhk-apip.azure-api.net
AZURE_MODEL=gpt-4o-mini
API_VERSION=2023-05-15

# Optional LLM parameters
LLM_MAX_TOKENS=500
LLM_TEMPERATURE=0.7
LLM_BATCH_SIZE=20
USE_LLM=true
```

**Note**: The project uses Azure OpenAI API for node-level text augmentation. If you don't have API access, the pipeline will still work but augmentation will be disabled.

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
    --node_strategy hybrid \
    --fusion_strategy concat \
    --augmentation_ratio 0.3 \
    --batch_size 20 \
    --sample_ratio 1.0
```

**Key Parameters**:
- `--batch_size`: Number of nodes per LLM API call (default: 20, recommended: 10-20). Higher values reduce API calls and token usage.
- `--augmentation_ratio`: Ratio of nodes to augment per graph (default: 0.3)
- `--node_strategy`: Selection strategy - `uncertainty` (DBSCAN-based), `importance` (structural), or `hybrid` (default)

### Test Individual Modules

Individual modules can be tested by running them directly (they include test code in `__main__` blocks):

```bash
# Test Phase 3: Node augmentation (includes LM encoder test)
python node_augmentor.py

# Test Phase 4a: Feature fusion
python feature_fusion.py
```

**Note**: Other modules are integrated into the main pipeline and don't have standalone test scripts.

---

## Architecture

```
Raw Tweets
    ↓
[Phase 1] BERT Feature Extraction
    → X_initial: 768-dim BERT features per node (bert-base-uncased)
    ↓
[Phase 2] Adaptive Node Selection
    → Strategy: uncertainty (DBSCAN-based) / importance (structural) / hybrid
    → Selects top-k nodes per graph (default: 30% of nodes)
    ↓
[Phase 3] LLM + LM Encoding
    → LLM paraphrases selected node texts (batched API calls)
    → LM encodes augmented texts: 384-dim per node (all-MiniLM-L6-v2)
    ↓
[Phase 4] Feature Fusion + GNN
    → Fusion: concat / weighted / gated / attention
    → GNN: GCN or GAT backbone
    → Classification: 2 classes (non-rumor vs rumor)
```

### Pipeline Flow

1. **Data Loading**: Loads raw tweet/Weibo data and extracts BERT features (768-dim) for all nodes
2. **Node Selection**: Uses hybrid strategy combining DBSCAN-based uncertainty scores and structural importance (degree centrality, root position)
3. **Augmentation**: Batches selected nodes into single LLM API calls (token-efficient), then encodes augmented texts with language model
4. **Training**: Fuses baseline (768-dim) and augmented (384-dim) features, then trains GNN classifier

### Usage Example

```python
from seaug_pipeline import SeAugPipeline

# Create pipeline
pipeline = SeAugPipeline(
    enable_augmentation=True,
    node_selection_strategy="hybrid",
    fusion_strategy="concat",
    augmentation_ratio=0.3,
    gnn_backbone="gcn",
    batch_size=20  # Nodes per LLM API call
)

# Run complete pipeline
results = pipeline.run(
    dataset_name="Twitter15",
    sample_ratio=1.0
)

# Access results
print(f"Test Accuracy: {results['test_results']['accuracy']:.4f}")
print(f"Test F1: {results['test_results']['f1']:.4f}")
```

---

## Project Structure

### Core SeAug Modules

- `seaug_pipeline.py` - End-to-end pipeline orchestrator (main entry point)
- `bert_feature_extractor.py` - Phase 1: BERT feature extraction (768-dim)
- `node_selector.py` - Phase 2: Adaptive node selection (uncertainty/importance/hybrid)
- `node_augmentor.py` - Phase 3: LLM augmentation + LM encoding (384-dim)
- `feature_fusion.py` - Phase 4a: Feature fusion strategies (concat/weighted/gated/attention)
- `model_seaug.py` - Phase 4b: SeAug GNN model (GCN/GAT backbones)

### Infrastructure

- `config.py` - Project configuration (Config class with all hyperparameters)
- `data_preprocessing.py` - Data preprocessing utilities (Twitter/Weibo processors)
- `run_experiments.py` - Experiment runner for 4 configurations (GAT/GCN baseline + SeAug)

### Data Directories

- `data/`
  - `Twitter/` - Twitter dataset files
    - `Twitter15/` - Twitter15 data files
    - `Twitter16/` - Twitter16 data files
  - `Weibo/` - Weibo dataset files
  - `processed/` - Preprocessed graph data (.pkl files, auto-generated)
- `checkpoints/` - Saved model checkpoints (auto-generated)
- `results_summary.csv` - Experiment results summary (auto-generated)

---

## Configuration

### Key Parameters in `config.py`

The `Config` class contains all hyperparameters. Key settings:

```python
# Phase 1: BERT Features
FEATURE_DIM = 768             # BERT feature dimension (bert-base-uncased)
NUM_CLASSES = 2               # Binary classification (non-rumor vs rumor)

# Phase 2: Node Selection (configured via command line)
# - Strategy: uncertainty / importance / hybrid
# - Top-k ratio: 0.3 (30% of nodes per graph)
# - Min/Max nodes: 1-10 per graph

# Phase 3: LM Augmentation
# - Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
# - LLM: Azure OpenAI (gpt-4o-mini)
# - Batch size: 20 nodes per API call (token-efficient)

# Phase 4: Feature Fusion & GNN
HIDDEN_DIM = 32               # GNN hidden dimension
NUM_GNN_LAYERS = 2            # Number of GNN layers
DROPOUT = 0.7                 # Dropout rate
GAT_HEADS = 4                 # GAT attention heads

# Training
BATCH_SIZE = 32                # Training batch size
LEARNING_RATE = 0.001         # Learning rate
WEIGHT_DECAY = 1e-3           # L2 regularization
NUM_EPOCHS = 100              # Max training epochs
PATIENCE = 5                  # Early stopping patience

# Data Split
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
```

### Command Line Arguments

#### `seaug_pipeline.py`

```bash
# Dataset selection
--dataset Twitter15|Twitter16|Weibo

# Sampling (for quick testing)
--sample_ratio 0.1            # Use 10% of data

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

# LLM API batch size (token efficiency)
--batch_size 20               # Nodes per API call (default: 20, recommended: 10-20)
```

#### `run_experiments.py`

```bash
# Run all 4 experiments
python run_experiments.py --dataset Twitter15

# Run specific experiments only
python run_experiments.py --only GAT_Baseline GCN_Baseline

# Skip specific experiments
python run_experiments.py --skip SeAug_GAT

# Custom parameters
python run_experiments.py \
    --dataset Twitter16 \
    --augmentation_ratio 0.2 \
    --node_strategy hybrid \
    --batch_size 15 \
    --sample_ratio 0.5
```

---

## Datasets

### Supported Datasets

| Dataset | Graphs | Nodes (avg) | Classes | Description |
|---------|--------|-------------|---------|-------------|
| **Twitter15** | 795 | ~23 | 2 | Binary: Non-rumor (0) vs Rumor (1) |
| **Twitter16** | 818 | ~25 | 2 | Binary: Non-rumor (0) vs Rumor (1) |
| **Weibo** | 4,664 | ~18 | 2 | Binary: Non-rumor (0) vs Rumor (1) |

**Note**: Twitter15/16 are converted from 4-class (True/False/Unverified/Non-rumor) to binary classification (Non-rumor vs Rumor) during preprocessing.

### Data Format

- **Node features**: BERT 768-dim semantic embeddings (from bert-base-uncased)
- **Graph structure**: Propagation tree (edge_index in PyG format)
- **Labels**: Graph-level binary classification (0: non-rumor, 1: rumor)

### Data Setup

Ensure your data directory structure matches:

```
data/
  Twitter/
    Twitter15/
      data.TD_RvNN.vol_5000.txt
      Twitter15_label_All.txt
      Twitter15_source_tweets.txt
    Twitter16/
      data.TD_RvNN.vol_5000.txt
      Twitter16_label_All.txt
      Twitter16_source_tweets.txt
  Weibo/
    weibotree.txt
    weibo_id_label.txt
  processed/          # Auto-generated (processed .pkl files)
```

The preprocessing pipeline automatically:
1. Loads raw data from the above structure
2. Extracts BERT features for all nodes
3. Saves processed graphs to `data/processed/` as `.pkl` files
4. Reuses processed data on subsequent runs (faster startup)

---

## Technical Background

The SeAug framework combines several key techniques:

- **GCN & GAT**: Graph Neural Networks for modeling propagation structure
- **BERT**: Pre-trained language model (bert-base-uncased) for 768-dim semantic feature extraction
- **Adaptive Node Selection**: Hybrid strategy combining:
  - **Uncertainty-based**: DBSCAN clustering to identify semantic outliers
  - **Importance-based**: Structural metrics (degree centrality, root position)
- **Selective LLM Augmentation**: Batched API calls for token-efficient text paraphrasing
- **Language Model Encoding**: Sentence transformer (all-MiniLM-L6-v2) for 384-dim augmented features
- **Feature Fusion**: Multiple strategies (concat, weighted, gated, attention) to combine baseline and augmented features

### Design Rationale

1. **Dual GNN Backbones**: Supports both GCN and GAT to demonstrate that performance improvements come from selective augmentation rather than a specific GNN architecture.

2. **Hybrid Node Selection**: Combines semantic uncertainty (DBSCAN) and structural importance to identify nodes that benefit most from augmentation.

3. **Token-Efficient Batching**: Batches multiple nodes into single LLM API calls, reducing token usage by ~58% compared to individual calls.

4. **Flexible Fusion**: Multiple fusion strategies allow the model to learn optimal ways to combine baseline and augmented features.

---

## Results

Training results are automatically saved:

### Output Files

1. **JSON Results** (`checkpoints/{dataset}_results_{timestamp}.json`):
   - Complete experiment configuration
   - Training history (loss, accuracy, F1 per epoch)
   - Test set metrics (accuracy, precision, recall, F1)
   - Pipeline statistics (nodes augmented, augmentation time)

2. **CSV Summary** (`results_summary.csv`):
   - Appends each experiment run as a row
   - Includes all configuration parameters and final metrics
   - Useful for comparing multiple experiments

3. **Model Checkpoints** (`checkpoints/{dataset}_seaug_best.pt`):
   - Best model weights (based on validation accuracy)
   - Can be loaded for inference or further training

### Metrics

The pipeline reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Precision for rumor class (binary)
- **Recall**: Recall for rumor class (binary)
- **F1-Score**: F1-score for rumor class (binary)
- **Training History**: Loss, accuracy, and F1 per epoch
- **Augmentation Stats**: Number of nodes augmented, API call statistics

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
