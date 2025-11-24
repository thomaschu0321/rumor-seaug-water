# Graph-Based Rumor Detection with Selective Large Language Model Augmentation

## Overview

SeAug implements a 4-stage node-level augmentation pipeline for rumor detection:

- **Phase 1**: BERT extracts 768-dim semantic features
- **Phase 2**: Adaptive node selection (uncertainty/importance/hybrid strategies)
- **Phase 3**: LLM+LM selectively augments key nodes
- **Phase 4**: Feature fusion + GNN classification

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Install PyTorch with CUDA 12.6 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file for LLM augmentation:

**Azure OpenAI:**
```bash
LLM_PROVIDER=azure
AZURE_API_KEY=your_api_key_here
AZURE_ENDPOINT=https://cuhk-apip.azure-api.net
AZURE_MODEL=gpt-4o-mini
API_VERSION=2023-05-15
```

**DeepSeek:**
```bash
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

## Quick Start

### Run Experiments

```bash
# Run all 4 configurations
python run_experiments.py --dataset Twitter15

# Run with sampling for quick testing
python run_experiments.py --dataset Twitter15 --sample_ratio 0.1
```

### Run Individual Pipeline

```bash
# Run SeAug pipeline
python seaug_pipeline.py --dataset Twitter15 --enable_augmentation

# Custom configuration
python seaug_pipeline.py \
    --dataset Twitter15 \
    --gnn_backbone gat \
    --enable_augmentation \
    --node_strategy hybrid \
    --fusion_strategy concat \
    --augmentation_ratio 0.3 \
    --batch_size 50
```

## Architecture

```
Raw Tweets
    ↓
[Phase 1] BERT Feature Extraction → 768-dim features
    ↓
[Phase 2] Adaptive Node Selection → Selects top-k nodes
    ↓
[Phase 3] LLM + LM Encoding → 384-dim augmented features
    ↓
[Phase 4] Feature Fusion + GNN → Classification
```

## Project Structure

- `seaug_pipeline.py` - Main pipeline orchestrator
- `bert_feature_extractor.py` - BERT feature extraction
- `node_selector.py` - Node selection strategies
- `node_augmentor.py` - LLM augmentation + LM encoding
- `feature_fusion.py` - Feature fusion strategies
- `model_seaug.py` - SeAug GNN model
- `config.py` - Configuration
- `data_preprocessing.py` - Data preprocessing
- `run_experiments.py` - Experiment runner

## Configuration

Key parameters in `config.py`:

```python
FEATURE_DIM = 768
NUM_CLASSES = 2
HIDDEN_DIM = 32
NUM_GNN_LAYERS = 2
DROPOUT = 0.7
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
```

## Datasets

| Dataset | Graphs | Nodes (avg) | Classes |
|---------|--------|-------------|---------|
| Twitter15 | 795 | ~23 | 2 |
| Twitter16 | 818 | ~25 | 2 |
| Weibo | 4,664 | ~18 | 2 |

## Results

Results are saved to `results/`:
- JSON results files
- Model checkpoints (`.pt` files)
- CSV summary (`results_summary.csv`)
