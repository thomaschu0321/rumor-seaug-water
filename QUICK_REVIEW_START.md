# Quick Start: Code Review Checklist

Use this checklist to systematically review your code. Check off items as you complete them.

## 🚀 Start Here (15 minutes)

- [v] **Read the README** - Understand the project overview and architecture
- [v] **Review project structure** - Familiarize yourself with file organization
- [v] **Check configuration** - Review `config.py` for settings and paths
- [v] **Understand the 4-phase pipeline** - Know what each phase does

## 📋 Phase-by-Phase Review

### Phase 1: Entry Points & Configuration
- [v] Review `run_experiments.py` - Experiment runner logic
- [v] Review `seaug_pipeline.py` - Main pipeline orchestrator  
- [ ] Review `config.py` - All configuration settings
- [ ] Check command-line argument parsing
- [ ] Verify environment variable handling

### Phase 2: Data Processing
- [v] Review `data_preprocessing.py` - Data loading and graph construction
- [ ] Check train/val/test split logic
- [ ] Verify data format handling (Twitter15, Twitter16, Weibo)
- [ ] Check data sampling functionality

### Phase 3: Feature Extraction (Phase 1)
- [v] Review `bert_feature_extractor.py` - BERT feature extraction
- [ ] Verify feature dimensions (should be 768-dim)
- [ ] Check batch processing efficiency
- [ ] Review GPU utilization

### Phase 4: Node Selection (Phase 2)
- [v] Review `node_selector.py` - DBSCAN and selection strategies
- [ ] Check outlier detection logic
- [ ] Verify selection strategies (uncertainty, importance, hybrid)
- [ ] Test edge cases (empty graphs, all outliers)

### Phase 5: Node Augmentation (Phase 3)
- [ ] Review `node_augmentor.py` - Augmentation logic
- [ ] Check LLM API integration (Azure OpenAI)
- [ ] Verify caching mechanism
- [ ] Check fallback to LM when LLM disabled
- [ ] Verify augmented feature dimensions (384-dim)

### Phase 6: Feature Fusion & Model (Phase 4)
- [ ] Review `feature_fusion.py` - Fusion strategies
- [aug not yet] Review `model_seaug.py` - GNN model architecture
- [ ] Verify GCN and GAT implementations
- [ ] Check feature fusion integration
- [ ] Verify final dimensions (1152 = 768 + 384)

## 🔍 Code Quality Checks

### Documentation
- [ ] All functions have docstrings
- [ ] Complex logic has inline comments
- [ ] README is up to date
- [ ] Type hints where appropriate

### Error Handling
- [ ] Try-except blocks for critical operations
- [ ] Meaningful error messages
- [ ] Graceful degradation (LLM disabled, etc.)
- [ ] Input validation

### Code Style
- [ ] Consistent naming conventions
- [ ] Proper indentation
- [ ] No unused imports
- [ ] PEP 8 compliance (run `pylint` or `flake8`)

## 🧪 Testing & Validation

- [ ] Run individual components:
  ```bash
  python bert_feature_extractor.py
  python node_selector.py
  python node_augmentor.py
  python feature_fusion.py
  python model_seaug.py
  ```

- [ ] Run full pipeline with sample data:
  ```bash
  python seaug_pipeline.py --dataset Twitter15 --sample_ratio 0.1
  ```

- [ ] Run experiment runner:
  ```bash
  python run_experiments.py --dataset Twitter15 --sample_ratio 0.1
  ```

- [ ] Check edge cases:
  - Empty graphs
  - Single-node graphs
  - All nodes selected
  - No nodes selected

## 🐛 Issues Found

Document any issues you find:

### Critical Issues
- [ ] Issue 1: _______________________
- [ ] Issue 2: _______________________

### Important Issues  
- [ ] Issue 1: _______________________
- [ ] Issue 2: _______________________

### Minor Issues
- [ ] Issue 1: _______________________
- [ ] Issue 2: _______________________

## 📝 Notes

Add any additional notes or observations:

_________________________________________________
_________________________________________________
_________________________________________________

---

**Review Date:** _______________
**Reviewer:** _______________

