# Visualization and Logging Guide

This document explains the visualization and logging features added to the SeAug framework.

---

## 📁 Directory Structure

After running experiments, logs will be saved in the following structure:

```
RumorDetection_FYP/
├── logs/
│   ├── Twitter15/
│   │   └── 2025.11.11.15:30:45/
│   │       ├── training_history_2025.11.11.15:30:45.png
│   │       ├── confusion_matrix_2025.11.11.15:30:45.png
│   │       ├── prediction_analysis_2025.11.11.15:30:45.png
│   │       └── results_summary_2025.11.11.15:30:45.txt
│   ├── Twitter16/
│   ├── Weibo/
│   └── comparison/
│       └── 2025.11.11.16:00:00/
│           ├── comparison_results_Twitter15_20251111_160000.json
│           └── comparison_summary_20251111_160000.txt
└── comparison_results_Twitter15_20251111_160000.json  (backup)
```

---

## 📊 Visualization Outputs

### 1. **Training History Plot** (`training_history_*.png`)

**Shows:**
- Training and validation Loss curves
- Training and validation Accuracy curves
- Training and validation F1-Score curves

**Purpose:**
- Diagnose overfitting/underfitting
- Verify early stopping effectiveness
- Monitor convergence speed

---

### 2. **Confusion Matrix** (`confusion_matrix_*.png`)

**Shows:**
- Classification accuracy for each class
- Misclassification patterns

**Purpose:**
- Identify which classes are easily confused
- Detect model bias towards certain classes
- Evaluate per-class performance

---

### 3. **Prediction Analysis** (`prediction_analysis_*.png`)

**Shows:**
- Left plot: Confidence distribution for correct vs incorrect predictions
- Right plot: Accuracy by confidence level (low/high)

**Purpose:**
- Identify low-confidence predictions that need manual review
- Evaluate model calibration
- Find uncertain samples

---

### 4. **Results Summary** (`results_summary_*.txt`)

**Contains:**
- Test set metrics (Accuracy, Precision, Recall, F1)
- Training process statistics
- Configuration parameters


### Running Single Experiment

```bash
# Run SeAug pipeline with augmentation
python seaug_pipeline.py --dataset Twitter15 --sample_ratio 0.05 --enable_augmentation

# Results will be saved to: logs/Twitter15/YYYY.MM.DD.HH:MM:SS/
```

### Running Comparison Experiments

```bash
# Compare baseline vs SeAug
python compare_seaug_vs_baseline.py --dataset Twitter15 --sample_ratio 0.05 --mode quick

