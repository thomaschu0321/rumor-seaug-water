"""
Visualization utilities for training results and model evaluation
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_history(history, save_path):
    """
    Plot training history curves
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc', 'train_f1', 'val_f1'
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curve
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'orange', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'orange', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score curve
    axes[2].plot(epochs, history['train_f1'], 'b-', label='Train F1', linewidth=2)
    axes[2].plot(epochs, history['val_f1'], 'orange', label='Val F1', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('F1-Score', fontsize=12)
    axes[2].set_title('F1-Score Curve', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Training history plot saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})
    
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('True Label', fontsize=13, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=15, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Confusion matrix saved to: {save_path}")


def plot_prediction_analysis(y_true, y_pred, y_prob, save_path, confidence_threshold=0.6):
    """
    Plot prediction confidence analysis
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (for positive class)
        save_path: Path to save the plot
        confidence_threshold: Threshold for low/high confidence
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Determine correctness
    correct = (y_true == y_pred)
    
    # Plot 1: Confidence distribution
    axes[0].hist(y_prob[correct], bins=30, alpha=0.7, color='green', 
                label='Correct Predictions', edgecolor='black')
    axes[0].hist(y_prob[~correct], bins=30, alpha=0.7, color='red', 
                label='Incorrect Predictions', edgecolor='black')
    axes[0].axvline(confidence_threshold, color='black', linestyle='--', 
                   linewidth=2, label=f'Threshold={confidence_threshold}')
    axes[0].set_xlabel('Confidence', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Performance by confidence level
    low_conf_mask = y_prob < confidence_threshold
    high_conf_mask = y_prob >= confidence_threshold
    
    low_conf_acc = np.mean(correct[low_conf_mask]) if np.any(low_conf_mask) else 0
    high_conf_acc = np.mean(correct[high_conf_mask]) if np.any(high_conf_mask) else 0
    
    low_conf_count = np.sum(low_conf_mask)
    high_conf_count = np.sum(high_conf_mask)
    
    x = ['Low Confidence', 'High Confidence']
    accuracies = [low_conf_acc, high_conf_acc]
    counts = [low_conf_count, high_conf_count]
    
    ax2 = axes[1]
    bars = ax2.bar(x, accuracies, color=['lightcoral', 'lightblue'], 
                   edgecolor='black', linewidth=2, alpha=0.8)
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold', color='steelblue')
    ax2.set_ylim([0, 1.0])
    ax2.set_title('Performance by Confidence Level', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add sample count on secondary y-axis
    ax2_twin = ax2.twinx()
    ax2_twin.plot(x, counts, 'o-', color='orange', linewidth=3, 
                 markersize=10, label='Sample Count')
    ax2_twin.set_ylabel('Sample Count', fontsize=12, fontweight='bold', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    # Add count labels
    for i, (xi, count) in enumerate(zip(x, counts)):
        ax2_twin.text(i, count, f'{count}', 
                     ha='center', va='bottom', fontsize=11, 
                     fontweight='bold', color='orange')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Prediction analysis saved to: {save_path}")


def save_results_summary(results, history, save_path, dataset_name, config_info=None):
    """
    Save text summary of results
    
    Args:
        results: Dictionary with test results
        history: Training history
        save_path: Path to save the text file
        dataset_name: Name of the dataset
        config_info: Optional configuration information
    """
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"Model Evaluation Results Summary - {dataset_name}\n")
        f.write("="*70 + "\n\n")
        
        # Test results
        f.write("Test Set Results:\n")
        f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
        f.write(f"  Precision: {results['precision']:.4f}\n")
        f.write(f"  Recall:    {results['recall']:.4f}\n")
        f.write(f"  F1-Score:  {results['f1']:.4f}\n\n")
        
        # Training process
        f.write("Training Process:\n")
        f.write(f"  Total Epochs: {len(history['train_loss'])}\n")
        f.write(f"  Final Training Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Final Validation Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Best Validation Accuracy: {max(history['val_acc']):.4f}\n")
        f.write(f"  Best Validation F1: {max(history['val_f1']):.4f}\n\n")
        
        # Configuration info
        if config_info:
            f.write("Configuration:\n")
            for key, value in config_info.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"  Results summary saved to: {save_path}")


def create_log_directory(base_dir, dataset_name):
    """
    Create timestamped log directory
    
    Args:
        base_dir: Base logs directory
        dataset_name: Name of the dataset
    
    Returns:
        Path to the created directory and timestamp
    """
    timestamp = datetime.now().strftime("%Y.%m.%d.%H:%M:%S")
    log_dir = os.path.join(base_dir, dataset_name, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, timestamp



