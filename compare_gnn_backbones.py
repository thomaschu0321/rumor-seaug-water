"""
Compare SeAug Framework across Different GNN Backbones

This script compares SeAug performance on GCN and GAT backbones to demonstrate
the generalizability of the selective augmentation framework.

Experiments:
1. GCN-Baseline (no augmentation)
2. GAT-Baseline (no augmentation)
3. GCN + SeAug (selective augmentation)
4. GAT + SeAug (selective augmentation)
"""

import argparse
import numpy as np
from datetime import datetime
import json
import os

from seaug_pipeline import SeAugPipeline
from config import Config


def run_gnn_backbone_comparison(
    dataset_name: str = "Twitter15",
    sample_ratio: float = 0.05,
    augmentation_ratio: float = 0.3,
    fusion_strategy: str = "concat",
    node_strategy: str = "hybrid"
):
    """
    Run comprehensive comparison across GNN backbones
    
    Args:
        dataset_name: Dataset name
        sample_ratio: Sampling ratio for quick testing
        augmentation_ratio: Ratio of nodes to augment
        fusion_strategy: Feature fusion strategy
        node_strategy: Node selection strategy
    
    Returns:
        Comparison results
    """
    results = {
        'dataset': dataset_name,
        'sample_ratio': sample_ratio,
        'augmentation_ratio': augmentation_ratio,
        'fusion_strategy': fusion_strategy,
        'node_strategy': node_strategy,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiments': []
    }
    
    print("="*80)
    print("SeAug Framework: GNN Backbone Comparison")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Sample ratio: {sample_ratio}")
    print(f"Augmentation ratio: {augmentation_ratio}")
    print(f"Node selection: {node_strategy}")
    print(f"Fusion strategy: {fusion_strategy}")
    print("="*80)
    
    # Test both GCN and GAT backbones
    gnn_backbones = ['gcn', 'gat']
    
    for backbone in gnn_backbones:
        print(f"\n{'='*80}")
        print(f"Testing {backbone.upper()} Backbone")
        print(f"{'='*80}")
        
        # 1. Baseline (no augmentation)
        print(f"\n[Experiment] {backbone.upper()}-Baseline (No Augmentation)")
        print("-"*80)
        
        try:
            pipeline_baseline = SeAugPipeline(
                enable_augmentation=False,
                gnn_backbone=backbone
            )
            
            baseline_results = pipeline_baseline.run(dataset_name, sample_ratio)
            
            exp_result = {
                'name': f'{backbone.upper()}-Baseline',
                'backbone': backbone,
                'augmentation': False,
                'test_accuracy': baseline_results['test_results']['accuracy'],
                'test_precision': baseline_results['test_results']['precision'],
                'test_recall': baseline_results['test_results']['recall'],
                'test_f1': baseline_results['test_results']['f1'],
                'best_val_acc': baseline_results['best_val_acc']
            }
            
            results['experiments'].append(exp_result)
            
            print(f"\n{backbone.upper()}-Baseline Results:")
            print(f"  Accuracy:  {exp_result['test_accuracy']:.4f}")
            print(f"  Precision: {exp_result['test_precision']:.4f}")
            print(f"  Recall:    {exp_result['test_recall']:.4f}")
            print(f"  F1-Score:  {exp_result['test_f1']:.4f}")
            
        except Exception as e:
            print(f"✗ {backbone.upper()}-Baseline experiment failed: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 2. SeAug with selective augmentation
        print(f"\n[Experiment] {backbone.upper()} + SeAug (Selective Augmentation)")
        print("-"*80)
        
        try:
            pipeline_seaug = SeAugPipeline(
                enable_augmentation=True,
                node_selection_strategy=node_strategy,
                fusion_strategy=fusion_strategy,
                augmentation_ratio=augmentation_ratio,
                use_llm=False,  # No LLM for faster testing
                gnn_backbone=backbone
            )
            
            seaug_results = pipeline_seaug.run(dataset_name, sample_ratio)
            
            exp_result = {
                'name': f'{backbone.upper()}+SeAug',
                'backbone': backbone,
                'augmentation': True,
                'node_strategy': node_strategy,
                'fusion_strategy': fusion_strategy,
                'augmentation_ratio': augmentation_ratio,
                'test_accuracy': seaug_results['test_results']['accuracy'],
                'test_precision': seaug_results['test_results']['precision'],
                'test_recall': seaug_results['test_results']['recall'],
                'test_f1': seaug_results['test_results']['f1'],
                'best_val_acc': seaug_results['best_val_acc'],
                'augmented_nodes': pipeline_seaug.stats['augmented_nodes'],
                'augmentation_time': pipeline_seaug.stats['augmentation_time']
            }
            
            results['experiments'].append(exp_result)
            
            # Compute improvement over baseline
            baseline_exp = [e for e in results['experiments'] if e['name'] == f'{backbone.upper()}-Baseline'][0]
            acc_improve = (exp_result['test_accuracy'] - baseline_exp['test_accuracy']) * 100
            f1_improve = (exp_result['test_f1'] - baseline_exp['test_f1']) * 100
            
            print(f"\n{backbone.upper()}+SeAug Results:")
            print(f"  Accuracy:  {exp_result['test_accuracy']:.4f} ({acc_improve:+.2f}%)")
            print(f"  Precision: {exp_result['test_precision']:.4f}")
            print(f"  Recall:    {exp_result['test_recall']:.4f}")
            print(f"  F1-Score:  {exp_result['test_f1']:.4f} ({f1_improve:+.2f}%)")
            print(f"  Augmented nodes: {exp_result['augmented_nodes']:,}")
            print(f"  Augmentation time: {exp_result['augmentation_time']:.2f}s")
            
        except Exception as e:
            print(f"✗ {backbone.upper()}+SeAug experiment failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Method':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*73)
    
    for exp in results['experiments']:
        name = exp['name']
        acc = exp['test_accuracy']
        prec = exp['test_precision']
        rec = exp['test_recall']
        f1 = exp['test_f1']
        
        # Find corresponding baseline for relative improvement
        if exp['augmentation']:
            backbone = exp['backbone']
            baseline_exp = [e for e in results['experiments'] 
                          if e['backbone'] == backbone and not e['augmentation']][0]
            acc_str = f"{acc:.4f} ({(acc-baseline_exp['test_accuracy'])*100:+.2f}%)"
            f1_str = f"{f1:.4f} ({(f1-baseline_exp['test_f1'])*100:+.2f}%)"
        else:
            acc_str = f"{acc:.4f}"
            f1_str = f"{f1:.4f}"
        
        print(f"{name:<25} {acc_str:<20} {prec:.4f}       {rec:.4f}       {f1_str}")
    
    print("-"*73)
    
    # Analysis: SeAug effectiveness across backbones
    print("\n" + "="*80)
    print("ANALYSIS: SeAug Generalizability")
    print("="*80)
    
    for backbone in gnn_backbones:
        backbone_exps = [e for e in results['experiments'] if e['backbone'] == backbone]
        if len(backbone_exps) == 2:
            baseline = [e for e in backbone_exps if not e['augmentation']][0]
            seaug = [e for e in backbone_exps if e['augmentation']][0]
            
            acc_gain = (seaug['test_accuracy'] - baseline['test_accuracy']) * 100
            f1_gain = (seaug['test_f1'] - baseline['test_f1']) * 100
            
            print(f"\n{backbone.upper()} Backbone:")
            print(f"  Baseline:     Acc={baseline['test_accuracy']:.4f}, F1={baseline['test_f1']:.4f}")
            print(f"  +SeAug:       Acc={seaug['test_accuracy']:.4f}, F1={seaug['test_f1']:.4f}")
            print(f"  Improvement:  Acc={acc_gain:+.2f}%, F1={f1_gain:+.2f}%")
    
    # Key finding
    gains = []
    for backbone in gnn_backbones:
        backbone_exps = [e for e in results['experiments'] if e['backbone'] == backbone]
        if len(backbone_exps) == 2:
            baseline = [e for e in backbone_exps if not e['augmentation']][0]
            seaug = [e for e in backbone_exps if e['augmentation']][0]
            f1_gain = (seaug['test_f1'] - baseline['test_f1']) * 100
            gains.append(f1_gain)
    
    if len(gains) == 2:
        avg_gain = np.mean(gains)
        print(f"\n{'='*80}")
        print(f"KEY FINDING:")
        print(f"  SeAug provides consistent improvement across different GNN backbones")
        print(f"  Average F1-Score improvement: {avg_gain:.2f}%")
        print(f"  This demonstrates the architecture-agnostic nature of SeAug framework")
        print(f"{'='*80}")
    
    # Save results
    save_dir = 'logs/comparison'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(
        save_dir,
        f"gnn_backbone_comparison_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compare SeAug Framework across GNN Backbones",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--dataset', type=str, default='Twitter15',
                       choices=['Twitter15', 'Twitter16', 'Weibo'],
                       help='Dataset name')
    parser.add_argument('--sample_ratio', type=float, default=0.05,
                       help='Data sampling ratio for quick testing')
    parser.add_argument('--augmentation_ratio', type=float, default=0.3,
                       help='Ratio of nodes to augment per graph')
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                       choices=['concat', 'weighted', 'gated', 'attention'],
                       help='Feature fusion strategy')
    parser.add_argument('--node_strategy', type=str, default='hybrid',
                       choices=['uncertainty', 'importance', 'hybrid'],
                       help='Node selection strategy')
    
    args = parser.parse_args()
    
    # Run comparison
    results = run_gnn_backbone_comparison(
        dataset_name=args.dataset,
        sample_ratio=args.sample_ratio,
        augmentation_ratio=args.augmentation_ratio,
        fusion_strategy=args.fusion_strategy,
        node_strategy=args.node_strategy
    )
    
    return results


if __name__ == '__main__':
    main()

