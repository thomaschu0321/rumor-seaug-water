"""
TAPE vs Baseline Comparison Experiment Script

This script compares the performance of:
1. Baseline GCN (no augmentation)
2. TAPE Framework (with node-level augmentation)
"""

import argparse
import numpy as np
from datetime import datetime
import json

from tape_pipeline import TAPEPipeline
from config import Config


def run_comparison(
    dataset_name: str = "Twitter15",
    sample_ratio: float = 0.05,
    fusion_strategies: list = None,
    node_strategies: list = None
):
    """
    Run comprehensive comparison
    
    Args:
        dataset_name: Dataset name
        sample_ratio: Sampling ratio
        fusion_strategies: List of fusion strategies to test
        node_strategies: List of node selection strategies to test
    
    Returns:
        Comparison results
    """
    if fusion_strategies is None:
        fusion_strategies = ["concat", "weighted", "gated"]
    
    if node_strategies is None:
        node_strategies = ["hybrid"]
    
    results = {
        'dataset': dataset_name,
        'sample_ratio': sample_ratio,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'experiments': []
    }
    
    print("="*70)
    print("TAPE vs Baseline Comparison")
    print("="*70)
    print(f"Dataset: {dataset_name}")
    print(f"Sample ratio: {sample_ratio}")
    print("="*70)
    
    # 1. Baseline (no augmentation)
    print("\n" + "="*70)
    print("Experiment 1: Baseline (No Augmentation)")
    print("="*70)
    
    try:
        pipeline_baseline = TAPEPipeline(
            enable_augmentation=False
        )
        
        baseline_results = pipeline_baseline.run(dataset_name, sample_ratio)
        
        exp_result = {
            'name': 'Baseline',
            'augmentation': False,
            'test_accuracy': baseline_results['test_results']['accuracy'],
            'test_precision': baseline_results['test_results']['precision'],
            'test_recall': baseline_results['test_results']['recall'],
            'test_f1': baseline_results['test_results']['f1'],
            'best_val_acc': baseline_results['best_val_acc']
        }
        
        results['experiments'].append(exp_result)
        
        print(f"\n✓ Baseline Results:")
        print(f"  Accuracy:  {exp_result['test_accuracy']:.4f}")
        print(f"  Precision: {exp_result['test_precision']:.4f}")
        print(f"  Recall:    {exp_result['test_recall']:.4f}")
        print(f"  F1-Score:  {exp_result['test_f1']:.4f}")
        
    except Exception as e:
        print(f"✗ Baseline experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 2. TAPE with different configurations
    exp_num = 2
    
    for node_strat in node_strategies:
        for fusion_strat in fusion_strategies:
            print("\n" + "="*70)
            print(f"Experiment {exp_num}: TAPE Framework")
            print(f"  Node selection: {node_strat}")
            print(f"  Fusion strategy: {fusion_strat}")
            print("="*70)
            
            try:
                pipeline_tape = TAPEPipeline(
                    enable_augmentation=True,
                    node_selection_strategy=node_strat,
                    fusion_strategy=fusion_strat,
                    augmentation_ratio=0.3,
                    use_llm=False  # Don't use LLM for speed
                )
                
                tape_results = pipeline_tape.run(dataset_name, sample_ratio)
                
                exp_result = {
                    'name': f'TAPE-{fusion_strat}-{node_strat}',
                    'augmentation': True,
                    'node_strategy': node_strat,
                    'fusion_strategy': fusion_strat,
                    'test_accuracy': tape_results['test_results']['accuracy'],
                    'test_precision': tape_results['test_results']['precision'],
                    'test_recall': tape_results['test_results']['recall'],
                    'test_f1': tape_results['test_results']['f1'],
                    'best_val_acc': tape_results['best_val_acc'],
                    'augmented_nodes': pipeline_tape.stats['augmented_nodes'],
                    'augmentation_time': pipeline_tape.stats['augmentation_time']
                }
                
                results['experiments'].append(exp_result)
                
                # Compute improvement
                acc_improve = (exp_result['test_accuracy'] - results['experiments'][0]['test_accuracy']) * 100
                f1_improve = (exp_result['test_f1'] - results['experiments'][0]['test_f1']) * 100
                
                print(f"\n✓ TAPE Results:")
                print(f"  Accuracy:  {exp_result['test_accuracy']:.4f} ({acc_improve:+.2f}%)")
                print(f"  Precision: {exp_result['test_precision']:.4f}")
                print(f"  Recall:    {exp_result['test_recall']:.4f}")
                print(f"  F1-Score:  {exp_result['test_f1']:.4f} ({f1_improve:+.2f}%)")
                print(f"  Augmented nodes: {exp_result['augmented_nodes']:,}")
                print(f"  Augmentation time: {exp_result['augmentation_time']:.2f}s")
                
            except Exception as e:
                print(f"✗ TAPE experiment {exp_num} failed: {e}")
                import traceback
                traceback.print_exc()
            
            exp_num += 1
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"\n{'Method':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*78)
    
    baseline_acc = results['experiments'][0]['test_accuracy']
    baseline_f1 = results['experiments'][0]['test_f1']
    
    for exp in results['experiments']:
        name = exp['name']
        acc = exp['test_accuracy']
        prec = exp['test_precision']
        rec = exp['test_recall']
        f1 = exp['test_f1']
        
        if exp['augmentation']:
            acc_str = f"{acc:.4f} ({(acc-baseline_acc)*100:+.2f}%)"
            f1_str = f"{f1:.4f} ({(f1-baseline_f1)*100:+.2f}%)"
        else:
            acc_str = f"{acc:.4f}"
            f1_str = f"{f1:.4f}"
        
        print(f"{name:<30} {acc_str:<12} {prec:.4f}       {rec:.4f}       {f1_str}")
    
    print("-"*78)
    
    # Find best
    best_exp = max(results['experiments'][1:], key=lambda x: x['test_f1'])
    print(f"\n🏆 Best Configuration: {best_exp['name']}")
    print(f"   F1-Score: {best_exp['test_f1']:.4f}")
    print(f"   Improvement over baseline: {(best_exp['test_f1']-baseline_f1)*100:.2f}%")
    
    # Save results (two locations for compatibility)
    import os
    from config import Config
    
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Save to project root (for backward compatibility)
    save_path_root = f"comparison_results_{dataset_name}_{timestamp_str}.json"
    with open(save_path_root, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 2. Save to logs directory
    log_dir = os.path.join(Config.LOG_DIR, 'comparison', datetime.now().strftime('%Y.%m.%d.%H:%M:%S'))
    os.makedirs(log_dir, exist_ok=True)
    save_path_log = os.path.join(log_dir, f'comparison_results_{dataset_name}_{timestamp_str}.json')
    with open(save_path_log, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save a text summary
    summary_path = os.path.join(log_dir, f'comparison_summary_{timestamp_str}.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"TAPE vs Baseline Comparison - {dataset_name}\n")
        f.write("="*70 + "\n\n")
        
        for exp in results['experiments']:
            f.write(f"\n{exp['name']}:\n")
            f.write(f"  Accuracy:  {exp['test_accuracy']:.4f}\n")
            f.write(f"  Precision: {exp['test_precision']:.4f}\n")
            f.write(f"  Recall:    {exp['test_recall']:.4f}\n")
            f.write(f"  F1-Score:  {exp['test_f1']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ Results saved to:")
    print(f"  - {save_path_root}")
    print(f"  - {save_path_log}")
    
    return results


def quick_compare(
    dataset_name: str = "Twitter15",
    sample_ratio: float = 0.05
):
    """
    Quick comparison: Baseline vs TAPE (concat fusion only)
    
    Args:
        dataset_name: Dataset name
        sample_ratio: Sampling ratio
    """
    print("="*70)
    print("Quick Comparison: Baseline vs TAPE")
    print("="*70)
    
    # Baseline
    print("\n1. Running Baseline...")
    pipeline_baseline = TAPEPipeline(enable_augmentation=False)
    baseline_results = pipeline_baseline.run(dataset_name, sample_ratio)
    
    # TAPE
    print("\n2. Running TAPE (concat fusion)...")
    pipeline_tape = TAPEPipeline(
        enable_augmentation=True,
        fusion_strategy="concat",
        node_selection_strategy="hybrid"
    )
    tape_results = pipeline_tape.run(dataset_name, sample_ratio)
    
    # Compare
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    baseline_acc = baseline_results['test_results']['accuracy']
    baseline_f1 = baseline_results['test_results']['f1']
    tape_acc = tape_results['test_results']['accuracy']
    tape_f1 = tape_results['test_results']['f1']
    
    print(f"\nBaseline:")
    print(f"  Accuracy: {baseline_acc:.4f}")
    print(f"  F1-Score: {baseline_f1:.4f}")
    
    print(f"\nTAPE:")
    print(f"  Accuracy: {tape_acc:.4f} ({(tape_acc-baseline_acc)*100:+.2f}%)")
    print(f"  F1-Score: {tape_f1:.4f} ({(tape_f1-baseline_f1)*100:+.2f}%)")
    
    print(f"\nImprovement:")
    print(f"  Accuracy: {(tape_acc-baseline_acc)*100:+.2f}%")
    print(f"  F1-Score: {(tape_f1-baseline_f1)*100:+.2f}%")
    
    if tape_f1 > baseline_f1:
        print(f"\n✅ TAPE outperforms baseline!")
    else:
        print(f"\n⚠️  TAPE does not improve over baseline in this run.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Compare TAPE vs Baseline performance"
    )
    
    parser.add_argument('--dataset', type=str, default='Twitter15',
                       choices=['Twitter15', 'Twitter16', 'Weibo'],
                       help='Dataset name')
    parser.add_argument('--sample_ratio', type=float, default=0.05,
                       help='Data sampling ratio')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'full'],
                       help='Comparison mode (quick or full)')
    parser.add_argument('--fusion_strategies', nargs='+',
                       default=['concat', 'weighted'],
                       help='Fusion strategies to test')
    parser.add_argument('--node_strategies', nargs='+',
                       default=['hybrid'],
                       help='Node selection strategies to test')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_compare(args.dataset, args.sample_ratio)
    else:
        run_comparison(
            dataset_name=args.dataset,
            sample_ratio=args.sample_ratio,
            fusion_strategies=args.fusion_strategies,
            node_strategies=args.node_strategies
        )


if __name__ == '__main__':
    main()

