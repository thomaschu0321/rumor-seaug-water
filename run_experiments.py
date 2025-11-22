"""
Experiment Runner for FYP - Run 4 Different Configurations

This script runs 4 experiments:
1. GAT baseline (no augmentation)
2. GCN baseline (no augmentation)
3. SeAug + GAT (with selective augmentation)
4. SeAug + GCN (with selective augmentation)
"""

import os
import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Experiment configurations
EXPERIMENTS = [
    {
        'name': 'GAT_Baseline',
        'description': 'GAT baseline without augmentation',
        'args': {
            'gnn_backbone': 'gat',
            'enable_augmentation': False,
        }
    },
    {
        'name': 'GCN_Baseline',
        'description': 'GCN baseline without augmentation',
        'args': {
            'gnn_backbone': 'gcn',
            'enable_augmentation': False,
        }
    },
    {
        'name': 'SeAug_GAT',
        'description': 'Selective augmentation with GAT',
        'args': {
            'gnn_backbone': 'gat',
            'enable_augmentation': True,
        }
    },
    {
        'name': 'SeAug_GCN',
        'description': 'Selective augmentation with GCN',
        'args': {
            'gnn_backbone': 'gcn',
            'enable_augmentation': True,
        }
    }
]


def build_command(experiment, dataset='Twitter15', **kwargs):
    """Build command for running an experiment"""
    cmd = [sys.executable, 'seaug_pipeline.py', '--dataset', dataset]
    
    # Add experiment-specific arguments
    if experiment['args'].get('enable_augmentation'):
        cmd.append('--enable_augmentation')
    
    cmd.extend(['--gnn_backbone', experiment['args']['gnn_backbone']])
    
    # Add any additional kwargs
    for key, value in kwargs.items():
        if isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        elif not isinstance(value, bool):
            cmd.extend([f'--{key}', str(value)])
    
    return cmd


def run_experiment(experiment, dataset='Twitter15', **kwargs):
    """Run a single experiment"""
    print("\n" + "="*80)
    print(f"Running Experiment: {experiment['name']}")
    print(f"Description: {experiment['description']}")
    print("="*80)
    
    cmd = build_command(experiment, dataset, **kwargs)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    # Run the experiment
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✓ Experiment '{experiment['name']}' completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Experiment '{experiment['name']}' failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ Experiment '{experiment['name']}' interrupted by user")
        return False


def main():
    """Main function to run all experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run all 4 FYP experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments on Twitter15
  python run_experiments.py
  
  # Run only GAT and GCN baselines (no augmentation)
  python run_experiments.py --only GAT_Baseline GCN_Baseline
  
  # Run baselines with sample_ratio 0.1
  python run_experiments.py --only GAT_Baseline GCN_Baseline --sample_ratio 0.1
  
  # Run on a different dataset
  python run_experiments.py --dataset Twitter16
  
  # Run with custom augmentation ratio
  python run_experiments.py --augmentation_ratio 0.2
        """
    )
    
    parser.add_argument('--dataset', type=str, default='Twitter15',
                       choices=['Twitter15', 'Twitter16', 'Weibo'],
                       help='Dataset to use (default: Twitter15)')
    parser.add_argument('--augmentation_ratio', type=float, default=0.3,
                       help='Ratio of nodes to augment (default: 0.3)')
    parser.add_argument('--node_strategy', type=str, default='hybrid',
                       choices=['uncertainty', 'importance', 'hybrid'],
                       help='Node selection strategy (default: hybrid)')
    parser.add_argument('--fusion_strategy', type=str, default='concat',
                       choices=['concat', 'weighted', 'gated', 'attention'],
                       help='Feature fusion strategy (default: concat)')
    parser.add_argument('--batch_size', type=int, default=20,
                       help='Batch size for LLM API calls (default: 20)')
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                       help='Data sampling ratio (default: 1.0)')
    parser.add_argument('--skip', type=str, nargs='+',
                       choices=['GAT_Baseline', 'GCN_Baseline', 'SeAug_GAT', 'SeAug_GCN'],
                       help='Skip specific experiments')
    parser.add_argument('--only', type=str, nargs='+',
                       choices=['GAT_Baseline', 'GCN_Baseline', 'SeAug_GAT', 'SeAug_GCN'],
                       help='Run only specific experiments')
    
    args = parser.parse_args()
    
    # Filter experiments
    experiments_to_run = EXPERIMENTS.copy()
    
    if args.only:
        experiments_to_run = [e for e in experiments_to_run if e['name'] in args.only]
    
    if args.skip:
        experiments_to_run = [e for e in experiments_to_run if e['name'] not in args.skip]
    
    # Print summary
    print("\n" + "="*80)
    print("FYP Experiment Runner")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Experiments to run: {len(experiments_to_run)}")
    for exp in experiments_to_run:
        print(f"  - {exp['name']}: {exp['description']}")
    print("\n" + "="*80)
    
    # Confirm before running
    if len(experiments_to_run) > 0:
        response = input("\nProceed with experiments? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return
    
    # Run experiments
    results = {}
    start_time = datetime.now()
    
    for i, experiment in enumerate(experiments_to_run, 1):
        print(f"\n\n{'='*80}")
        print(f"Experiment {i}/{len(experiments_to_run)}: {experiment['name']}")
        print(f"{'='*80}")
        
        success = run_experiment(
            experiment,
            dataset=args.dataset,
            augmentation_ratio=args.augmentation_ratio,
            node_strategy=args.node_strategy,
            fusion_strategy=args.fusion_strategy,
            batch_size=args.batch_size,
            sample_ratio=args.sample_ratio
        )
        
        results[experiment['name']] = {
            'success': success,
            'description': experiment['description']
        }
        
        if not success:
            print(f"\n⚠ Warning: Experiment '{experiment['name']}' failed!")
            # Check if running in interactive mode
            if sys.stdin.isatty():
                try:
                    response = input("Continue with remaining experiments? (y/n): ").strip().lower()
                    if response != 'y':
                        print("Stopping experiments.")
                        break
                except (EOFError, KeyboardInterrupt):
                    print("\nNon-interactive mode detected. Continuing with remaining experiments...")
            else:
                # Non-interactive mode: continue automatically
                print("Non-interactive mode detected. Continuing with remaining experiments...")
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n\n" + "="*80)
    print("Experiment Summary")
    print("="*80)
    print(f"\nTotal time: {duration}")
    print(f"\nResults:")
    for name, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"  {name:20s}: {status}")
    
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nSuccessful: {successful}/{len(results)}")
    print("="*80)
    
    # Save results summary
    summary_path = Path('experiment_summary.json')
    summary = {
        'timestamp': start_time.isoformat(),
        'duration_seconds': duration.total_seconds(),
        'dataset': args.dataset,
        'experiments': results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_path}")


if __name__ == '__main__':
    main()

