import sys
import subprocess
import json
from datetime import datetime
from pathlib import Path

EXPERIMENTS = [
    {
        'name': 'GAT_Baseline',
        'args': {
            'gnn_backbone': 'gat',
            'enable_augmentation': False,
        }
    },
    {
        'name': 'GCN_Baseline',
        'args': {
            'gnn_backbone': 'gcn',
            'enable_augmentation': False,
        }
    },
    {
        'name': 'SeAug_GAT',
        'args': {
            'gnn_backbone': 'gat',
            'enable_augmentation': True,
        }
    },
    {
        'name': 'SeAug_GCN',
        'args': {
            'gnn_backbone': 'gcn',
            'enable_augmentation': True,
        }
    }
]


def build_command(experiment, dataset='Twitter15', **kwargs):
    cmd = [sys.executable, 'seaug_pipeline.py', '--dataset', dataset]
    
    if experiment['args'].get('enable_augmentation'):
        cmd.append('--enable_augmentation')
    
    cmd.extend(['--gnn_backbone', experiment['args']['gnn_backbone']])
    
    for key, value in kwargs.items():
        if isinstance(value, bool) and value:
            cmd.append(f'--{key}')
        elif not isinstance(value, bool):
            cmd.extend([f'--{key}', str(value)])
    
    return cmd


def run_experiment(experiment, dataset='Twitter15', **kwargs):
    print(f"\nRunning: {experiment['name']}")
    cmd = build_command(experiment, dataset, **kwargs)
    
    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"{experiment['name']} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{experiment['name']} failed (exit code {e.returncode})")
        return False
    except KeyboardInterrupt:
        print(f"{experiment['name']} interrupted")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FYP experiments')
    
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
    
    experiments_to_run = EXPERIMENTS.copy()
    
    if args.only:
        experiments_to_run = [e for e in experiments_to_run if e['name'] in args.only]
    
    if args.skip:
        experiments_to_run = [e for e in experiments_to_run if e['name'] not in args.skip]
    
    print(f"\nDataset: {args.dataset}")
    print(f"Experiments: {len(experiments_to_run)}")
    for exp in experiments_to_run:
        print(f"  - {exp['name']}")
    
    results = {}
    start_time = datetime.now()
    
    for i, experiment in enumerate(experiments_to_run, 1):
        print(f"\n[{i}/{len(experiments_to_run)}] {experiment['name']}")
        
        success = run_experiment(
            experiment,
            dataset=args.dataset,
            augmentation_ratio=args.augmentation_ratio,
            node_strategy=args.node_strategy,
            fusion_strategy=args.fusion_strategy,
            batch_size=args.batch_size,
            sample_ratio=args.sample_ratio
        )
        
        results[experiment['name']] = {'success': success}
    
    duration = datetime.now() - start_time
    
    print(f"\nSummary - Time: {duration}")
    for name, result in results.items():
        status = "OK" if result['success'] else "FAILED"
        print(f"  {name}: {status}")
    
    summary_path = Path('experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': start_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'dataset': args.dataset,
            'experiments': results
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()

