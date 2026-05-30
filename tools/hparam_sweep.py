# coding:utf-8
"""
Hyperparameter sensitivity sweep runner.

Runs short training sessions (30 epochs) across hyperparameter values
to produce sensitivity curves for thesis experimental chapter.

Usage:
  python tools/hparam_sweep.py --base_data data/train.json --val_data data/val.json \
      --data_dir data/my_sequences --use_visual --output models/hparam_sweep
"""

import subprocess
import sys, os, json, argparse
from itertools import product


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter sensitivity sweep')
    parser.add_argument('--base_data', required=True, help='Training annotation JSON')
    parser.add_argument('--val_data', required=True, help='Validation annotation JSON')
    parser.add_argument('--data_dir', default='data/my_sequences')
    parser.add_argument('--use_visual', action='store_true')
    parser.add_argument('--output', default='models/hparam_sweep')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry_run', action='store_true', help='Print commands only')
    return parser.parse_args()


SWEEPS = {
    'learning_rate': {
        'values': [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        'flag': '--lr',
        'xlabel': 'Learning Rate',
    },
    'dropout': {
        'values': [0.05, 0.1, 0.15, 0.2, 0.3],
        'flag': '--dropout',
        'xlabel': 'Dropout Rate',
    },
    'tcn_kernel': {
        'values': [3, 5, 7, 9],
        'flag': '--tcn_kernel',
        'xlabel': 'TCN Kernel Size',
    },
}

SWEEP_EPOCHS = 30


def run_sweep(args):
    os.makedirs(args.output, exist_ok=True)
    results = {}

    for sweep_name, config in SWEEPS.items():
        print(f'\n{"="*60}')
        print(f'  SWEEP: {config["xlabel"]}')
        print(f'{"="*60}')

        sweep_results = []
        for val in config['values']:
            run_name = f'{sweep_name}_{str(val).replace(".", "p")}'
            run_dir = os.path.join(args.output, run_name)
            os.makedirs(run_dir, exist_ok=True)

            cmd = [
                sys.executable, 'tools/train.py',
                '--data', args.base_data,
                '--val_data', args.val_data,
                '--data_dir', args.data_dir,
                '--epochs', str(SWEEP_EPOCHS),
                '--batch_size', '16',
                '--output', run_dir,
                '--no_amp',
                '--augment',
                '--class_weight',
                config['flag'], str(val),
            ]
            if args.use_visual:
                cmd.append('--use_visual')

            print(f'\n  [{run_name}]')
            print(f'  CMD: {" ".join(cmd)}')

            if args.dry_run:
                sweep_results.append({'value': val, 'status': 'dry_run'})
                continue

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode == 0:
                    # Read best val_acc from training_metrics.json
                    metrics_path = os.path.join(run_dir, 'training_metrics.json')
                    if os.path.exists(metrics_path):
                        with open(metrics_path) as f:
                            metrics = json.load(f)
                        best = max(metrics, key=lambda m: m.get('val_acc', 0))
                        val_acc = best.get('val_acc', 0)
                        train_loss = best.get('train_loss', 0)
                        val_loss = best.get('val_loss', 0)
                        print(f'  BEST: val_acc={val_acc:.4f}, val_loss={val_loss:.4f}')
                        sweep_results.append({
                            'value': val,
                            'val_acc': val_acc,
                            'val_loss': val_loss,
                            'train_loss': train_loss,
                            'best_epoch': best.get('epoch', 0),
                        })
                    else:
                        print(f'  WARN: no metrics found')
                        sweep_results.append({'value': val, 'status': 'no_metrics'})
                else:
                    print(f'  FAILED (rc={result.returncode})')
                    print(f'  STDERR: {result.stderr[-200:]}')
                    sweep_results.append({'value': val, 'status': 'failed'})
            except subprocess.TimeoutExpired:
                print(f'  TIMEOUT')
                sweep_results.append({'value': val, 'status': 'timeout'})

        results[sweep_name] = {
            'xlabel': config['xlabel'],
            'results': sweep_results,
        }

        # Save incrementally
        output_path = os.path.join(args.output, 'sweep_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary
    print(f'\n{"="*60}')
    print(f'  SWEEP COMPLETE')
    print(f'{"="*60}')
    for sweep_name, data in results.items():
        print(f'\n  {data["xlabel"]}:')
        for r in data['results']:
            if 'val_acc' in r:
                print(f'    {r["value"]:>8}: val_acc={r["val_acc"]:.4f}')
            else:
                print(f'    {r["value"]:>8}: {r.get("status", "?")}')

    print(f'\nResults: {output_path}')


if __name__ == '__main__':
    run_sweep(parse_args())
