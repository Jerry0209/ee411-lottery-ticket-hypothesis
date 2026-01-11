"""
TEMPORARY script to resume experiments from Exp3.
Loads existing Exp1 & Exp2 results, runs Exp3, Exp4, Exp5.

Usage: python scripts/run_experiments_temp.py --config full
"""

import sys
import os
import argparse
import torch

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data import load_mnist
from src.utils import get_device, create_summary_report, validate_results

from src.experiments.all_experiments import (
    run_experiment3_iterative_winning,
    run_experiment4_oneshot_reinit,
    run_experiment5_iterative_reinit
)


def main():
    parser = argparse.ArgumentParser(description='Resume lottery ticket experiments from Exp3')
    parser.add_argument('--config', type=str, default='full',
                        choices=['minimal', 'test', 'full'],
                        help='Configuration to use')
    args = parser.parse_args()

    # Load config
    if args.config == 'minimal':
        from configs.config_minimal import config_minimal as config
    elif args.config == 'test':
        from configs.config_test import config_test as config
    else:
        from configs.config_full import config_full as config

    print("="*70)
    print("LOTTERY TICKET HYPOTHESIS - RESUME FROM EXP3")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Max iterations: {config['max_iterations']}")
    print("="*70)

    # Load existing results
    exp1_path = f"{config['results_dir']}/exp1_random_sparse.pt"
    exp2_path = f"{config['results_dir']}/exp2_oneshot_winning.pt"

    if not os.path.exists(exp1_path):
        print(f"ERROR: Exp1 results not found at {exp1_path}")
        sys.exit(1)
    if not os.path.exists(exp2_path):
        print(f"ERROR: Exp2 results not found at {exp2_path}")
        sys.exit(1)

    print(f"\nLoading existing results:")
    print(f"  - Exp1: {exp1_path}")
    print(f"  - Exp2: {exp2_path}")

    exp1_results = torch.load(exp1_path, weights_only=False)
    exp2_results = torch.load(exp2_path, weights_only=False)

    print(f"  - Exp1 loaded: {len(exp1_results)} sparsity levels")
    print(f"  - Exp2 loaded: {len(exp2_results)} sparsity levels")

    # Setup
    device = get_device(config['device'])
    train_loader, val_loader, test_loader = load_mnist(
        config['data_path'], config['train_size'],
        config['val_size'], config['batch_size']
    )

    # Run remaining experiments
    all_results = {
        'exp1': exp1_results,
        'exp2': exp2_results
    }

    print("\n" + "="*70)
    print("STARTING EXP3 (Iterative Winning Tickets)")
    print("="*70)

    # Exp 3: Iterative winning
    all_results['exp3'] = run_experiment3_iterative_winning(
        config, train_loader, val_loader, test_loader, device
    )

    print("\n" + "="*70)
    print("STARTING EXP4 (One-Shot Reinit)")
    print("="*70)

    # Exp 4: One-shot reinit (uses exp2 masks)
    all_results['exp4'] = run_experiment4_oneshot_reinit(
        config, all_results['exp2'], train_loader, val_loader, test_loader, device
    )

    print("\n" + "="*70)
    print("STARTING EXP5 (Iterative Reinit)")
    print("="*70)

    # Exp 5: Iterative reinit (uses exp3 masks)
    all_results['exp5'] = run_experiment5_iterative_reinit(
        config, all_results['exp3'], train_loader, val_loader, test_loader, device
    )

    # Validate and summarize
    is_valid, messages = validate_results(all_results, config)
    for msg in messages:
        print(msg)

    if is_valid:
        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        create_summary_report(all_results, config, f"{config['results_dir']}/summary.txt")
        print(f"\nResults saved to: {config['results_dir']}/")
        print(f"\nNext: Generate figures with:")
        print(f"  python scripts/generate_figures.py --results {config['results_dir']}/")
    else:
        print("\n" + "="*70)
        print("SOME EXPERIMENTS INCOMPLETE")
        print("="*70)
        sys.exit(1)


if __name__ == '__main__':
    main()
