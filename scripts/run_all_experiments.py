"""
Main execution script for all experiments.
Usage: python scripts/run_all_experiments.py --config minimal
"""

import sys
import os
import argparse

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Now imports will work
from src.data import load_mnist
from src.utils import get_device, create_summary_report, validate_results

# Import all experiment functions from consolidated file
from src.experiments.all_experiments import (
    run_experiment1_random_sparse,
    run_experiment2_oneshot_winning,
    run_experiment3_iterative_winning,
    run_experiment4_oneshot_reinit,
    run_experiment5_iterative_reinit
)


def main():
    parser = argparse.ArgumentParser(description='Run lottery ticket experiments')
    parser.add_argument('--config', type=str, default='minimal',
                        choices=['minimal', 'test', 'full', 'paper_same'],
                        help='Configuration to use')
    args = parser.parse_args()

    # Load config
    if args.config == 'minimal':
        from configs.config_minimal import config_minimal as config
    elif args.config == 'test':
        from configs.config_test import config_test as config
    elif args.config == 'paper_same':
        from configs.config_paper_same import config_paper_same as config
    else:
        from configs.config_full import config_full as config
    
    print("="*70)
    print("LOTTERY TICKET HYPOTHESIS - REPRODUCTION")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Max iterations: {config['max_iterations']}")
    print("="*70)
    
    # Setup
    device = get_device(config['device'])
    train_loader, val_loader, test_loader = load_mnist(
        config['data_path'], config['train_size'],
        config['val_size'], config['batch_size']
    )
    
    # Run all experiments
    all_results = {}

    print("\nStarting experiments...")

    # Exp 1: Random sparse
    all_results['exp1'] = run_experiment1_random_sparse(config, train_loader, val_loader, test_loader, device)

    # Exp 2: One-shot winning
    all_results['exp2'] = run_experiment2_oneshot_winning(config, train_loader, val_loader, test_loader, device)

    # Exp 3: Iterative winning
    all_results['exp3'] = run_experiment3_iterative_winning(config, train_loader, val_loader, test_loader, device)

    # Exp 4: One-shot reinit
    all_results['exp4'] = run_experiment4_oneshot_reinit(config, all_results['exp2'], train_loader, val_loader, test_loader, device)

    # Exp 5: Iterative reinit
    all_results['exp5'] = run_experiment5_iterative_reinit(config, all_results['exp3'], train_loader, val_loader, test_loader, device)
    
    # Validate and summarize
    is_valid, messages = validate_results(all_results, config)
    for msg in messages:
        print(msg)
    
    if is_valid:
        print("\n✓ All experiments completed successfully!")
        create_summary_report(all_results, config, f"{config['results_dir']}/summary.txt")
        print(f"\nResults saved to: {config['results_dir']}/")
        print(f"\nNext: Generate figures with:")
        print(f"  python scripts/generate_figures.py --results {config['results_dir']}/")
    else:
        print("\n❌ Some experiments incomplete. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()