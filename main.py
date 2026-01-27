import sys
import os
from datetime import datetime

from configs.config import config
from src.data import load_dataset
from src.models import SUPPORTED_MODELS
from src.utils import get_device, validate_results
from src.experiments import (
    run_experiment1_random_sparse,
    run_experiment2_oneshot_winning,
    run_experiment3_iterative_winning,
    run_experiment4_oneshot_reinit,
    run_experiment5_iterative_reinit
)
from src.figures import generate_all_figures


def setup_results_dirs(config):
    """Create timestamped directories for storing experiment results."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_name = f"{config['model']}_{config['dataset']}"

    base_dir = f"./results/{base_name}/{timestamp}"
    metrics_dir = f"{base_dir}/metrics"
    figures_dir = f"{base_dir}/figures"

    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    config['base_dir'] = base_dir
    config['results_dir'] = metrics_dir
    config['figures_dir'] = figures_dir
    config['log_file'] = f"{base_dir}/experiment.log"

    return base_dir, metrics_dir, figures_dir


def validate_model(model_name):
    """Validate that the requested model is supported."""
    if model_name not in SUPPORTED_MODELS:
        print(f"Error: Model '{model_name}' not supported.")
        print(f"Supported models: {SUPPORTED_MODELS}")
        sys.exit(1)


def main():
    """Run all configured lottery ticket experiments and generate figures."""
    validate_model(config['model'])
    base_dir, metrics_dir, figures_dir = setup_results_dirs(config)
    device = get_device(config['device'])
    train_loader, val_loader, test_loader = load_dataset(config)
    all_results = {}

    if 'exp1' in config:
        all_results['exp1'] = run_experiment1_random_sparse(
            config, train_loader, val_loader, test_loader, device
        )

    if 'exp2' in config:
        all_results['exp2'] = run_experiment2_oneshot_winning(
            config, train_loader, val_loader, test_loader, device
        )

    if 'exp3' in config:
        all_results['exp3'] = run_experiment3_iterative_winning(
            config, train_loader, val_loader, test_loader, device
        )

    if 'exp4' in config:
        all_results['exp4'] = run_experiment4_oneshot_reinit(
            config, all_results['exp2'], train_loader, val_loader, test_loader, device
        )

    if 'exp5' in config:
        all_results['exp5'] = run_experiment5_iterative_reinit(
            config, all_results['exp3'], train_loader, val_loader, test_loader, device
        )

    is_valid, messages = validate_results(all_results, config)
    for msg in messages:
        print(msg)
    if is_valid:
        generate_all_figures(config)
        print(f"Done. Outputs: {config['base_dir']}/")
    else:
        print("Some experiments incomplete. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
