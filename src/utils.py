"""
Utility functions for saving, loading, and managing results.
"""

import torch
import os
import json
from datetime import datetime


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_results(results, filepath):
    """
    Save experiment results to file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    torch.save(results, filepath)
    print(f"✓ Saved results to {filepath}")


def load_results(filepath):
    """
    Load experiment results from file.
    
    Args:
        filepath: Path to results file
    
    Returns:
        results: Dictionary of results
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    results = torch.load(filepath)
    print(f"✓ Loaded results from {filepath}")
    return results


def get_device(device_str='auto'):
    """
    Get PyTorch device.
    
    Args:
        device_str: 'auto', 'cuda', 'mps', or 'cpu'
    
    Returns:
        device: PyTorch device
    """
    if device_str == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    
    print(f"Using device: {device}")
    return device


def print_experiment_header(exp_name, sparsity, trial, total_trials):
    """Print formatted experiment header."""
    print(f"\n{'='*70}")
    print(f"{exp_name}")
    print(f"Sparsity: {sparsity*100:.1f}% | Trial: {trial}/{total_trials}")
    print(f"{'='*70}")


def print_results_summary(results):
    """Print summary of experiment results."""
    print(f"\n{'─'*70}")
    print(f"Early stop: iteration {results['early_stop_iter']}")
    print(f"Val loss at early stop: {results['early_stop_val_loss']:.4f}")
    print(f"Test accuracy at early stop: {results['early_stop_test_acc']:.2f}%")
    print(f"{'─'*70}")


def create_summary_report(all_results, config, output_path):
    """
    Create a text summary of all experiments.
    
    Args:
        all_results: Dictionary containing all experiment results
        config: Configuration used
        output_path: Path to save summary
    """
    ensure_dir(os.path.dirname(output_path))
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("LOTTERY TICKET HYPOTHESIS - EXPERIMENT SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration: {config.get('name', 'Unknown')}\n")
        f.write(f"Max iterations: {config['max_iterations']}\n")
        f.write(f"Device: {config['device']}\n")
        f.write("\n")
        
        # Experiment 1: Random Sparse
        if 'exp1' in all_results:
            f.write("-"*70 + "\n")
            f.write("EXPERIMENT 1: Random Sparse Networks\n")
            f.write("-"*70 + "\n")
            for sparsity in sorted(all_results['exp1'].keys(), reverse=True):
                trials = all_results['exp1'][sparsity]
                avg_early_stop = sum(t['early_stop_iter'] for t in trials) / len(trials)
                avg_acc = sum(t['early_stop_test_acc'] for t in trials) / len(trials)
                f.write(f"  {sparsity*100:>5.1f}%: {avg_early_stop:>7.0f} iter, {avg_acc:>5.2f}% acc\n")
            f.write("\n")
        
        # Similar for other experiments...
        f.write("="*70 + "\n")
        f.write("Summary generation complete.\n")
    
    print(f"✓ Summary saved to {output_path}")


def validate_results(all_results, config):
    """
    Validate that all experiments completed successfully.
    
    Args:
        all_results: Dictionary of all results
        config: Configuration used
    
    Returns:
        is_valid: Boolean indicating if all experiments are complete
        messages: List of validation messages
    """
    messages = []
    is_valid = True
    
    # Check Experiment 1
    if 'exp1' in all_results:
        exp1 = all_results['exp1']
        expected_trials = config['num_trials_random']
        for sparsity in config['one_shot_sparsity']:
            if sparsity not in exp1:
                messages.append(f"❌ Exp1: Missing sparsity {sparsity}")
                is_valid = False
            elif len(exp1[sparsity]) != expected_trials:
                messages.append(f"❌ Exp1: Expected {expected_trials} trials at {sparsity}, got {len(exp1[sparsity])}")
                is_valid = False
        if is_valid:
            messages.append("✓ Experiment 1: Complete")
    else:
        messages.append("❌ Experiment 1: Missing")
        is_valid = False
    
    # Check Experiment 2
    if 'exp2' in all_results:
        exp2 = all_results['exp2']
        expected_trials = config['num_trials_winning']
        for sparsity in config['one_shot_sparsity']:
            if sparsity not in exp2:
                messages.append(f"❌ Exp2: Missing sparsity {sparsity}")
                is_valid = False
            elif len(exp2[sparsity]) != expected_trials:
                messages.append(f"❌ Exp2: Expected {expected_trials} trials, got {len(exp2[sparsity])}")
                is_valid = False
            elif 'mask' not in exp2[sparsity][0]:
                messages.append(f"❌ Exp2: Missing mask at sparsity {sparsity}")
                is_valid = False
        if is_valid:
            messages.append("✓ Experiment 2: Complete")
    else:
        messages.append("❌ Experiment 2: Missing")
        is_valid = False
    
    # Check Experiment 3
    if 'exp3' in all_results:
        exp3 = all_results['exp3']
        expected_trials = config['num_trials_winning']
        for sparsity in config['iterative_sparsity']:
            if sparsity not in exp3:
                messages.append(f"❌ Exp3: Missing sparsity {sparsity}")
                is_valid = False
            elif len(exp3[sparsity]) != expected_trials:
                messages.append(f"❌ Exp3: Expected {expected_trials} trials, got {len(exp3[sparsity])}")
                is_valid = False
            elif 'learning_curve' not in exp3[sparsity][0]:
                messages.append(f"❌ Exp3: Missing learning curve at {sparsity}")
                is_valid = False
        if is_valid:
            messages.append("✓ Experiment 3: Complete")
    else:
        messages.append("❌ Experiment 3: Missing")
        is_valid = False
    
    return is_valid, messages