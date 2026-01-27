import torch
import os
import json


def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_results(results, filepath):
    """Save results dictionary to file."""
    ensure_dir(os.path.dirname(filepath))
    torch.save(results, filepath)


def load_results(filepath):
    """Load results dictionary from file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Results file not found: {filepath}")
    
    results = torch.load(filepath)
    return results


def get_device(device_str='auto'):
    """Get compute device (MPS/CUDA/CPU)."""
    if device_str == 'auto':
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)
    return device


def validate_results(all_results, config):
    """Validate that all experiments completed successfully."""
    messages = []
    is_valid = True
    if 'exp1' in all_results:
        exp1 = all_results['exp1']
        expected_trials = config['num_trials_random']
        for sparsity in config['one_shot_sparsity']:
            if sparsity not in exp1:
                messages.append(f"[ERROR] Exp1: Missing sparsity {sparsity}")
                is_valid = False
            elif len(exp1[sparsity]) != expected_trials:
                messages.append(f"[ERROR] Exp1: Expected {expected_trials} trials at {sparsity}, got {len(exp1[sparsity])}")
                is_valid = False
        if is_valid:
            messages.append("[OK] Experiment 1: Complete")
    else:
        messages.append("[ERROR] Experiment 1: Missing")
        is_valid = False
    if 'exp2' in all_results:
        exp2 = all_results['exp2']
        expected_trials = config['num_trials_winning']
        for sparsity in config['one_shot_sparsity']:
            if sparsity not in exp2:
                messages.append(f"[ERROR] Exp2: Missing sparsity {sparsity}")
                is_valid = False
            elif len(exp2[sparsity]) != expected_trials:
                messages.append(f"[ERROR] Exp2: Expected {expected_trials} trials, got {len(exp2[sparsity])}")
                is_valid = False
            elif 'mask' not in exp2[sparsity][0]:
                messages.append(f"[ERROR] Exp2: Missing mask at sparsity {sparsity}")
                is_valid = False
        if is_valid:
            messages.append("[OK] Experiment 2: Complete")
    else:
        messages.append("[ERROR] Experiment 2: Missing")
        is_valid = False
    if 'exp3' in all_results:
        exp3 = all_results['exp3']
        expected_trials = config['num_trials_winning']
        for sparsity in config['iterative_sparsity']:
            if sparsity not in exp3:
                messages.append(f"[ERROR] Exp3: Missing sparsity {sparsity}")
                is_valid = False
            elif len(exp3[sparsity]) != expected_trials:
                messages.append(f"[ERROR] Exp3: Expected {expected_trials} trials, got {len(exp3[sparsity])}")
                is_valid = False
            elif 'learning_curve' not in exp3[sparsity][0]:
                messages.append(f"[ERROR] Exp3: Missing learning curve at {sparsity}")
                is_valid = False
        if is_valid:
            messages.append("[OK] Experiment 3: Complete")
    else:
        messages.append("[ERROR] Experiment 3: Missing")
        is_valid = False
    
    return is_valid, messages