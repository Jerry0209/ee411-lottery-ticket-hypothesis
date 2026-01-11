import torch
import torch.nn as nn
import copy
from collections import defaultdict
from ..models import create_model
from ..training import train_with_early_stop_tracking, create_optimizer
from ..pruning import create_random_mask, apply_mask
from ..utils import print_experiment_header, print_results_summary, save_results


def run_experiment1_random_sparse(config, train_loader, val_loader, device):
    """
    Experiment 1: Random Sparse Networks
    
    Baseline experiment testing randomly pruned networks at different sparsity levels.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: RANDOM SPARSE NETWORKS")
    print("="*70)
    
    exp_config = config['exp1']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()
    
    for sparsity in exp_config['sparsity_levels']:
        for trial in range(1, exp_config['num_trials'] + 1):
            print_experiment_header("Random Sparse", sparsity, trial, exp_config['num_trials'])
            
            # Create model
            model = create_model(config['model'], device=device)
            
            # Create random mask
            mask_dict = create_random_mask(model, sparsity)
            apply_mask(model, mask_dict)
            
            # Setup optimizer
            optimizer = create_optimizer(model, config)
            
            # Train
            results = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                mask_dict, config['eval_every'], config['verbose']
            )
            
            print_results_summary(results)
            all_results[sparsity].append(results)
    
    # Save results
    save_path = f"{config['results_dir']}/exp1_random_sparse.pt"
    save_results(dict(all_results), save_path)
    
    return dict(all_results)