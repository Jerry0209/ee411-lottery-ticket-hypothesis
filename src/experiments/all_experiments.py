"""
All lottery ticket experiments in one file.
"""

import torch
import torch.nn as nn
import copy
from collections import defaultdict
import sys
import os

# Ensure imports work
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models import create_model
from src.training import train_with_early_stop_tracking, create_optimizer
from src.pruning import (create_random_mask, apply_mask, 
                        create_magnitude_mask_layerwise, get_sparsity)
from src.utils import print_experiment_header, print_results_summary, save_results


def run_experiment1_random_sparse(config, train_loader, val_loader, test_loader, device):
    """
    Experiment 1: Random Sparse Networks
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
                mask_dict, config['eval_every'], config['verbose'],
                test_loader=test_loader
            )

            print_results_summary(results)
            all_results[sparsity].append(results)

    # Save results
    save_path = f"{config['results_dir']}/exp1_random_sparse.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)


def run_experiment2_oneshot_winning(config, train_loader, val_loader, test_loader, device):
    """
    Experiment 2: One-Shot Winning Tickets
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: ONE-SHOT WINNING TICKETS")
    print("="*70)

    exp_config = config['exp2']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for sparsity in exp_config['sparsity_levels']:
        for trial in range(1, exp_config['num_trials'] + 1):
            print_experiment_header("One-Shot Winning Ticket", sparsity, trial, exp_config['num_trials'])

            # Step 1: Create model and save initial weights
            print("[1] Creating model and saving initial weights...")
            model = create_model(config['model'], device=device)
            initial_state = copy.deepcopy(model.state_dict())

            # Step 2: Train full network
            print("[2] Training full network...")
            optimizer = create_optimizer(model, config)
            _ = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                None, config['eval_every'], config['verbose'],
                test_loader=test_loader
            )

            # Step 3: Create magnitude mask
            print(f"[3] Creating magnitude mask at {sparsity*100:.1f}%...")
            mask_dict = create_magnitude_mask_layerwise(
                model, sparsity, None,
                config['layer_specific_rates'], config['prune_rate']
            )

            # Step 4: Reset to initial weights
            print("[4] Resetting to initial weights and training winning ticket...")
            model.load_state_dict(initial_state)
            apply_mask(model, mask_dict)

            # Step 5: Train winning ticket
            optimizer = create_optimizer(model, config)
            results = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                mask_dict, config['eval_every'], config['verbose'],
                test_loader=test_loader
            )

            results['mask'] = mask_dict
            print_results_summary(results)
            all_results[sparsity].append(results)

    save_path = f"{config['results_dir']}/exp2_oneshot_winning.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)


def run_experiment3_iterative_winning(config, train_loader, val_loader, test_loader, device):
    """
    Experiment 3: Iterative Winning Tickets
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: ITERATIVE WINNING TICKETS")
    print("="*70)

    exp_config = config['exp3']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for target_sparsity in exp_config['sparsity_levels']:
        for trial in range(1, exp_config['num_trials'] + 1):
            print_experiment_header("Iterative Winning Ticket", target_sparsity, trial, exp_config['num_trials'])

            # Step 1: Create model and save initial weights
            print("[1] Saving initial weights...")
            model = create_model(config['model'], device=device)
            initial_state = copy.deepcopy(model.state_dict())

            # Step 2: Iterative pruning loop
            current_sparsity = 1.0
            cumulative_mask = None
            round_num = 0

            while current_sparsity > target_sparsity + 0.001:
                round_num += 1
                print(f"\n--- Round {round_num} (current: {current_sparsity*100:.2f}%) ---")

                # Train
                print("[TRAIN] Training...")
                optimizer = create_optimizer(model, config)
                _ = train_with_early_stop_tracking(
                    model, device, train_loader, val_loader,
                    optimizer, criterion, config['max_iterations'],
                    cumulative_mask, config['eval_every'], config['verbose'],
                    test_loader=test_loader
                )

                # Prune
                next_sparsity = current_sparsity * (1 - config['prune_rate'])
                if next_sparsity < target_sparsity:
                    next_sparsity = target_sparsity

                print(f"[PRUNE] Pruning to {next_sparsity*100:.2f}%...")
                new_mask = create_magnitude_mask_layerwise(
                    model, next_sparsity, cumulative_mask,
                    config['layer_specific_rates'], config['prune_rate']
                )

                if cumulative_mask is not None:
                    for name in new_mask:
                        new_mask[name] = new_mask[name] * cumulative_mask[name]

                cumulative_mask = new_mask
                current_sparsity = next_sparsity

                # Reset
                print("[RESET] Resetting to initial weights...")
                model.load_state_dict(initial_state)
                apply_mask(model, cumulative_mask)

                if current_sparsity <= target_sparsity:
                    break

            # Step 3: Final training
            print(f"\n{'='*50}")
            print(f"FINAL TRAINING at {current_sparsity*100:.2f}%")
            print(f"{'='*50}")

            optimizer = create_optimizer(model, config)
            results = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                cumulative_mask, config['eval_every'], config['verbose'],
                test_loader=test_loader
            )

            results['mask'] = cumulative_mask
            results['num_rounds'] = round_num
            results['actual_sparsity'] = get_sparsity(cumulative_mask, model)

            print_results_summary(results)
            all_results[target_sparsity].append(results)

    save_path = f"{config['results_dir']}/exp3_iterative_winning.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)


def run_experiment4_oneshot_reinit(config, exp2_results, train_loader, val_loader, test_loader, device):
    """
    Experiment 4: One-Shot Random Reinitialization
    """
    print("\n" + "="*70)
    print("EXPERIMENT 4: ONE-SHOT RANDOM REINITIALIZATION")
    print("="*70)

    exp_config = config['exp4']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for sparsity in exp_config['sparsity_levels']:
        winning_tickets = exp2_results[sparsity]

        for ticket_idx, ticket in enumerate(winning_tickets):
            mask_dict = ticket['mask']

            for reinit_num in range(1, exp_config['num_reinits'] + 1):
                print_experiment_header(
                    "One-Shot Reinit",
                    sparsity,
                    f"{ticket_idx+1}.{reinit_num}",
                    f"{len(winning_tickets)}x{exp_config['num_reinits']}"
                )

                # Create NEW model with random initialization
                model = create_model(config['model'], device=device)
                apply_mask(model, mask_dict)

                # Train from scratch
                optimizer = create_optimizer(model, config)
                results = train_with_early_stop_tracking(
                    model, device, train_loader, val_loader,
                    optimizer, criterion, config['max_iterations'],
                    mask_dict, config['eval_every'], config['verbose'],
                    test_loader=test_loader
                )

                print_results_summary(results)
                all_results[sparsity].append(results)

    save_path = f"{config['results_dir']}/exp4_oneshot_reinit.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)


def run_experiment5_iterative_reinit(config, exp3_results, train_loader, val_loader, test_loader, device):
    """
    Experiment 5: Iterative Random Reinitialization
    """
    print("\n" + "="*70)
    print("EXPERIMENT 5: ITERATIVE RANDOM REINITIALIZATION")
    print("="*70)

    exp_config = config['exp5']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for sparsity in exp_config['sparsity_levels']:
        if sparsity not in exp3_results:
            print(f"Skipping sparsity {sparsity} - not in exp3 results")
            continue

        winning_tickets = exp3_results[sparsity]

        for ticket_idx, ticket in enumerate(winning_tickets):
            mask_dict = ticket['mask']

            for reinit_num in range(1, exp_config['num_reinits'] + 1):
                print_experiment_header(
                    "Iterative Reinit",
                    sparsity,
                    f"{ticket_idx+1}.{reinit_num}",
                    f"{len(winning_tickets)}x{exp_config['num_reinits']}"
                )

                model = create_model(config['model'], device=device)
                apply_mask(model, mask_dict)

                optimizer = create_optimizer(model, config)
                results = train_with_early_stop_tracking(
                    model, device, train_loader, val_loader,
                    optimizer, criterion, config['max_iterations'],
                    mask_dict, config['eval_every'], config['verbose'],
                    test_loader=test_loader
                )

                print_results_summary(results)
                all_results[sparsity].append(results)

    save_path = f"{config['results_dir']}/exp5_iterative_reinit.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)