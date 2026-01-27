import torch
import torch.nn as nn
import copy
from collections import defaultdict

from .models import LeNet300_100
from .training import train_with_early_stop_tracking, create_optimizer
from .pruning import (create_random_mask, apply_mask,
                        create_magnitude_mask_layerwise, get_sparsity)
from .utils import save_results


def create_initialized_model(model_name, device):
    """Create and initialize model with Xavier initialization."""
    if model_name == 'LeNet300_100':
        model = LeNet300_100()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    return model.to(device)


def run_experiment1_random_sparse(config, train_loader, val_loader, test_loader, device):
    """Experiment 1: Train randomly pruned sparse networks."""
    exp_config = config['exp1']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for sparsity in exp_config['sparsity_levels']:
        for trial in range(1, exp_config['num_trials'] + 1):
            model = create_initialized_model(config['model'], device)
            mask_dict = create_random_mask(model, sparsity)
            apply_mask(model, mask_dict)
            optimizer = create_optimizer(model, config)
            results = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                mask_dict, config['eval_every'], config['verbose'],
                test_loader=test_loader
            )
            all_results[sparsity].append(results)

    save_path = f"{config['results_dir']}/exp1_random_sparse.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)


def run_experiment2_oneshot_winning(config, train_loader, val_loader, test_loader, device):
    """Experiment 2: One-shot magnitude pruning (winning tickets)."""
    exp_config = config['exp2']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for sparsity in exp_config['sparsity_levels']:
        for trial in range(1, exp_config['num_trials'] + 1):
            model = create_initialized_model(config['model'], device)
            initial_state = copy.deepcopy(model.state_dict())
            optimizer = create_optimizer(model, config)
            _ = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                None, config['eval_every'], config['verbose'],
                test_loader=test_loader
            )
            mask_dict = create_magnitude_mask_layerwise(
                model, sparsity, None,
                config['layer_specific_rates'], config['prune_rate']
            )
            model.load_state_dict(initial_state)
            apply_mask(model, mask_dict)
            optimizer = create_optimizer(model, config)
            results = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                mask_dict, config['eval_every'], config['verbose'],
                test_loader=test_loader
            )
            results['mask'] = mask_dict
            all_results[sparsity].append(results)

    save_path = f"{config['results_dir']}/exp2_oneshot_winning.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)


def run_experiment3_iterative_winning(config, train_loader, val_loader, test_loader, device):
    """Experiment 3: Iterative magnitude pruning (winning tickets)."""
    exp_config = config['exp3']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for target_sparsity in exp_config['sparsity_levels']:
        for trial in range(1, exp_config['num_trials'] + 1):
            model = create_initialized_model(config['model'], device)
            initial_state = copy.deepcopy(model.state_dict())
            current_sparsity = 1.0
            cumulative_mask = None
            round_num = 0

            while current_sparsity > target_sparsity + 0.001:
                round_num += 1
                optimizer = create_optimizer(model, config)
                _ = train_with_early_stop_tracking(
                    model, device, train_loader, val_loader,
                    optimizer, criterion, config['max_iterations'],
                    cumulative_mask, config['eval_every'], config['verbose'],
                    test_loader=test_loader
                )
                next_sparsity = current_sparsity * (1 - config['prune_rate'])
                if next_sparsity < target_sparsity:
                    next_sparsity = target_sparsity
                new_mask = create_magnitude_mask_layerwise(
                    model, next_sparsity, cumulative_mask,
                    config['layer_specific_rates'], config['prune_rate']
                )

                if cumulative_mask is not None:
                    for name in new_mask:
                        new_mask[name] = new_mask[name] * cumulative_mask[name]

                cumulative_mask = new_mask
                current_sparsity = next_sparsity
                model.load_state_dict(initial_state)
                apply_mask(model, cumulative_mask)

                if current_sparsity <= target_sparsity:
                    break

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
            all_results[target_sparsity].append(results)

    save_path = f"{config['results_dir']}/exp3_iterative_winning.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)


def run_experiment4_oneshot_reinit(config, exp2_results, train_loader, val_loader, test_loader, device):
    """Experiment 4: One-shot winning tickets with random reinitialization."""
    exp_config = config['exp4']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for sparsity in exp_config['sparsity_levels']:
        winning_tickets = exp2_results[sparsity]

        for ticket_idx, ticket in enumerate(winning_tickets):
            mask_dict = ticket['mask']

            for reinit_num in range(1, exp_config['num_reinits'] + 1):
                model = create_initialized_model(config['model'], device)
                if mask_dict is not None:
                    apply_mask(model, mask_dict)
                optimizer = create_optimizer(model, config)
                results = train_with_early_stop_tracking(
                    model, device, train_loader, val_loader,
                    optimizer, criterion, config['max_iterations'],
                    mask_dict, config['eval_every'], config['verbose'],
                    test_loader=test_loader
                )
                all_results[sparsity].append(results)

    save_path = f"{config['results_dir']}/exp4_oneshot_reinit.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)


def run_experiment5_iterative_reinit(config, exp3_results, train_loader, val_loader, test_loader, device):
    """Experiment 5: Iterative winning tickets with random reinitialization."""
    exp_config = config['exp5']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()

    for sparsity in exp_config['sparsity_levels']:
        if sparsity not in exp3_results:
            continue
        winning_tickets = exp3_results[sparsity]
        for ticket_idx, ticket in enumerate(winning_tickets):
            mask_dict = ticket['mask']
            for reinit_num in range(1, exp_config['num_reinits'] + 1):
                model = create_initialized_model(config['model'], device)
                if mask_dict is not None:
                    apply_mask(model, mask_dict)

                optimizer = create_optimizer(model, config)
                results = train_with_early_stop_tracking(
                    model, device, train_loader, val_loader,
                    optimizer, criterion, config['max_iterations'],
                    mask_dict, config['eval_every'], config['verbose'],
                    test_loader=test_loader
                )
                all_results[sparsity].append(results)

    save_path = f"{config['results_dir']}/exp5_iterative_reinit.pt"
    save_results(dict(all_results), save_path)

    return dict(all_results)
