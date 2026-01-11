def run_experiment3_iterative_winning(config, train_loader, val_loader, device):
    """
    Experiment 3: Iterative Winning Tickets
    
    Repeatedly train → prune 20% → reset → repeat until target sparsity reached.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: ITERATIVE WINNING TICKETS")
    print("="*70)
    
    exp_config = config['exp3']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()
    
    from ..pruning import create_magnitude_mask_layerwise, get_sparsity
    
    for target_sparsity in exp_config['sparsity_levels']:
        for trial in range(1, exp_config['num_trials'] + 1):
            print_experiment_header("Iterative Winning Ticket", target_sparsity, trial, exp_config['num_trials'])
            
            # Step 1: Create model and save initial weights
            print(f"[1] Saving θ₀...")
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
                print(f"[TRAIN] Training...")
                optimizer = create_optimizer(model, config)
                _ = train_with_early_stop_tracking(
                    model, device, train_loader, val_loader,
                    optimizer, criterion, config['max_iterations'],
                    cumulative_mask, config['eval_every'], config['verbose']
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
                print(f"[RESET] Resetting to θ₀...")
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
                cumulative_mask, config['eval_every'], config['verbose']
            )
            
            results['mask'] = cumulative_mask
            results['num_rounds'] = round_num
            results['actual_sparsity'] = get_sparsity(cumulative_mask, model)
            
            print_results_summary(results)
            all_results[target_sparsity].append(results)
    
    save_path = f"{config['results_dir']}/exp3_iterative_winning.pt"
    save_results(dict(all_results), save_path)
    
    return dict(all_results)