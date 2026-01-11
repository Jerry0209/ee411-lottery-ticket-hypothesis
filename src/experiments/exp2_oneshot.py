def run_experiment2_oneshot_winning(config, train_loader, val_loader, device):
    """
    Experiment 2: One-Shot Winning Tickets
    
    Train full network → prune by magnitude → reset to initial weights → retrain.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: ONE-SHOT WINNING TICKETS")
    print("="*70)
    
    exp_config = config['exp2']
    all_results = defaultdict(list)
    criterion = nn.CrossEntropyLoss()
    
    from ..pruning import create_magnitude_mask_layerwise
    
    for sparsity in exp_config['sparsity_levels']:
        for trial in range(1, exp_config['num_trials'] + 1):
            print_experiment_header("One-Shot Winning Ticket", sparsity, trial, exp_config['num_trials'])
            
            # Step 1: Create model and save initial weights
            print("[1] Creating model and saving θ₀...")
            model = create_model(config['model'], device=device)
            initial_state = copy.deepcopy(model.state_dict())
            
            # Step 2: Train full network
            print("[2] Training full network...")
            optimizer = create_optimizer(model, config)
            _ = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                None, config['eval_every'], config['verbose']
            )
            
            # Step 3: Create magnitude mask
            print(f"[3] Creating magnitude mask at {sparsity*100:.1f}%...")
            mask_dict = create_magnitude_mask_layerwise(
                model, sparsity, None,
                config['layer_specific_rates'], config['prune_rate']
            )
            
            # Step 4: Reset to initial weights
            print("[4] Resetting to θ₀ and training winning ticket...")
            model.load_state_dict(initial_state)
            apply_mask(model, mask_dict)
            
            # Step 5: Train winning ticket
            optimizer = create_optimizer(model, config)
            results = train_with_early_stop_tracking(
                model, device, train_loader, val_loader,
                optimizer, criterion, config['max_iterations'],
                mask_dict, config['eval_every'], config['verbose']
            )
            
            results['mask'] = mask_dict
            print_results_summary(results)
            all_results[sparsity].append(results)
    
    save_path = f"{config['results_dir']}/exp2_oneshot_winning.pt"
    save_results(dict(all_results), save_path)
    
    return dict(all_results)