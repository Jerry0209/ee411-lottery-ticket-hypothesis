def run_experiment5_iterative_reinit(config, exp3_results, train_loader, val_loader, device):
    """Experiment 5: Iterative Random Reinitialization"""
    print("\n" + "="*70)
    print("EXPERIMENT 5: ITERATIVE RANDOM REINITIALIZATION")
    print("="*70)
    
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
                    mask_dict, config['eval_every'], config['verbose']
                )
                
                print_results_summary(results)
                all_results[sparsity].append(results)
    
    save_path = f"{config['results_dir']}/exp5_iterative_reinit.pt"
    save_results(dict(all_results), save_path)
    
    return dict(all_results)