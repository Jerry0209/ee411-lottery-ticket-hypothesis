"""
Minimal test configuration for lottery ticket reproduction.
Quick validation of all experiments and plotting pipeline.
"""

config_minimal = {
    # Dataset
    'dataset': 'MNIST',
    'data_path': './data',
    'train_size': 55000,
    'val_size': 5000,
    'batch_size': 60,
    
    # Architecture
    'model': 'LeNet300_100',
    
    # Sparsity levels to test
    'one_shot_sparsity': [1.0, 0.5, 0.25],           # Just 3 levels
    'iterative_sparsity': [1.0, 0.64, 0.41],         # 3 levels (max 3 rounds)
    
    # Experiment parameters
    'num_trials_random': 1,        # Number of trials for random sparse
    'num_trials_winning': 1,       # Number of trials for winning tickets
    'num_reinits': 1,              # Number of reinitializations per mask
    
    # Training
    'max_iterations': 25000,       # Training iterations per round
    'eval_every': 100,             # Evaluate every N iterations
    'optimizer': 'adam',
    'learning_rate': 0.0012,       # From paper
    
    # Pruning
    'prune_rate': 0.2,             # Prune 20% per round
    'layer_specific_rates': False, # If True: 20% hidden, 10% output
    'pruning_strategy': 'magnitude', # 'magnitude' or 'random'
    
    # Storage
    'save_models': False,          # Don't save models for minimal test
    'save_learning_curves': True,  # Do save learning curves
    'results_dir': './results/minimal',
    'figures_dir': './figures/minimal',
    
    # Logging
    'verbose': True,
    'log_file': './results/minimal/experiment.log',
    
    # Device
    'device': 'auto',  # 'auto', 'cuda', 'mps', or 'cpu'
}

# Experiment-specific configs
config_minimal['exp1'] = {
    'name': 'Random Sparse Networks',
    'sparsity_levels': config_minimal['one_shot_sparsity'],
    'num_trials': config_minimal['num_trials_random']
}

config_minimal['exp2'] = {
    'name': 'One-Shot Winning Tickets',
    'sparsity_levels': config_minimal['one_shot_sparsity'],
    'num_trials': config_minimal['num_trials_winning']
}

config_minimal['exp3'] = {
    'name': 'Iterative Winning Tickets',
    'sparsity_levels': config_minimal['iterative_sparsity'],
    'num_trials': config_minimal['num_trials_winning']
}

config_minimal['exp4'] = {
    'name': 'One-Shot Random Reinitialization',
    'sparsity_levels': config_minimal['one_shot_sparsity'],
    'num_reinits': config_minimal['num_reinits']
}

config_minimal['exp5'] = {
    'name': 'Iterative Random Reinitialization',
    'sparsity_levels': [0.64, 0.41],  # Only show 2 levels in Figure 3
    'num_reinits': config_minimal['num_reinits']
}