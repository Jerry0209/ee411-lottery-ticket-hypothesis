"""
Test configuration for lottery ticket reproduction.
Intermediate between minimal and full - good for validation before full run.

Expected time: ~2-4 hours on GPU, ~8-12 hours on CPU
"""

config_test = {
    # Dataset
    'dataset': 'MNIST',
    'data_path': './data',
    'train_size': 55000,
    'val_size': 5000,
    'batch_size': 60,

    # Architecture
    'model': 'LeNet300_100',

    # Sparsity levels to test (more than minimal, key points from paper)
    'one_shot_sparsity': [1.0, 0.5, 0.25, 0.125, 0.0625],  # 5 levels

    # Iterative: Following 0.8^n pattern
    'iterative_sparsity': [
        1.0,      # 0.8^0 = 1.0
        0.512,    # 0.8^3 ~ 0.512
        0.262,    # 0.8^6 ~ 0.262
        0.134,    # 0.8^9 ~ 0.134
    ],

    # Experiment parameters
    'num_trials_random': 2,        # 2 trials for random sparse
    'num_trials_winning': 2,       # 2 trials for winning tickets
    'num_reinits': 2,              # 2 reinitializations per mask

    # Training
    'max_iterations': 35000,       # 35K iterations (70% of full)
    'eval_every': 100,             # Evaluate every 100 iterations
    'optimizer': 'adam',
    'learning_rate': 0.0012,       # From paper

    # Pruning
    'prune_rate': 0.2,             # Prune 20% per round (as in paper)
    'layer_specific_rates': True,  # 20% for hidden, 10% for output (as in paper)
    'pruning_strategy': 'magnitude',

    # Storage
    'save_models': False,          # Don't save models
    'save_learning_curves': True,  # Save learning curves for Figure 3
    'results_dir': './results/test',
    'figures_dir': './figures/test',

    # Logging
    'verbose': True,
    'log_file': './results/test/experiment.log',

    # Device
    'device': 'auto',  # 'auto', 'cuda', 'mps', or 'cpu'
}

# Experiment-specific configs
config_test['exp1'] = {
    'name': 'Random Sparse Networks',
    'sparsity_levels': config_test['one_shot_sparsity'],
    'num_trials': config_test['num_trials_random']
}

config_test['exp2'] = {
    'name': 'One-Shot Winning Tickets',
    'sparsity_levels': config_test['one_shot_sparsity'],
    'num_trials': config_test['num_trials_winning']
}

config_test['exp3'] = {
    'name': 'Iterative Winning Tickets',
    'sparsity_levels': config_test['iterative_sparsity'],
    'num_trials': config_test['num_trials_winning']
}

config_test['exp4'] = {
    'name': 'One-Shot Random Reinitialization',
    'sparsity_levels': config_test['one_shot_sparsity'],
    'num_reinits': config_test['num_reinits']
}

config_test['exp5'] = {
    'name': 'Iterative Random Reinitialization',
    'sparsity_levels': [0.512, 0.262],  # Key sparsity levels for Figure 3
    'num_reinits': config_test['num_reinits']
}
