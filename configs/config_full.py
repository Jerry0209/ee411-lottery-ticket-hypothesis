"""
Full reproduction configuration for lottery ticket experiments.
This matches the paper's methodology as closely as possible.

Expected time: ~100 hours on GPU (4-5 days)
"""

config_full = {
    # Dataset
    'dataset': 'MNIST',
    'data_path': './data',
    'train_size': 55000,
    'val_size': 5000,
    'batch_size': 60,

    # Architecture
    'model': 'LeNet300_100',

    # Sparsity levels to test
    # One-shot: Arbitrary levels for Figure 1 & 4c
    'one_shot_sparsity': [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625],

    # Iterative: Following 0.8^n pattern (20% pruning per round)
    'iterative_sparsity': [
        1.0,      # 0.8^0 = 1.0
        0.512,    # 0.8^3 ~ 0.512
        0.262,    # 0.8^6 ~ 0.262
        0.134,    # 0.8^9 ~ 0.134
        0.069,    # 0.8^12 ~ 0.069
        0.035,    # 0.8^15 ~ 0.035
    ],

    # Experiment parameters
    'num_trials_random': 5,        # Paper uses 10, we use 5 for speed
    'num_trials_winning': 3,       # Paper uses 5, we use 3
    'num_reinits': 2,              # Paper uses 3, we use 2

    # Training
    'max_iterations': 50000,       # Full 50K iterations as in paper
    'eval_every': 100,             # Evaluate every 100 iterations
    'optimizer': 'adam',
    'learning_rate': 0.0012,       # From paper

    # Pruning
    'prune_rate': 0.2,             # Prune 20% per round (as in paper)
    'layer_specific_rates': True,  # 20% for hidden, 10% for output
    'pruning_strategy': 'magnitude',

    # Storage
    'save_models': False,          # Don't save models (saves disk space)
    'save_learning_curves': True,  # Save learning curves for Figure 3
    'results_dir': './results/full',
    'figures_dir': './figures/full',

    # Logging
    'verbose': True,
    'log_file': './results/full/experiment.log',

    # Device
    'device': 'auto',  # 'auto', 'cuda', 'mps', or 'cpu'
}

# Experiment-specific configs
config_full['exp1'] = {
    'name': 'Random Sparse Networks',
    'sparsity_levels': config_full['one_shot_sparsity'],
    'num_trials': config_full['num_trials_random']
}

config_full['exp2'] = {
    'name': 'One-Shot Winning Tickets',
    'sparsity_levels': config_full['one_shot_sparsity'],
    'num_trials': config_full['num_trials_winning']
}

config_full['exp3'] = {
    'name': 'Iterative Winning Tickets',
    'sparsity_levels': config_full['iterative_sparsity'],
    'num_trials': config_full['num_trials_winning']
}

config_full['exp4'] = {
    'name': 'One-Shot Random Reinitialization',
    'sparsity_levels': config_full['one_shot_sparsity'],
    'num_reinits': config_full['num_reinits']
}

config_full['exp5'] = {
    'name': 'Iterative Random Reinitialization',
    'sparsity_levels': [0.512, 0.262],  # Only show these in Figure 3
    'num_reinits': config_full['num_reinits']
}
