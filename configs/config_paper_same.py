"""
Paper-exact reproduction configuration for lottery ticket experiments.
This matches the paper's methodology EXACTLY including trial counts.

Paper: "The Lottery Ticket Hypothesis" (1803.03635v5)

Expected time: ~200+ hours on GPU (significantly longer due to more trials)
"""

config_paper_same = {
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

    # Experiment parameters - EXACTLY matching the paper
    'num_trials_random': 10,       # Paper: 10 trials for random sparse networks
    'num_trials_winning': 5,       # Paper: 5 trials for winning tickets
    'num_reinits': 3,              # Paper: 3 random reinitializations per ticket

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
    'results_dir': './results/paper_same',
    'figures_dir': './figures/paper_same',

    # Logging
    'verbose': True,
    'log_file': './results/paper_same/experiment.log',

    # Device
    'device': 'auto',  # 'auto', 'cuda', 'mps', or 'cpu'
}

# Experiment-specific configs
config_paper_same['exp1'] = {
    'name': 'Random Sparse Networks',
    'sparsity_levels': config_paper_same['one_shot_sparsity'],
    'num_trials': config_paper_same['num_trials_random']
}

config_paper_same['exp2'] = {
    'name': 'One-Shot Winning Tickets',
    'sparsity_levels': config_paper_same['one_shot_sparsity'],
    'num_trials': config_paper_same['num_trials_winning']
}

config_paper_same['exp3'] = {
    'name': 'Iterative Winning Tickets',
    'sparsity_levels': config_paper_same['iterative_sparsity'],
    'num_trials': config_paper_same['num_trials_winning']
}

config_paper_same['exp4'] = {
    'name': 'One-Shot Random Reinitialization',
    'sparsity_levels': config_paper_same['one_shot_sparsity'],
    'num_reinits': config_paper_same['num_reinits']
}

config_paper_same['exp5'] = {
    'name': 'Iterative Random Reinitialization',
    'sparsity_levels': [0.512, 0.262],  # Only show these in Figure 3
    'num_reinits': config_paper_same['num_reinits']
}
