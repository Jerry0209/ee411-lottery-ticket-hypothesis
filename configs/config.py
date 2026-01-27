# Experiment configuration for Lottery Ticket Hypothesis experiments
config = {
    'dataset': 'MNIST',
    'data_path': './data',
    'train_size': 55000,
    'val_size': 5000,
    'batch_size': 60,

    'model': 'LeNet300_100',
    'one_shot_sparsity': [1.0, 0.5, 0.25, 0.1, 0.05],
    'iterative_sparsity': [1.0, 0.512, 0.262, 0.134, 0.035],

    'num_trials_random': 5,
    'num_trials_winning': 5,
    'num_reinits': 5,

    'max_iterations': 50000,
    'eval_every': 100,
    'optimizer': 'adam',
    'learning_rate': 0.0012,

    'prune_rate': 0.2,
    'layer_specific_rates': True,
    'pruning_strategy': 'magnitude',

    'verbose': True,
    'device': 'auto',
}

config['exp1'] = {
    'name': 'Random Sparse Networks',
    'sparsity_levels': config['one_shot_sparsity'],
    'num_trials': config['num_trials_random']
}

config['exp2'] = {
    'name': 'One-Shot Winning Tickets',
    'sparsity_levels': config['one_shot_sparsity'],
    'num_trials': config['num_trials_winning']
}

config['exp3'] = {
    'name': 'Iterative Winning Tickets',
    'sparsity_levels': config['iterative_sparsity'],
    'num_trials': config['num_trials_winning']
}

config['exp4'] = {
    'name': 'One-Shot Random Reinitialization',
    'sparsity_levels': config['one_shot_sparsity'],
    'num_reinits': config['num_reinits']
}

config['exp5'] = {
    'name': 'Iterative Random Reinitialization',
    'sparsity_levels': config['iterative_sparsity'],
    'num_reinits': config['num_reinits']
}

config['fig1'] = {
    'name': 'Winning Tickets vs Random Sparse',
    'requires': ['exp1', 'exp2']
}

config['fig3'] = {
    'name': 'Learning Curves',
    'requires': ['exp3', 'exp5']
}

config['fig4'] = {
    'name': 'Complete Comparison (All Methods)',
    'requires': ['exp2', 'exp3', 'exp4', 'exp5']
}
