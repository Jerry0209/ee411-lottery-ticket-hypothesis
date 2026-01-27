# Lottery Ticket Hypothesis - Reproduction

Reproduction of "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carlin, 2019).

## Project Structure

```
EE-411/
├── main.py               # Entry point - runs experiments and generates figures
├── configs/
│   └── config.py         # All configuration (experiments, figures, hyperparameters)
├── src/
│   ├── data.py           # Dataset loading (MNIST, CIFAR-10)
│   ├── models.py         # Neural network architectures (LeNet300_100)
│   ├── training.py       # Training and validation loops
│   ├── pruning.py        # Pruning utilities and mask operations
│   ├── utils.py          # General utilities (I/O, device detection)
│   ├── experiments.py    # Experiment implementations (exp1-5)
│   └── figures.py        # Figure generation (fig1, fig3, fig4)
└── results/              # Output directory (created automatically)
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Experiments
```bash
python main.py
```

This runs all experiments defined in `configs/config.py` and automatically generates figures.

## Configuration

All settings are in `configs/config.py`. Edit this file to:

- **Select experiments**: Comment out `exp1`-`exp5` entries to skip them
- **Select figures**: Comment out `fig1`, `fig3`, `fig4` entries to skip them
- **Adjust parameters**: Change sparsity levels, iterations, trials, etc.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset` | Dataset to use | `'MNIST'` |
| `model` | Network architecture | `'LeNet300_100'` |
| `max_iterations` | Training iterations per run | `2000` |
| `one_shot_sparsity` | Sparsity levels for one-shot pruning | `[1.0, 0.5]` |
| `iterative_sparsity` | Sparsity levels for iterative pruning | `[1.0, 0.512]` |
| `prune_rate` | Fraction pruned per round (iterative) | `0.2` |

## Experiments

| Experiment | Description |
|------------|-------------|
| **exp1** | Random sparse networks (baseline) |
| **exp2** | One-shot magnitude pruning with original initialization |
| **exp3** | Iterative magnitude pruning with original initialization |
| **exp4** | One-shot pruning with random reinitialization |
| **exp5** | Iterative pruning with random reinitialization |

## Figures

| Figure | Description | Requires |
|--------|-------------|----------|
| **fig1** | Random sparse vs winning tickets comparison | exp1, exp2 |
| **fig3** | Learning curves showing training dynamics | exp3, exp5 |
| **fig4** | Complete comparison of all methods | exp2, exp3, exp4, exp5 |

## Output

Results are saved to `results/<model>_<dataset>/<timestamp>/`:
- `metrics/` - Experiment data (.pt files)
- `figures/` - Generated plots (.png files)

## Citation

```bibtex
@inproceedings{frankle2019lottery,
  title={The lottery ticket hypothesis: Finding sparse, trainable neural networks},
  author={Frankle, Jonathan and Carlin, Michael},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```
