# Lottery Ticket Hypothesis - Reproduction

Reproduction of "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carbin, 2019).

## Project Structure

```
lottery_ticket_reproduction/
├── configs/              # Configuration files
├── src/                  # Source code
│   ├── experiments/      # Experiment implementations
│   └── plotting/         # Plotting functions
├── scripts/              # Execution scripts
├── results/              # Experiment results
└── figures/              # Generated figures
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Minimal Test (40 minutes)
```bash
python scripts/run_all_experiments.py --config minimal
```

### 3. Generate Figures
```bash
python scripts/generate_figures.py --results results/minimal/
```

## Configurations

- **minimal**: 3 sparsity levels, 1 trial, 25K iterations (~40 min)
- **test**: 6 sparsity levels, 3 trials, 25K iterations (~5 hours)
- **full**: Full reproduction with 50K iterations (~4 days)

## Experiments

1. **Experiment 1**: Random sparse networks (baseline)
2. **Experiment 2**: One-shot magnitude pruning
3. **Experiment 3**: Iterative magnitude pruning
4. **Experiment 4**: One-shot random reinitialization
5. **Experiment 5**: Iterative random reinitialization

## Figures

- **Figure 1**: Overview comparison (random vs winning tickets)
- **Figure 3**: Learning curves showing training dynamics
- **Figure 4**: Complete analysis (4a, 4b, 4c)

## Citation

```
@inproceedings{frankle2019lottery,
  title={The lottery ticket hypothesis: Finding sparse, trainable neural networks},
  author={Frankle, Jonathan and Carbin, Michael},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```
