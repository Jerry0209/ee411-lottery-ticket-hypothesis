# EE-411 Lottery Ticket Hypothesis

This repository contains a reproduction and empirical analysis of
“The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks” by Frankle & Carbin (ICLR 2019).

We investigate whether dense neural networks contain sparse subnetworks (winning tickets) that can be trained in isolation to reach comparable accuracy
when initialized with the same original weights.

- Reproduced one-shot and iterative magnitude pruning experiments
- Evaluated LeNet, Conv-2, Conv-4, Conv-6, and ResNet-20
- Experiments on MNIST and CIFAR-10
- Analyzed on:
      Final accuracy,
      Training speed (early stopping iteration),
      Effect of initialization and learning-rate schedules
  
## Team Members
- **Katherine**: Conv-2
- **Furkan**: Conv-4  
- **Xavier**: Conv-6 (CIFAR-10)
- **Gagan**: LeNet (MNIST), Winning Tickets
- **Jerry**: ResNet-18, Pruning Algorithm

Each of the team member contributed to the writing the report.

## Project Structure
```
ee411-lottery-ticket-hypothesis/
├── shared/              # Shared utilities (.py files)
│   ├── training_utils.py
│   ├── pruning.py
│   └── visualization.py (not available currently)
├── notebooks/           # Individual experiments (.ipynb)
│   ├── katherine/
│   ├── xavier/
│   ├── gagan/
│   ├── furkan/
│   └── jerry/
├── results/                     # Experiment results (git-ignored)
│   ├── checkpoints/             # Saved model weights (.pth)
│   ├── logs/                    # Training logs
│   └── figures/                 # Generated plots
│
├── data/                        # Dataset files (git-ignored, auto-downloaded)
│
├── experiment_template.ipynb    # Template notebook for experiments
├── requirements.txt             # Python dependencies
├── environment.yml              # Conda environment on WSL
├── .gitignore                   # Git ignore rules
└── README.md                    # This file

```

## Timeline （Updated on 26.01.2026）
The project was developed step by step, with regular meetings and close collaboration among team members. We first set up a shared repository structure and common experiment templates so that all models could be trained and evaluated in a consistent way. Different architectures, including LeNet, Conv-2/4/6, and ResNet-20, were then trained and pruned independently by team members, which allowed us to explore multiple settings in parallel.

During the project, we met regularly to share baseline results, pruning curves, and observations about training behavior. Final figures and conclusions were obtained by checking consistency across implementations, ensuring that the results are reproducible and fairly comparable across different pruning strategies.

4. Navigate to `results/data/` to view JSON results
5. Compare branches to see different team members' results

