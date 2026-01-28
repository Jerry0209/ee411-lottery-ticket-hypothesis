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
- **Katherine**: Conv-2 (CIFAR-10)
- **Furkan**: Conv-4 (CIFAR-10)
- **Xavier**: Conv-6 (CIFAR-10)
- **Gagan**: LeNet (MNIST), Winning Tickets
- **Tianrui (Jerry)**: ResNet-18, Pruning Algorithm

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

## Notebooks and Results

Attention please! We have created branches for each team member, where the training notebooks and results locate. Due to the different traing setup for each person, the training_utils.py and pruning.py in the main branch are not used by everyone. Each team member has own adapted version or different training utilities and model notebook. The details of each model training is in personal branch, please kindly check them in the corresponding branch.

In the main branch, we are trying to gather all the notebooks and results in 'Notebooks' and 'results_json_pt' folder, but we haven't finish it yet, so please refers to corresponding personal branch for each model.

Summaries for each model training are below.

### Conv-2
Details are in the corresponding branch.

### Conv-4
Details are in the corresponding branch.

### Conv-6
Details are in the corresponding branch.

### LeNet
Details are in the corresponding branch.

### ResNet-18
Details are in the corresponding branch.

**Model & Setup**

* **Architecture:** ResNet-20 (designed for CIFAR-10, ~270k parameters).
* **Dataset:** CIFAR-10 with standard augmentation (Random Crop, Horizontal Flip, Normalization).
* **Reproducibility:** Experiments conducted over **3 independent trials** (Seeds 42, 43, 44) to ensure statistical significance.

**Methodology**

1. **Iterative Magnitude Pruning (IMP):**
* Standard pipeline: Train  Prune 20% lowest magnitude weights  Reset remaining weights to   Repeat.


2. **Random Reinitialization (Control):**
* Validates the importance of initialization by applying the discovered sparsity mask () to a fresh set of random weights () rather than .


3. **Hyperparameters:**
* Optimizer: SGD with Momentum (0.9).
* **Learning Rate Study:** Investigated sensitivity to LR schedules:
* Standard (0.1)
* Low (0.01)
* **Linear Warmup (Peak 0.03)** – *Key finding for deep networks.*



**Key Files & Notebooks**

* **Core Scripts:**
* `training_utils.py`: Contains standardized `set_seed`, `train_epoch`, and `fit` functions.
* `pruning.py`: Implements `iterative_pruning` (IMP loop) and `random_reinit_pruning` (Control experiment).


* **Experiments:**
* `resnet18_iterative_random_init_experiment_LR_*.ipynb`: Separate notebooks for running experiments under LR 0.1, 0.01, and 0.03 (Warmup).
* `resnet18_create_figures_3_trials.ipynb`: Aggregates results from `.pth` checkpoints and generates visualization figures.


