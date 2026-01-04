# EE-411 Lottery Ticket Hypothesis

Reproducing "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carbin, 2019)

## Team Members
- **Jerry**: ResNet-18, Conv-2, Pruning Algorithm
- **Katherine**: Conv-2
- **Furkan**: Conv-4  
- **Xavier**: Conv-6 (CIFAR-10)
- **Gagan**: LeNet (MNIST), Winning Tickets

## Project Structure
```
ee411-lottery-ticket-hypothesis/
├── shared/              # Shared utilities (.py files)
│   ├── training_utils.py
│   ├── data_loaders.py
│   ├── pruning.py
│   └── visualization.py
├── notebooks/           # Individual experiments (.ipynb)
│   ├── jerry/
│   ├── katherine/
│   ├── furkan/
│   ├── xavier/
│   └── gagan/
└── results/            # Saved models and figures
```

## Quick Start

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Work in your notebook**
- Define your model in `notebooks/your_name/`
- Import shared utilities: `from shared.training_utils import fit, predict`

## Timeline
- **Jan 6-7**: Setup repository structure
- **Jan 7-10**: Individual model training
- **Jan 10**: Team meeting
- **Jan 10-19**: Pruning experiments
- **Jan 28**: Final submission
