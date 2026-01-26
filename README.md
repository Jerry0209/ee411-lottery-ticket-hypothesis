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



## Quick Start

1. **Clone the Repository and checkout to your branch (already created)**

    - Guideline at the end of the document.

2. **Install dependencies (if not yet)**
    - `requirements.txt` only for reference
    - There is also a reference environment on WSL in environment.yml in the main branch.

```bash
pip install -r requirements.txt
```


3. Copy the template:
```bash
   cp experiment_template.ipynb notebooks/your_name/your_model_experiment.ipynb
```

4. Modify three things:
   - Cell 3: Replace model definition
   - Cell 5: Change pruning config
   - Cell 6: Update model instance

5. Run all cells!

## Timeline （Updated on 06.01.2026）
- **Jan 6-7**: Setup repository structure
- **Jan 7-10**: Individual model training
- **Jan 10**: Team meeting

### For Team Meeting (Jan 10):
- Share your baseline accuracy
- Show your pruning curves
- Discuss any challenges or interesting findings
- Compare results across different models


## Git Workflow (Detailed Guide)

### Basic Git Commands You Need to Know

#### 1. **Check Status** (What changed?)
```bash
git status
```
This shows:
- Modified files (red = not staged, green = staged)
- New files
- Your current branch

#### 2. **Add Files** (Prepare to save)
```bash
# Add a specific file
git add notebooks/jerry/my_experiment.ipynb

# Add all files in a folder
git add notebooks/jerry/

# Add everything (use carefully!)
git add .
```

#### 3. **Commit** (Save changes locally)
```bash
git commit -m "Your message describing what you did"

# Example:
git commit -m "Add Conv-2 baseline training"
```

#### 4. **Push** (Upload to GitHub)
```bash
# First time pushing your branch
git push -u origin your-name-dev

# After that, just:
git push
```

#### 5. **Pull** (Download latest changes)
```bash
# Get latest changes from main branch
git checkout main
git pull origin main

# Go back to your branch
git checkout your-name-dev

# Merge latest changes into your branch
git merge main
```


#### First Time Setup (Only Once)

```bash
# 1. Clone the repository
cd ~
git clone git@github.com:your_username/ee411-lottery-ticket-hypothesis.git
cd ee411-lottery-ticket-hypothesis

# 2. Create Katherine's branch (not required to do that as it's already created)
git checkout -b katherine-dev

# 3. Verify branch
git branch
# Output: * katherine-dev

# 4. Copy template
cp experiment_template.ipynb notebooks/katherine/conv2_experiment.ipynb

# 5. Initial commit
git add notebooks/katherine/
git commit -m "Add Conv-2 experiment notebook template"

# 6. Push to GitHub
git push -u origin katherine-dev
```

#### Viewing Results on GitHub
1. Go to the repository on GitHub
2. Switch branches using the branch dropdown (top-left)
3. Navigate to `results/figures/` to view plots
4. Navigate to `results/data/` to view JSON results
5. Compare branches to see different team members' results

