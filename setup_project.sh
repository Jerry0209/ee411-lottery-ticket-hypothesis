#!/bin/bash

# Lottery Ticket Hypothesis - Project Setup Script
# This script creates the complete directory structure and placeholder files

echo "Creating Lottery Ticket Hypothesis project structure..."

# Create main directories
mkdir -p configs
mkdir -p src/experiments
mkdir -p src/plotting
mkdir -p scripts
mkdir -p results/minimal
mkdir -p results/test
mkdir -p results/full
mkdir -p figures/minimal
mkdir -p figures/test
mkdir -p figures/full
mkdir -p notebooks

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/experiments/__init__.py
touch src/plotting/__init__.py
touch configs/__init__.py

# Create placeholder files (will be populated with code)
touch configs/config_minimal.py
touch configs/config_test.py
touch configs/config_full.py

touch src/models.py
touch src/data.py
touch src/training.py
touch src/pruning.py
touch src/utils.py

touch src/experiments/exp1_random.py
touch src/experiments/exp2_oneshot.py
touch src/experiments/exp3_iterative.py
touch src/experiments/exp4_reinit_oneshot.py
touch src/experiments/exp5_reinit_iterative.py

touch src/plotting/figure1.py
touch src/plotting/figure3.py
touch src/plotting/figure4.py

touch scripts/run_all_experiments.py
touch scripts/generate_figures.py

# Create requirements.txt
cat > requirements.txt << 'EOF'
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.7.0
tqdm>=4.65.0
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
*.egg-info/

# Data and results
data/
results/
*.pt
*.pth
*.log

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF

# Create README
cat > README.md << 'EOF'
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
EOF

echo ""
echo "[OK] Directory structure created!"
echo ""
echo "Project structure:"
tree -L 3 -I '__pycache__|*.pyc|data' 2>/dev/null || find . -type d -not -path '*/\.*' | sed 's|[^/]*/| |g'

echo ""
echo "Next steps:"
echo "1. Copy the Python code into the respective files"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Run minimal test: python scripts/run_all_experiments.py --config minimal"
echo ""
echo "Done!"