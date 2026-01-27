# EE-411 Lottery Ticket Hypothesis

Reproducing "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks" (Frankle & Carbin, 2019)

## Team Members
- **Katherine**: Conv-2 (CIFAR-10)
- **Furkan**: Conv-4 (CIFAR-10)
- **Xavier**: Conv-6 (CIFAR-10)
- **Gagan**: LeNet (MNIST), Winning Tickets
- **Jerry**: ResNet-18, Pruning Algorithm

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

**Good commit messages:**
- ✅ "Add Conv-2 model definition"
- ✅ "Fix bug in pruning algorithm"
- ✅ "Complete baseline training experiments"

**Bad commit messages:**
- ❌ "update"
- ❌ "fix"
- ❌ "asdfasdf"

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


## 📖 Complete Git Workflow Example

### Scenario: Katherine wants to work on Conv-2

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

#### Daily Workflow

**Each time you start working:**

```bash
# 1. Make sure you're on your branch
git checkout katherine-dev

# 2. Get latest changes from main
git checkout main
git pull origin main
git checkout katherine-dev
git merge main

# 3. Start working...
```

**After you finish working:**

```bash
# 1. Check what changed
git status

# 2. Add your changes
git add notebooks/katherine/conv2_experiment.ipynb
git add results/figures/katherine_*.png       

# 3. Commit with a message
git commit -m "Complete Conv-2 baseline training"

# 4. Push to GitHub
git push

# Done! Your work is saved online.
```

#### Viewing Results on GitHub
1. Go to the repository on GitHub
2. Switch branches using the branch dropdown (top-left)
3. Navigate to `results/figures/` to view plots
4. Navigate to `results/data/` to view JSON results
5. Compare branches to see different team members' results



## 🛠️ Common Git Scenarios

### Scenario 1: "I modified some files, how do I save them?"

```bash
# 1. Check what you changed
git status

# 2. Add files you want to save
git add notebooks/your-name/your_file.ipynb

# 3. Commit
git commit -m "Describe what you did"

# 4. Push to GitHub
git push
```

### Scenario 2: "I want to see the latest code from the team"

```bash
# 1. Go to main branch
git checkout main

# 2. Download latest changes
git pull origin main

# 3. Go back to your branch
git checkout your-name-dev

# 4. Merge latest changes
git merge main
```

### Scenario 3: "I accidentally modified the wrong file"

```bash
# Undo changes to a file (before committing)
git checkout -- path/to/file.py

# Example:
git checkout -- shared/pruning.py
```

### Scenario 4: "I committed something wrong"

```bash
# Undo the last commit (keep changes)
git reset --soft HEAD~1

# Now you can modify and commit again
```

### Scenario 5: "Git says there's a conflict"

**This happens when you and others both modified the same file.**

```bash
# Git will show something like:
# CONFLICT (content): Merge conflict in shared/pruning.py

# 1. Open the file in a text editor
# 2. Look for conflict markers:
<<<<<<< HEAD
Your changes
=======
Jerry's changes
>>>>>>> main

# 3. Decide what to keep:
#    - Keep your version, or
#    - Keep Jerry's version, or
#    - Combine both
# 4. Remove the conflict markers (<<<, ===, >>>)
# 5. Save the file
# 6. Add and commit:
git add shared/pruning.py
git commit -m "Resolve merge conflict in pruning.py"
git push
```



## ⚠️ Important Rules

### ✅ DO:
1. **Work on your own branch** (`your-name-dev`)
2. **Commit often** (every 1-2 hours of work)
3. **Write clear commit messages**
4. **Only modify files in your folder** (`notebooks/your-name/`)
5. **Pull from main before starting work each day**
6. **Ask for help if confused** (better than breaking things!)

### ❌ DON'T:
1. **Don't work directly on `main` branch**
2. **Don't modify other people's notebooks**
3. **Don't commit large binary files** (> 10MB)
4. **Don't commit passwords or API keys**
5. **Don't use `git push --force`** (unless you know what you're doing)

### Special Rule for `shared/` folder:
- **Only Jerry modifies files in `shared/`**
- If you find a bug, tell Jerry (don't fix it yourself)
- This prevents conflicts


## Quick Reference Card

```bash
# 🚀 Start working
git checkout your-name-dev
git pull origin main
jupyter notebook

# 💾 Save your work
git add notebooks/your-name/
git commit -m "Description"
git push

# 📥 Get team's updates
git checkout main
git pull origin main
git checkout your-name-dev
git merge main

# ❓ Check status
git status
git branch

# 🆘 Undo changes
git checkout -- filename
```
