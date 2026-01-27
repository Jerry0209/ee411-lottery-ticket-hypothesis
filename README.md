# Conv-4 Experiments – Lottery Ticket Hypothesis

This branch contains my individual experiments for the EE-411 course project  
**“The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks”**  
by Frankle & Carbin (2019).

The experiments in this branch focus exclusively on the **Conv-4 architecture trained on CIFAR-10**.

---

## Scope of This Branch

This branch includes all work related to the Conv-4 model, organized into:
- Final experiments used in analysis and reporting
- Intermediate trial experiments conducted during development
- Result files (JSON logs and figures) used to extract stable measurements

Team-wide structure, shared utilities, and coordination details are intentionally excluded.

---

## Folder Structure

### `main/`

This folder contains the **final and authoritative Conv-4 experiments**.

- `conv4_winning_ticket.ipynb` is the **primary reference notebook** for Conv-4.
- Experimental results are saved in **JSON files** under the `results/` directory.
- All figures used in analysis are generated directly from these JSON result files.

### `trials/`

This folder contains **experimental and exploratory notebooks** used during development to:
- Test pruning schedules
- Tune hyperparameters
- Debug training instabilities
- Validate early-stopping behavior

Notebooks in this folder may include:
- Interrupted executions
- Partial runs
- Temporary debugging or logging code

Results in `trials/` are **not intended for reporting**, but are kept for transparency and reproducibility of the development process.

---

## Notes on Result Files

- During early development, results were logged at every iteration, which caused significant slowdowns.
- To reduce runtime, logging frequency was later reduced and experiments were split across multiple executions.
- As a result, some notebooks may display interrupted runs or incomplete iteration ranges.

These artifacts **do not affect the validity of the reported results**.

All final experimental outcomes were:
- Collected from completed runs
- Verified for consistency
- Consolidated into dedicated JSON result files

All figures in `results/figures/` are generated exclusively from these consolidated results.

---

## Reproducibility

- Experiments in `main/` can be reproduced by running the corresponding notebooks.
- Random seeds, pruning masks, and training configurations are logged in the result files.
- Trial notebooks are provided as-is and may not be fully cleaned or optimized.

---

## Author

**Furkan Akkoyun**  
EE-411 – EPFL  
Conv-4 (CIFAR-10)
