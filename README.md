# Conv-4 Experiments – Lottery Ticket Hypothesis

This branch contains my individual experiments for the EE-411 project  
**“The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks”**  
(Frankle & Carbin, 2019).

The focus of this branch is the **Conv-4 architecture trained on CIFAR-10**.

---

## Scope of This Branch

This branch includes:

- Final Conv-4 experiments used for analysis and reporting
- Intermediate trial experiments used during development
- Result files (plots and logs) necessary to interpret training behavior

Team-wide structure, shared utilities, and coordination details are intentionally omitted here.

---

## Folder Structure


### `trials/`
This folder contains **experimental notebooks** used to:
- Test pruning schedules
- Tune hyperparameters
- Debug training instabilities
- Validate early-stopping behavior

Results in this folder are **not meant to be reported**, but kept for transparency and reproducibility.

### `main/`
This folder contains the **final Conv-4 experiments** referenced in analysis and figures.

---

## Notes on Result Files

- During training, results were originally saved **at every iteration**, which caused significant slowdowns.
- To reduce runtime, logging frequency was later reduced and experiments were **split into multiple runs**.
- Although some runs do not cover the full iteration range continuously,  
  **the relevant training dynamics (early stopping, accuracy trends, pruning behavior)** are still clearly observable.
- All figures included in `results/figures/` are generated from these saved results.

This trade-off was necessary to complete experiments within reasonable time limits.

---

## Reproducibility

Experiments in `main/` can be reproduced by running the corresponding notebooks.
Trial notebooks are kept as-is and may not be fully cleaned or optimized.

---

## Author

**Furkan Akkoyun**  
EE-411 – EPFL  
Conv-4 (CIFAR-10)
