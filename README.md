# Feasibility of Missing Value Imputation Using Synthetic Data: CVAE vs. MICE

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the codebase for the research project **"A Study on the Feasibility of Missing Data Imputation Using Synthetic Data: Focusing on a Comparison between CVAE and Multiple Imputation."**

This interdisciplinary project bridges deep generative modeling with psychological/behavioral data analysis, addressing missing data in large-scale tabular psychometric datasets. A Conditional Variational Autoencoder (CVAE) is implemented and evaluated as an imputation method for a Big Five personality dataset (N=100,000, 50 items) under multiple missingness patterns and rates.

---

## Overview

Missing data in behavioral experiments (e.g., Likert-scale psychometric tests) compromises study validity and statistical power. Multiple Imputation by Chained Equations (MICE), the conventional gold-standard method, relies on iterative conditional regression, whose computational cost scales poorly with dataset size — at N=100,000 this becomes computationally impractical, and MICE's convergence issues under high missingness and non-linear relationships are documented in prior literature.

This project therefore implements and evaluates a **CVAE-based imputation framework** as a single-pass alternative, testing its performance across four missingness patterns — MCAR (random) and three MNAR variants (negative-uni, positive-uni, bidirectional) — at 10%, 30%, and 50% missing rates (12 conditions total).

**Note:** MICE was not run directly on this dataset due to its computational infeasibility at this scale; comparisons to MICE's known limitations are based on prior literature rather than a direct empirical benchmark on this data.

---

## Key Technical Features

- **Tabular Generative Modeling**: Encoder-decoder CVAE architectures for Likert-scale, high-dimensional tabular data
- **Missingness Simulation**: Pipelines for injecting MCAR and MNAR (uni-directional and bidirectional) patterns at 10/30/50% rates
- **Evaluation Metrics**: Accuracy, MAE, and RMSE computed across all 12 (missingness type × rate) conditions

---

## Key Results Summary

Across all 12 conditions, CVAE achieved:
- **RMSE**: 1.18–1.67
- **MAE**: 0.77–1.26
- **Accuracy**: 0.27–0.45

Performance degraded gracefully as missingness increased (e.g., RMSE 1.18 → 1.28 for MCAR at 10% → 50%), rather than showing the abrupt convergence failure reported for MICE at high missingness in the literature. Degradation was more pronounced under MNAR conditions (e.g., negative-uni RMSE 1.47 → 1.67), consistent with the greater difficulty of reconstructing systematically non-random missingness.

Full result per versions are available in `https://docs.google.com/spreadsheets/d/1PyIyvoJUZUByq9l42EW5VofCBxhEVzk7rFuneFZlFvE/edit?usp=drive_link`.

---

## Repository Structure

```text
├── data/                     # Big Five personality dataset
├── Model/                    # CVAE model variants developed across iterations
│   ├── CVAE0811.pt
│   ├── model_0720.pt
│   └── ...
├── NoteBook/                 # Training/evaluation notebooks (multiple experimental runs)
│   ├── CVAE0626.ipynb
│   ├── CVAE260127.ipynb
│   └── ...
├── requirements.txt
└── README.md
```

Developed and tested in Google Colab (Python 3.12, CUDA 12.8).
*Note: `Model/` and `Notebook/` contain multiple iterations from the development process; the final reported results correspond to [NoteBook/CVAE260127].*
