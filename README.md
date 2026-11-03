# Prediction of Concrete Compressive Strength

This repository holds the code, papers and data for two published studies on predicting concrete compressive strength with stacking ensembles of gradient-boosting models.

Both studies use the same benchmark dataset. They pursue different goals:

- **Work 1** aims for **accuracy**. It uses domain-informed features, Optuna tuning and a leakage-free stacking ensemble, and explains the model with SHAP.
- **Work 2** aims for **reliability**. It enforces physics-based monotonicity constraints, uses out-of-fold stacking and adds prediction intervals from quantile regression.

---

## Dataset

**UCI Concrete Compressive Strength** — [archive.ics.uci.edu/dataset/165](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

- 1030 laboratory concrete mixes, with no missing values.
- **Inputs (8):**
  - cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate and fine aggregate (all in kg/m³);
  - curing age (days).
- **Target:** compressive strength (MPa), ranging from 2.33 to 82.60 with a mean of 35.82.

Each work folder includes a copy of the dataset (`concrete+compressive+strength.zip`).

---

## Work 1 — Hybrid Stacking Ensemble with Domain-Informed Feature Engineering

**Md. Siam Ansary.** *A hybrid stacking ensemble framework with domain-informed feature engineering for accurate prediction of concrete compressive strength.* **Discover Civil Engineering** 3, 272 (2026). DOI: [10.1007/s44290-026-00667-9](https://doi.org/10.1007/s44290-026-00667-9)

**Method**
- **Domain features:** six features grounded in mix-design practice are added: binder content (B), W/B, W/C, SP/B, the coarse-to-fine aggregate ratio and log(1 + Age). This takes the input from 8 to 14 features.
- **Interactions:** second-order pairwise interaction features are added.
- **Base learners:** XGBoost, LightGBM and CatBoost, each tuned with **Optuna** (TPE sampler, 40 trials, 5-fold CV).
- **Stacking:** an **ElasticNet meta-learner** is trained only on out-of-fold predictions. All preprocessing is fitted inside the training folds, so no information leaks from validation or test data.
- **Interpretability:** SHAP is used to explain the full ensemble.

**Key results (held-out test set)**

| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| Best single model (CatBoost, default) | 3.024 | 4.667 | 0.917 |
| **Proposed stacking ensemble** | **2.695** | **4.071** | **0.936** |

- **Comparison:** the ensemble beats Linear Regression, SVR, Random Forest, Extra Trees, MLP and the default boosting models. It also beats five published methods evaluated on the same dataset.
- **SHAP ranking:** the strongest predictors are cement (6.92 MPa mean |SHAP|), log curing age (4.85) and the derived **W/B ratio** (2.87).
- **Ablation:** stacking is the most important component. Polynomial interaction features hurt a single XGBoost model (R² drops to 0.885).

📁 Folder: [`A hybrid stacking ensemble framework with domain-informed feature engineering for accurate prediction of concrete compressive strength`](https://github.com/MdSiamAnsary/Prediction-of-Concrete-Compressive-Strength/tree/main/A%20hybrid%20stacking%20ensemble%20framework%20with%20domain-informed%20feature%20engineering%20for%20accurate%20prediction%20of%20concrete%20compressive%20strength)

---

## Work 2 — Physics-Informed and Uncertainty-Aware Stacking Ensemble

**Md. Siam Ansary.** *Physics-Informed and Uncertainty-Aware Stacking Ensemble for Robust Prediction of Concrete Compressive Strength.* **Shock and Vibration** 2026, 3817773. DOI: [10.1155/vib/3817773](https://doi.org/10.1155/vib/3817773)

**Method**
- **Features:** the same domain-informed features as Work 1. Polynomial features are left out on purpose because they hurt tree-based models.
- **Physics-informed constraints:** strength is forced to be non-increasing in **w/c** and non-decreasing in **curing age**.
  - XGBoost and LightGBM enforce this natively.
  - CatBoost has no such constraint here. Explicit ratio features and strong L2 regularization keep it approximately monotone, with compliance above 97 %.
- **Stacking:** 10-fold **out-of-fold stacking** on the training set only, with an ElasticNet meta-learner. Hyperparameters are found by Bayesian optimization.
- **Uncertainty:** a parallel **quantile-regression branch** (τ = 0.1, 0.5, 0.9) runs on the stacked meta-features and produces prediction intervals alongside the point predictions.

**Key results (held-out test set)**

| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| CatBoost (baseline) | 2.812 | 4.302 | 0.928 |
| **Proposed physics-informed stacking** | **2.601** | **3.842** | **0.945** |

| Uncertainty metric | Value |
|---|---|
| PICP (interval coverage) | 0.914 |
| MPIW (mean interval width) | 5.12 MPa |

- **Significance:** the improvement over CatBoost is statistically significant (paired t-test, p < 0.01).
- **Ablation:**
  - Monotonicity constraints raise R² from 0.920 to 0.932.
  - Out-of-fold stacking gives the largest single gain, to 0.945.
  - Uncertainty modeling adds confidence bounds without changing the point accuracy.

📁 Folder: [`Physics-Informed and Uncertainty-Aware Stacking Ensemble for Robust Prediction of Concrete Compressive Strength`](https://github.com/MdSiamAnsary/Prediction-of-Concrete-Compressive-Strength/tree/main/Physics-Informed%20and%20Uncertainty-Aware%20Stacking%20Ensemble%20for%20Robust%20Prediction%20of%20Concrete%20Compressive%20Strength)

---

## The Two Works at a Glance

| | Work 1 (Discover Civil Engineering) | Work 2 (Shock and Vibration) |
|---|---|---|
| Main goal | Accuracy and interpretability | Reliability and physical consistency |
| Base learners | XGBoost, LightGBM, CatBoost | XGBoost, LightGBM, CatBoost (monotone-constrained) |
| Meta-learner | ElasticNet on out-of-fold predictions | ElasticNet on out-of-fold predictions |
| Feature engineering | Domain features plus polynomial interactions | Domain features only |
| Tuning | Optuna (TPE, 40 trials, 5-fold CV) | Bayesian optimization (10-fold CV) |
| Explainability | SHAP | Monotonicity constraints and compliance check |
| Uncertainty | — | Quantile regression (PICP, MPIW) |
| Test R² / RMSE | 0.936 / 4.071 MPa | 0.945 / 3.842 MPa |

---

## Repository Structure

```
Prediction of Concrete Compressive Strength/
├── README.md                                   ← this file
├── A hybrid stacking ensemble framework with domain-informed feature engineering .../
│   ├── ReadME.md                               ← Python code + reported results
│   ├── s44290-026-00667-9.pdf                  ← published paper
│   └── concrete+compressive+strength.zip       ← UCI dataset
└── Physics-Informed and Uncertainty-Aware Stacking Ensemble .../
    ├── ReadMe.md                               ← Python code + reported results
    ├── Shock and Vibration.pdf                 ← published paper
    └── concrete+compressive+strength.zip       ← UCI dataset
```

Each work's `ReadMe.md` contains the full Python pipeline, section by section, followed by the result tables as reported in the paper.

## Requirements

```bash
pip install numpy pandas matplotlib seaborn scipy xlrd scikit-learn xgboost lightgbm catboost optuna shap
```

Work 1 reports these versions: Python 3.10, scikit-learn 1.4.0, XGBoost 2.0.3, LightGBM 4.3.0, CatBoost 1.2.2, Optuna 3.5.0 and SHAP 0.44.0. Both works use random seed 42 and an 80:20 train–test split.

---

## Citation

```bibtex
@article{ansary2026hybrid,
  author  = {Ansary, Md. Siam},
  title   = {A hybrid stacking ensemble framework with domain-informed feature engineering
             for accurate prediction of concrete compressive strength},
  journal = {Discover Civil Engineering},
  volume  = {3},
  pages   = {272},
  year    = {2026},
  doi     = {10.1007/s44290-026-00667-9}
}

@article{ansary2026physics,
  author  = {Ansary, Md. Siam},
  title   = {Physics-Informed and Uncertainty-Aware Stacking Ensemble for Robust
             Prediction of Concrete Compressive Strength},
  journal = {Shock and Vibration},
  volume  = {2026},
  pages   = {3817773},
  year    = {2026},
  doi     = {10.1155/vib/3817773}
}
```

## Contact

**Md. Siam Ansary** — Assistant Professor, Department of Computer Science and Engineering, Ahsanullah University of Science and Technology, Dhaka, Bangladesh
✉️ siamansary.cse@aust.edu · ORCID: [0000-0002-4942-2541](https://orcid.org/0000-0002-4942-2541)
