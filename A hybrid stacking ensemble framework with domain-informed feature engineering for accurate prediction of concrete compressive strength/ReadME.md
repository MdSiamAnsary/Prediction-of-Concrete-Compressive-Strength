# A Hybrid Stacking Ensemble Framework with Domain-Informed Feature Engineering for Accurate Prediction of Concrete Compressive Strength

**Author:** Md. Siam Ansary
**Journal:** *Discover Civil Engineering* (2026) 3:272
**DOI:** https://doi.org/10.1007/s44290-026-00667-9
**Dataset:** UCI Concrete Compressive Strength — https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength (file: `Concrete_Data.xls`, 1030 samples, 8 inputs + 1 target)

---

## Part A — Python Code

### A.0 Environment (versions as reported in the paper, Sect. 10)

```bash
pip install numpy pandas matplotlib seaborn xlrd \
    scikit-learn==1.4.0 xgboost==2.0.3 lightgbm==4.3.0 catboost==1.2.2 \
    optuna==3.5.0 shap==0.44.0
```

Python 3.10, random seed 42.

### A.1 Imports and configuration

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import optuna
import shap

SEED = 42
np.random.seed(SEED)

DATA_PATH = "Concrete_Data.xls"   # UCI file

TEST_SIZE   = 0.20   # 80:20 split (Sect. 3.4)
N_TRIALS    = 40     # Optuna trials per base learner (Sect. 4.4)
HPO_FOLDS   = 5      # 5-fold CV inside each Optuna trial (Sect. 4.4)
STACK_FOLDS = 5      # K-fold out-of-fold stacking (Sect. 4.5)

# Set to False to skip Optuna and use the optimized values reported in Tables 3-5
RUN_OPTUNA = True
```

### A.2 Load the dataset (Table 1 nomenclature)

```python
RAW_COLS = ["C", "BFS", "FA", "W", "SP", "CoAgg", "FiAgg", "Age"]
TARGET   = "CS"

df = pd.read_excel(DATA_PATH)
df.columns = RAW_COLS + [TARGET]          # columns are in the UCI order

print("Shape:", df.shape)                 # (1030, 9)
print("Missing values:", int(df.isnull().sum().sum()))

# Table 2 - descriptive statistics
desc = df.describe().T[["mean", "std", "min", "max"]].round(2)
desc.columns = ["Mean", "Std Dev", "Min", "Max"]
print(desc)
```

### A.3 Exploratory data analysis (Fig. 1)

```python
fig, axes = plt.subplots(2, 1, figsize=(9, 13))

sns.heatmap(df.corr(method="pearson"), annot=True, fmt=".2f", cmap="coolwarm",
            square=True, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title("(a) Pearson correlation heatmap")

sns.histplot(df[TARGET], bins=30, kde=True, ax=axes[1])
axes[1].set_xlabel("CS (MPa)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("(b) Distribution of concrete compressive strength")

plt.tight_layout()
plt.savefig("fig1_eda.png", dpi=300)
plt.show()
```

### A.4 Domain-derived features (Sect. 3.3, Eqs. 1–6)

```python
def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, parameter-free transformations -> no information can leak across rows."""
    X = X.copy()
    X["B"]           = X["C"] + X["BFS"] + X["FA"]   # Eq. (1) binder content
    X["W/B"]         = X["W"] / X["B"]               # Eq. (2) water-to-binder ratio
    X["W/C"]         = X["W"] / X["C"]               # Eq. (3) water-to-cement ratio
    X["SP/B"]        = X["SP"] / X["B"]              # Eq. (4) superplasticizer-to-binder ratio
    X["CoAgg/FiAgg"] = X["CoAgg"] / X["FiAgg"]       # Eq. (5) aggregate ratio
    X["Age_log"]     = np.log1p(X["Age"])            # Eq. (6) log(1 + Age)
    return X

FE_COLS = RAW_COLS + ["B", "W/B", "W/C", "SP/B", "CoAgg/FiAgg", "Age_log"]   # 8 -> 14 features
```

### A.5 Train/test split (Sect. 3.4)

```python
X_raw = df[RAW_COLS]
y     = df[TARGET]

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_raw, y, test_size=TEST_SIZE, random_state=SEED
)
X_tr_raw, X_te_raw = X_tr_raw.reset_index(drop=True), X_te_raw.reset_index(drop=True)
y_tr, y_te         = y_tr.reset_index(drop=True), y_te.reset_index(drop=True)

X_tr_fe = add_domain_features(X_tr_raw)[FE_COLS]
X_te_fe = add_domain_features(X_te_raw)[FE_COLS]

print("Train:", X_tr_fe.shape, " Test:", X_te_fe.shape)
```

### A.6 Leakage-free preprocessing pipeline + metrics

Every fitted transformation (polynomial expansion, Z-score scaling) lives inside a scikit-learn `Pipeline`, so it is re-fitted on the training part of every CV fold and never sees validation/test rows.

```python
def make_pipeline(model, pif: bool = False) -> Pipeline:
    steps = []
    if pif:
        # Sect. 4.2, Eq. (7): pairwise products x_i * x_j (i < j); pure squares x_i^2 excluded
        steps.append(("pif", PolynomialFeatures(degree=2, interaction_only=True,
                                                include_bias=False)))
    steps.append(("scaler", StandardScaler()))      # Z-score normalisation
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE (MPa)":  mean_absolute_error(y_true, y_pred),
        "RMSE (MPa)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":         r2_score(y_true, y_pred),
    }
```

### A.7 Base learners — fixed settings and paper-reported optimized values (Tables 3–5)

```python
XGB_FIXED = dict(objective="reg:squarederror", tree_method="hist",
                 random_state=SEED, n_jobs=-1)
LGB_FIXED = dict(boosting_type="gbdt", random_state=SEED, n_jobs=-1, verbose=-1)
CAT_FIXED = dict(loss_function="RMSE", random_seed=SEED, verbose=0, thread_count=-1)

PAPER_PARAMS = {
    "XGB": dict(n_estimators=1867, max_depth=4, learning_rate=0.0456, subsample=0.8349,
                colsample_bytree=0.9749, gamma=2.3096, reg_alpha=0.1526, reg_lambda=1.5575),
    "LGB": dict(n_estimators=2337, max_depth=9, num_leaves=168, learning_rate=0.0136,
                subsample=0.9398, colsample_bytree=0.9436),
    "CAT": dict(iterations=2080, depth=4, learning_rate=0.0418, l2_leaf_reg=2.5849),
}
```

### A.8 Automated hyperparameter optimization with Optuna (Sect. 4.4)

TPE sampler, 40 trials, objective = mean 5-fold CV R² on the training set, run independently for each base learner before stacking.

```python
def cv_r2(model, X, y, pif: bool) -> float:
    kf = KFold(n_splits=HPO_FOLDS, shuffle=True, random_state=SEED)
    return cross_val_score(make_pipeline(model, pif=pif), X, y,
                           cv=kf, scoring="r2", n_jobs=1).mean()


def xgb_objective(trial, X, y, pif):                       # Table 3 search space
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 500, 3000),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        gamma            = trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    )
    return cv_r2(XGBRegressor(**params, **XGB_FIXED), X, y, pif)


def lgb_objective(trial, X, y, pif):                       # Table 4 search space
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators", 500, 4000),
        max_depth        = trial.suggest_int("max_depth", 3, 12),
        num_leaves       = trial.suggest_int("num_leaves", 31, 255),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
    )
    return cv_r2(LGBMRegressor(**params, **LGB_FIXED), X, y, pif)


def cat_objective(trial, X, y, pif):                       # Table 5 search space
    params = dict(
        iterations    = trial.suggest_int("iterations", 500, 3000),
        depth         = trial.suggest_int("depth", 3, 8),
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    )
    return cv_r2(CatBoostRegressor(**params, **CAT_FIXED), X, y, pif)


def run_study(objective, X, y, pif, name):
    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(lambda t: objective(t, X, y, pif), n_trials=N_TRIALS,
                   show_progress_bar=True)
    print(f"[{name}] best CV R2 = {study.best_value:.4f}")
    print(f"[{name}] best params = {study.best_params}")
    return study.best_params


optuna.logging.set_verbosity(optuna.logging.WARNING)

if RUN_OPTUNA:
    # Base learners of the stack: domain features + polynomial interaction features
    best_params = {
        "XGB": run_study(xgb_objective, X_tr_fe, y_tr, pif=True, name="XGB"),
        "LGB": run_study(lgb_objective, X_tr_fe, y_tr, pif=True, name="LGB"),
        "CAT": run_study(cat_objective, X_tr_fe, y_tr, pif=True, name="CAT"),
    }
    # Ablation A4 (A2 + HPO): single XGBoost tuned on domain features without PIF
    best_xgb_A4 = run_study(xgb_objective, X_tr_fe, y_tr, pif=False, name="XGB_A4")
else:
    best_params = PAPER_PARAMS
    best_xgb_A4 = PAPER_PARAMS["XGB"]
```

### A.9 Leakage-free out-of-fold stacking (Sect. 4.5, Eqs. 10–12)

```python
class OOFStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Meta-learner is trained ONLY on out-of-fold predictions of the base learners.
    Each base learner is a full Pipeline (PIF -> scaler -> model), so all fitted
    preprocessing is redone inside every fold.
    """

    def __init__(self, base_estimators, meta_estimator, n_folds=5, random_state=SEED):
        self.base_estimators = base_estimators     # list of (name, estimator)
        self.meta_estimator  = meta_estimator
        self.n_folds         = n_folds
        self.random_state    = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Z_train: complete out-of-fold prediction matrix
        self.oof_ = np.zeros((len(X), len(self.base_estimators)))
        for j, (name, est) in enumerate(self.base_estimators):
            for tr_idx, va_idx in kf.split(X):
                m = clone(est).fit(X.iloc[tr_idx], y.iloc[tr_idx])
                self.oof_[va_idx, j] = m.predict(X.iloc[va_idx])

        # Base learners re-trained on the full training set -> used to build Z_test
        self.fitted_base_ = [(name, clone(est).fit(X, y)) for name, est in self.base_estimators]

        # Meta-learner g(Z) trained on out-of-fold predictions only
        self.meta_ = clone(self.meta_estimator).fit(self.oof_, y)
        return self

    def meta_features(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([m.predict(X) for _, m in self.fitted_base_])

    def predict(self, X):
        return self.meta_.predict(self.meta_features(X))


base_learners = [
    ("XGB", make_pipeline(XGBRegressor(**best_params["XGB"], **XGB_FIXED), pif=True)),
    ("LGB", make_pipeline(LGBMRegressor(**best_params["LGB"], **LGB_FIXED), pif=True)),
    ("CAT", make_pipeline(CatBoostRegressor(**best_params["CAT"], **CAT_FIXED), pif=True)),
]

# Table 6 - ElasticNet meta-learner
meta_learner = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=1000, selection="cyclic",
                          random_state=SEED)

stack = OOFStackingRegressor(base_learners, meta_learner, n_folds=STACK_FOLDS)
stack.fit(X_tr_fe, y_tr)

y_pred_stack = stack.predict(X_te_fe)
stack_metrics = evaluate(y_te, y_pred_stack)

print("Table 8 - Proposed stacking ensemble (test set)")
for k, v in stack_metrics.items():
    print(f"  {k:<11}: {v:.3f}")

print("Meta-learner weights:", dict(zip([n for n, _ in base_learners],
                                         np.round(stack.meta_.coef_, 4))),
      " intercept:", round(stack.meta_.intercept_, 4))
```

### A.10 Individual algorithm comparison (Sect. 5.1, Table 7)

Default hyperparameters, same (domain-feature) input set, same test set.

```python
comparison_models = {
    "Linear regression":               LinearRegression(),
    "Support Vector Regression (SVR)": SVR(),
    "Random Forest (RF)":              RandomForestRegressor(random_state=SEED, n_jobs=-1),
    "Extra Trees (ET)":                ExtraTreesRegressor(random_state=SEED, n_jobs=-1),
    "Multilayer Perceptron (MLP)":     MLPRegressor(max_iter=1000, random_state=SEED),
    "XGBoost (default)":               XGBRegressor(**XGB_FIXED),
    "LightGBM (default)":              LGBMRegressor(**LGB_FIXED),
    "CatBoost (default)":              CatBoostRegressor(**CAT_FIXED),
}

rows = []
for name, model in comparison_models.items():
    pipe = make_pipeline(model, pif=False).fit(X_tr_fe, y_tr)
    rows.append({"Model": name, **evaluate(y_te, pipe.predict(X_te_fe))})
rows.append({"Model": "Proposed stacking ensemble", **stack_metrics})

table7 = pd.DataFrame(rows).set_index("Model").round(3)
print(table7)
```

### A.11 Parity and residual analysis (Sect. 5.3, Fig. 3)

```python
residuals = y_te - y_pred_stack
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

lims = [min(y_te.min(), y_pred_stack.min()) - 2, max(y_te.max(), y_pred_stack.max()) + 2]
axes[0].scatter(y_te, y_pred_stack, alpha=0.7, edgecolor="k", linewidth=0.4)
axes[0].plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_xlabel("Actual CS (MPa)"); axes[0].set_ylabel("Predicted CS (MPa)")
axes[0].set_title(f"(a) Parity plot  (R$^2$ = {stack_metrics['R2']:.3f})")
axes[0].legend()

axes[1].scatter(y_pred_stack, residuals, alpha=0.7, edgecolor="k", linewidth=0.4)
axes[1].axhline(0, color="r", ls="--", lw=1.5)
axes[1].set_xlabel("Predicted CS (MPa)"); axes[1].set_ylabel("Residual (MPa)")
axes[1].set_title("(b) Residual plot")

plt.tight_layout()
plt.savefig("fig3_parity_residual.png", dpi=300)
plt.show()
```

### A.12 SHAP-based interpretability of the full stacking ensemble (Sect. 5.5, Fig. 4)

The stack is treated as a black box over the 14 domain-level features (the polynomial expansion happens inside each base pipeline).

```python
def stack_predict(arr):
    return stack.predict(pd.DataFrame(arr, columns=FE_COLS))

background = shap.sample(X_tr_fe, 100, random_state=SEED)
explainer  = shap.Explainer(stack_predict, shap.maskers.Independent(background),
                            feature_names=FE_COLS, seed=SEED)
shap_values = explainer(X_te_fe)

mean_abs_shap = (pd.Series(np.abs(shap_values.values).mean(axis=0), index=FE_COLS)
                 .sort_values(ascending=False))
print("Mean |SHAP| (MPa):")
print(mean_abs_shap.round(2))

plt.figure(figsize=(8, 6))
mean_abs_shap[::-1].plot(kind="barh", color="steelblue", edgecolor="k")
plt.xlabel("Mean |SHAP value| (MPa)")
plt.title("SHAP feature importance - stacking ensemble")
plt.tight_layout()
plt.savefig("fig4_shap_importance.png", dpi=300)
plt.show()

shap.plots.beeswarm(shap_values, max_display=14, show=False)
plt.tight_layout()
plt.savefig("fig4b_shap_beeswarm.png", dpi=300)
plt.show()
```

### A.13 Ablation study A1–A5 (Sect. 6, Table 10, Fig. 5)

```python
ablation = {}

# A1: Baseline - XGBoost (default) on the 8 raw features
m = make_pipeline(XGBRegressor(**XGB_FIXED), pif=False).fit(X_tr_raw, y_tr)
ablation["A1: Baseline (XGBoost, raw features)"] = evaluate(y_te, m.predict(X_te_raw))

# A2: A1 + Feature Engineering
m = make_pipeline(XGBRegressor(**XGB_FIXED), pif=False).fit(X_tr_fe, y_tr)
ablation["A2: A1 + Feature Engineering (FE)"] = evaluate(y_te, m.predict(X_te_fe))

# A3: A2 + Polynomial Interaction Features (single model)
m = make_pipeline(XGBRegressor(**XGB_FIXED), pif=True).fit(X_tr_fe, y_tr)
ablation["A3: A2 + Polynomial Interaction Features (PIF)"] = evaluate(y_te, m.predict(X_te_fe))

# A4: A2 + Hyperparameter Optimization (Optuna-tuned single XGBoost)
m = make_pipeline(XGBRegressor(**best_xgb_A4, **XGB_FIXED), pif=False).fit(X_tr_fe, y_tr)
ablation["A4: A2 + Hyperparameter Optimization (HPO)"] = evaluate(y_te, m.predict(X_te_fe))

# A5: Proposed full stacking
ablation["A5: Proposed Stacking (full)"] = stack_metrics

table10 = pd.DataFrame(ablation).T.round(3)
print(table10)

# Fig. 5 - separate panels for error metrics and R2
labels = ["A1", "A2", "A3", "A4", "A5"]
fig, axes = plt.subplots(2, 1, figsize=(9, 9))
x, w = np.arange(len(labels)), 0.38
axes[0].bar(x - w/2, table10["MAE (MPa)"],  w, label="MAE",  edgecolor="k")
axes[0].bar(x + w/2, table10["RMSE (MPa)"], w, label="RMSE", edgecolor="k")
axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
axes[0].set_ylabel("Error (MPa)"); axes[0].set_title("Ablation study - error metrics")
axes[0].legend()

axes[1].bar(x, table10["R2"], 0.5, color="seagreen", edgecolor="k")
axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
axes[1].set_ylim(0.85, 0.95); axes[1].set_ylabel("R$^2$")
axes[1].set_title("Ablation study - R$^2$")
for i, v in enumerate(table10["R2"]):
    axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center")

plt.tight_layout()
plt.savefig("fig5_ablation.png", dpi=300)
plt.show()
```

---

## Part B — Results Reported in the Paper

### B.1 Dataset descriptive statistics (Table 2)

| Variable | Mean | Std Dev | Min | Max |
|---|---|---|---|---|
| Cement (C) | 281.17 | 104.51 | 102.00 | 540.00 |
| Blast furnace slag (BFS) | 73.90 | 86.28 | 0.00 | 359.40 |
| Fly ash (FA) | 54.19 | 63.99 | 0.00 | 200.10 |
| Water (W) | 181.57 | 21.36 | 121.75 | 247.00 |
| Superplasticizer (SP) | 6.20 | 5.97 | 0.00 | 32.20 |
| Coarse aggregate (CoAgg) | 972.92 | 77.75 | 801.00 | 1145.00 |
| Fine aggregate (FiAgg) | 773.58 | 80.18 | 594.00 | 992.60 |
| Curing age (Age, days) | 45.66 | 63.17 | 1.00 | 365.00 |
| Compressive strength (CS, MPa) | 35.82 | 16.71 | 2.33 | 82.60 |

### B.2 Optimized hyperparameters (Tables 3–6)

**XGBoost (Table 3)**

| Parameter | Search range | Optimized value |
|---|---|---|
| n_estimators | [500, 3000] | 1867 |
| max_depth | [3, 8] | 4 |
| learning_rate | [0.005, 0.3] | 0.0456 |
| subsample | [0.6, 1.0] | 0.8349 |
| colsample_bytree | [0.6, 1.0] | 0.9749 |
| gamma | [0, 5] | 2.3096 |
| reg_alpha (L1) | [0.001, 1.0] | 0.1526 |
| reg_lambda (L2) | [0.001, 5.0] | 1.5575 |
| Objective | — | reg:squarederror |

**LightGBM (Table 4)**

| Parameter | Search range | Optimized value |
|---|---|---|
| n_estimators | [500, 4000] | 2337 |
| max_depth | [3, 12] | 9 |
| num_leaves | [31, 255] | 168 |
| learning_rate | [0.005, 0.3] | 0.0136 |
| subsample | [0.6, 1.0] | 0.9398 |
| colsample_bytree | [0.6, 1.0] | 0.9436 |
| boosting_type | — | gbdt |

**CatBoost (Table 5)**

| Parameter | Search range | Optimized value |
|---|---|---|
| iterations | [500, 3000] | 2080 |
| depth | [3, 8] | 4 |
| learning_rate | [0.005, 0.3] | 0.0418 |
| l2_leaf_reg | [1.0, 10.0] | 2.5849 |
| loss_function | — | RMSE |

**ElasticNet meta-learner (Table 6)**

| Parameter | Value |
|---|---|
| Regularization α | 0.001 |
| L1 ratio λ | 0.5 |
| Max iterations | 1000 |
| Solver | Cyclic coordinate descent |

### B.3 Individual algorithm comparison (Table 7, default hyperparameters, same feature set)

| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| Linear regression | 5.231 | 6.874 | 0.831 |
| Support Vector Regression (SVR) | 4.188 | 5.512 | 0.891 |
| Random Forest (RF) | 3.412 | 4.983 | 0.911 |
| Extra Trees (ET) | 3.218 | 4.721 | 0.918 |
| Multilayer Perceptron (MLP) | 3.654 | 5.102 | 0.907 |
| XGBoost (default) | 3.105 | 4.693 | 0.916 |
| LightGBM (default) | 3.076 | 4.701 | 0.915 |
| CatBoost (default) | 3.024 | 4.667 | 0.917 |
| **Proposed stacking ensemble** | **2.695** | **4.071** | **0.936** |

### B.4 Proposed stacking ensemble on the test set (Table 8)

| Metric | Value |
|---|---|
| MAE | 2.695 MPa |
| RMSE | 4.071 MPa |
| R² | 0.936 |

R² = 0.936 means about 93.6 % of the variance is explained; the MAE is below 7.5 % of the mean target (35.82 MPa). The stack improves on the best single model (default CatBoost) by about 0.6 MPa RMSE and 2 percentage points R².

### B.5 Comparison with state-of-the-art on the UCI dataset (Table 9)

| Study | Method | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|---|
| Pan et al. [13] | Stacking (XGB+ET+DT) | 3.12 | 4.87 | 0.921 |
| Sun et al. [17] | WOA-XGBoost | 3.05 | 4.51 | 0.928 |
| Zhang et al. [1] | Ensemble RF+XGB | 3.41 | 5.10 | 0.912 |
| Elshaarawy et al. [19] | Hybrid GBMs | 2.98 | 4.38 | 0.929 |
| Naved et al. [15] | DNN (Keras Tuner) | 3.35 | 5.21 | 0.909 |
| **Proposed** | **Stacking (XGB+LGB+CAT)** | **2.695** | **4.071** | **0.936** |

### B.6 SHAP feature importance (Sect. 5.5, Fig. 4)

| Rank | Feature | Mean \|SHAP\| (MPa) |
|---|---|---|
| 1 | Cement (C) | 6.92 |
| 2 | Log-transformed curing age (Age_log) | 4.85 |
| 3 | Water-to-binder ratio (W/B), a derived feature | 2.87 |

### B.7 Ablation study (Table 10)

| Configuration | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| A1: Baseline (XGBoost, raw features) | 2.996 | 4.606 | 0.918 |
| A2: A1 + Feature Engineering (FE) | 2.946 | 4.586 | 0.918 |
| A3: A2 + Polynomial Interaction Features (PIF) | 3.439 | 5.441 | 0.885 |
| A4: A2 + Hyperparameter Optimization (HPO) | 3.019 | 4.733 | 0.913 |
| **A5: Proposed Stacking (full)** | **2.695** | **4.071** | **0.936** |

### B.8 Computational details (Sect. 5.6 / Sect. 10)

- All Optuna trials took about 45 minutes on an Intel Core i7 with 16 GB RAM; inference is under 1 ms per sample.
- Software: Python 3.10, scikit-learn 1.4.0, XGBoost 2.0.3, LightGBM 4.3.0, CatBoost 1.2.2, Optuna 3.5.0, SHAP 0.44.0. Random seed 42.

---

### Implementation notes

- **Stacking folds:** the value of K in Sect. 4.5 is garbled in the PDF text layer. The code uses K = 5 to match the 5-fold CV used throughout the paper. Change `STACK_FOLDS` if the published value differs.
- **Where PIF are applied:** the code follows the Fig. 2 order (FE → PIF → HPO → stacking), so the three base learners are tuned and trained on the 14 domain features plus their 91 pairwise interaction terms.
- **Table 7 feature set:** the code uses the 14 domain features without PIF. The default-XGBoost row (R² 0.916) is close to A2 (0.918) and far from A3 (0.885), which suggests this is the setting used.
- **MLP:** `max_iter=1000` is set so that the default MLP converges. All other settings are scikit-learn defaults.
