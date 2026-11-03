# Physics-Informed and Uncertainty-Aware Stacking Ensemble for Robust Prediction of Concrete Compressive Strength

**Author:** Md. Siam Ansary
**Journal:** *Shock and Vibration*, 2026, Article ID 3817773
**DOI:** https://doi.org/10.1155/vib/3817773
**Dataset:** UCI Concrete Compressive Strength — https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength (file: `Concrete_Data.xls`, 1030 samples, 8 inputs + 1 target)

---

## Part A — Python Code

### A.0 Environment

```bash
pip install numpy pandas matplotlib seaborn scipy xlrd \
    scikit-learn xgboost lightgbm catboost optuna
```

Seed 42. The paper reports an Intel Core i5 machine with 16 GB RAM and no GPU.

### A.1 Imports and configuration

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_pinball_loss)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import optuna

SEED = 42
np.random.seed(SEED)

DATA_PATH = "Concrete_Data.xls"

TEST_SIZE = 0.20      # 80:20 train-test split (Sect. 5.1)
CV_FOLDS  = 10        # 10-fold CV on the training set only (Sect. 5.1) - used for OOF and HPO
N_TRIALS  = 40        # Bayesian-optimisation trials per model (not stated in the paper)
TAUS      = [0.1, 0.5, 0.9]   # quantile levels (Sect. 4.6, Eq. 11)
```

### A.2 Load the dataset (paper nomenclature)

Here `FA` means **fine aggregate** and `F` means fly ash, as in the paper's nomenclature.

```python
RAW_COLS = ["C", "S", "F", "W", "SP", "CA", "FA", "Age"]   # Eq. (1)
TARGET   = "CS"

df = pd.read_excel(DATA_PATH)
df.columns = RAW_COLS + [TARGET]

print("Shape:", df.shape)                                  # (1030, 9)
print("Missing values:", int(df.isnull().sum().sum()))
print(df.describe().T[["mean", "std", "min", "max"]].round(2))
```

### A.3 Domain-informed feature engineering (Sect. 4.2, Eqs. 2–5)

```python
def add_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise, parameter-free -> leakage-free by construction."""
    X = X.copy()
    X["B"]       = X["C"] + X["S"] + X["F"]     # Eq. (2) binder content
    X["wb"]      = X["W"] / X["B"]              # Eq. (3) water-to-binder ratio
    X["wc"]      = X["W"] / X["C"]              # Eq. (3) water-to-cement ratio
    X["sp_b"]    = X["SP"] / X["B"]             # superplasticizer ratio (listed in Sect. 3)
    X["ragg"]    = X["CA"] / X["FA"]            # Eq. (4) aggregate ratio CA/FA
    X["Age_log"] = np.log1p(X["Age"])           # Eq. (5) log(1 + Age)
    return X

FE_COLS = RAW_COLS + ["B", "wb", "wc", "sp_b", "ragg", "Age_log"]
# Design decision (Sect. 4.2): polynomial interaction features are NOT used in the final model.
```

### A.4 Train/test split

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
```

### A.5 Pipelines and metrics

All features are standardized (Sect. 5.1). The scaler sits inside each pipeline, so it is re-fitted within every CV fold. Standardization is a positive affine map, so it does not change the direction of any monotone constraint.

```python
def make_pipeline(model, poly: bool = False) -> Pipeline:
    steps = []
    if poly:   # used only in the ablation "+ Polynomial features"
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def evaluate(y_true, y_pred) -> dict:
    return {
        "MAE (MPa)":  mean_absolute_error(y_true, y_pred),
        "RMSE (MPa)": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":         r2_score(y_true, y_pred),
    }
```

### A.6 Physics-informed monotonicity constraints (Sect. 4.3, Eq. 6)

∂ŷ/∂(w/c) < 0 and ∂ŷ/∂Age > 0, enforced natively in XGBoost and LightGBM.

```python
MONOTONE_RULES = {"wc": -1, "Age": +1, "Age_log": +1}   # Age_log is a monotone transform of Age

def monotone_vector(columns):
    return [MONOTONE_RULES.get(c, 0) for c in columns]

MONO_FE = monotone_vector(FE_COLS)
print(dict(zip(FE_COLS, MONO_FE)))

XGB_FIXED = dict(objective="reg:squarederror", tree_method="hist",
                 random_state=SEED, n_jobs=-1)
LGB_FIXED = dict(boosting_type="gbdt", random_state=SEED, n_jobs=-1, verbose=-1)
CAT_FIXED = dict(loss_function="RMSE", random_seed=SEED, verbose=0, thread_count=-1)

def xgb_constraints(mono):   # XGBoost expects a tuple-string, e.g. "(0,0,-1,1)"
    return "(" + ",".join(str(v) for v in mono) + ")"
```

### A.7 Bayesian hyperparameter optimization (Sect. 4.7, Eq. 14)

θ* = argmax E_CV[R²(θ)], using Optuna with the TPE sampler. The search ranges are not reported in this paper; the ones below are standard ranges for these learners.

```python
def cv_r2(model, X, y, poly=False) -> float:
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    return cross_val_score(make_pipeline(model, poly=poly), X, y,
                           cv=kf, scoring="r2", n_jobs=1).mean()


def xgb_space(trial):
    return dict(
        n_estimators     = trial.suggest_int("n_estimators", 300, 3000),
        max_depth        = trial.suggest_int("max_depth", 3, 8),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        gamma            = trial.suggest_float("gamma", 0.0, 5.0),
        reg_alpha        = trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    )


def lgb_space(trial):
    return dict(
        n_estimators     = trial.suggest_int("n_estimators", 300, 4000),
        max_depth        = trial.suggest_int("max_depth", 3, 12),
        num_leaves       = trial.suggest_int("num_leaves", 15, 255),
        learning_rate    = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        subsample        = trial.suggest_float("subsample", 0.6, 1.0),
        subsample_freq   = 1,
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_samples= trial.suggest_int("min_child_samples", 5, 50),
    )


def cat_space(trial):
    return dict(
        iterations    = trial.suggest_int("iterations", 300, 3000),
        depth         = trial.suggest_int("depth", 3, 8),
        learning_rate = trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        l2_leaf_reg   = trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),  # strong L2 (Sect. 4.3)
    )


def build_xgb(p, mono=None):
    extra = {"monotone_constraints": xgb_constraints(mono)} if mono is not None else {}
    return XGBRegressor(**p, **XGB_FIXED, **extra)

def build_lgb(p, mono=None):
    extra = {"monotone_constraints": mono} if mono is not None else {}
    return LGBMRegressor(**p, **LGB_FIXED, **extra)

def build_cat(p, mono=None):          # no monotone API used for CatBoost (Sect. 4.3)
    return CatBoostRegressor(**p, **CAT_FIXED)


def tune(space_fn, builder, X, y, mono=None, name=""):
    def objective(trial):
        return cv_r2(builder(space_fn(trial), mono), X, y)
    study = optuna.create_study(direction="maximize", study_name=name,
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print(f"[{name}] best CV R2 = {study.best_value:.4f} | {study.best_params}")
    params = study.best_params
    if builder is build_lgb:
        params["subsample_freq"] = 1
    return params


optuna.logging.set_verbosity(optuna.logging.WARNING)

# Physics-constrained base learners
p_xgb = tune(xgb_space, build_xgb, X_tr_fe, y_tr, mono=MONO_FE, name="XGB-mono")
p_lgb = tune(lgb_space, build_lgb, X_tr_fe, y_tr, mono=MONO_FE, name="LGB-mono")
p_cat = tune(cat_space, build_cat, X_tr_fe, y_tr, mono=None,    name="CAT")

# Unconstrained XGBoost - needed for the ablation step "+ Hyperparameter optimization"
p_xgb_free = tune(xgb_space, build_xgb, X_tr_fe, y_tr, mono=None, name="XGB-free")
```

### A.8 Base learners

```python
base_learners = [
    ("XGB", make_pipeline(build_xgb(p_xgb, MONO_FE))),
    ("LGB", make_pipeline(build_lgb(p_lgb, MONO_FE))),
    ("CAT", make_pipeline(build_cat(p_cat))),
]
```

### A.9 Out-of-fold meta-feature generation and stacking (Sects. 4.4–4.5, Eqs. 8–10)

```python
class OOFStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Z_train = out-of-fold predictions of f1, f2, f3 (K-fold on the training set only);
    Z_test  = predictions of base learners re-fitted on the full training set;
    y_hat   = g(Z), with g = ElasticNet.
    """

    def __init__(self, base_estimators, meta_estimator, n_folds=10, random_state=SEED):
        self.base_estimators = base_estimators
        self.meta_estimator  = meta_estimator
        self.n_folds         = n_folds
        self.random_state    = random_state

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        self.oof_ = np.zeros((len(X), len(self.base_estimators)))
        for j, (_, est) in enumerate(self.base_estimators):
            for tr_idx, va_idx in kf.split(X):
                m = clone(est).fit(X.iloc[tr_idx], y.iloc[tr_idx])
                self.oof_[va_idx, j] = m.predict(X.iloc[va_idx])

        self.fitted_base_ = [(n, clone(e).fit(X, y)) for n, e in self.base_estimators]
        self.meta_ = clone(self.meta_estimator).fit(self.oof_, y)
        return self

    def meta_features(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([m.predict(X) for _, m in self.fitted_base_])

    def predict(self, X):
        return self.meta_.predict(self.meta_features(X))


# ElasticNet meta-learner on standardized meta-features (Eq. 10).
# alpha and l1_ratio are chosen by internal CV on the OOF matrix (training data only).
meta_learner = Pipeline([
    ("scaler", StandardScaler()),
    ("enet", ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0],
                          cv=CV_FOLDS, max_iter=10000, random_state=SEED)),
])

stack = OOFStackingRegressor(base_learners, meta_learner, n_folds=CV_FOLDS)
stack.fit(X_tr_fe, y_tr)

y_pred_stack  = stack.predict(X_te_fe)
stack_metrics = evaluate(y_te, y_pred_stack)
print("Proposed model (test):", {k: round(v, 3) for k, v in stack_metrics.items()})

enet = stack.meta_.named_steps["enet"]
print("ElasticNet alpha =", round(enet.alpha_, 5), "| l1_ratio =", enet.l1_ratio_,
      "| weights =", dict(zip(["XGB", "LGB", "CAT"], np.round(enet.coef_, 4))))
```

### A.10 Uncertainty quantification via quantile regression (Sect. 4.6, Eqs. 11–13)

Three gradient-boosting quantile regressors run as a parallel branch on the meta-features Z, not on X. They are trained on the OOF matrix Z_train and evaluated on Z_test.

```python
def pinball_loss(y, y_hat, tau):
    """Eq. (13): L_tau = tau*(y - y_hat) if y >= y_hat else (1 - tau)*(y_hat - y)."""
    diff = np.asarray(y) - np.asarray(y_hat)
    return np.mean(np.where(diff >= 0, tau * diff, (tau - 1) * diff))


Z_tr = stack.oof_                    # unbiased OOF meta-features (training set)
Z_te = stack.meta_features(X_te_fe)  # meta-features for the held-out test set

quantile_models = {
    tau: GradientBoostingRegressor(loss="quantile", alpha=tau, n_estimators=300,
                                   max_depth=3, learning_rate=0.05,
                                   min_samples_leaf=9, random_state=SEED).fit(Z_tr, y_tr)
    for tau in TAUS
}

Q = np.column_stack([quantile_models[t].predict(Z_te) for t in TAUS])
Q = np.sort(Q, axis=1)               # guard against quantile crossing
y_lo, y_med, y_hi = Q[:, 0], Q[:, 1], Q[:, 2]

for i, tau in enumerate(TAUS):
    print(f"Pinball loss tau={tau}: {pinball_loss(y_te, Q[:, i], tau):.3f}  "
          f"(sklearn check: {mean_pinball_loss(y_te, Q[:, i], alpha=tau):.3f})")

# Interval [y_0.1, y_0.9]  (Eq. 12)
PICP = np.mean((y_te.values >= y_lo) & (y_te.values <= y_hi))   # coverage probability
MPIW = np.mean(y_hi - y_lo)                                     # mean interval width
print(f"PICP = {PICP:.3f} | MPIW = {MPIW:.2f} MPa")
print("Median (tau = 0.5) model as alternative point estimate:",
      {k: round(v, 3) for k, v in evaluate(y_te, y_med).items()})
```

### A.11 Monotonicity compliance check (Sect. 4.3)

Each feature is swept across its observed training range while all other features stay at their training medians. Compliance is the share of adjacent prediction pairs that move in the required direction. The check is run for each base learner, with CatBoost as the case of interest.

```python
def monotonic_compliance(model, X_ref, feature, direction, n_points=200, tol=1e-9):
    grid = np.linspace(X_ref[feature].min(), X_ref[feature].max(), n_points)
    probe = pd.DataFrame([X_ref.median()] * n_points).reset_index(drop=True)
    probe[feature] = grid
    pred = model.predict(probe)
    d = np.diff(pred)
    ok = (d <= tol) if direction < 0 else (d >= -tol)
    return ok.mean(), grid, pred


fitted = dict(stack.fitted_base_)
rows = []
for name, model in fitted.items():
    for feat, direction in [("wc", -1), ("Age", +1)]:
        rate, _, _ = monotonic_compliance(model, X_tr_fe, feat, direction)
        rows.append({"Model": name, "Feature": feat, "Direction": direction,
                     "Compliance rate (%)": round(100 * rate, 2)})
print(pd.DataFrame(rows))

# Response curves for w/c (should be non-increasing) and Age (non-decreasing)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, (feat, direction) in zip(axes, [("wc", -1), ("Age", +1)]):
    for name, model in fitted.items():
        _, g, p = monotonic_compliance(model, X_tr_fe, feat, direction)
        ax.plot(g, p, label=name)
    ax.set_xlabel(feat); ax.set_ylabel("Predicted CS (MPa)")
    ax.set_title(f"Partial response to {feat}")
    ax.legend()
plt.tight_layout()
plt.savefig("monotonic_response_curves.png", dpi=300)
plt.show()
```

### A.12 Baseline models (Sect. 5.2, Table 1)

Default settings on the original 8 features, with no extra feature engineering or tuning.

```python
baselines = {
    "Linear regression": LinearRegression(),
    "Random forest":     RandomForestRegressor(random_state=SEED, n_jobs=-1),
    "XGBoost":           XGBRegressor(**XGB_FIXED),
    "LightGBM":          LGBMRegressor(**LGB_FIXED),
    "CatBoost":          CatBoostRegressor(**CAT_FIXED),
}

rows, baseline_preds = [], {}
for name, model in baselines.items():
    pipe = make_pipeline(model).fit(X_tr_raw, y_tr)
    baseline_preds[name] = pipe.predict(X_te_raw)
    rows.append({"Model": name, **evaluate(y_te, baseline_preds[name])})
rows.append({"Model": "Proposed model", **stack_metrics})

table1 = pd.DataFrame(rows).set_index("Model").round(3)
print(table1)
```

### A.13 Statistical significance: paired t-test (Sect. 5.5)

H0: no difference in prediction errors between the proposed model and the best baseline (CatBoost).

```python
abs_err_cat  = np.abs(y_te.values - baseline_preds["CatBoost"])
abs_err_prop = np.abs(y_te.values - y_pred_stack)

t_stat, p_val = stats.ttest_rel(abs_err_cat, abs_err_prop)
print(f"Paired t-test on absolute errors: t = {t_stat:.3f}, p = {p_val:.4g}")
print("Significant at 0.01" if p_val < 0.01 else "Not significant at 0.01")
```

### A.14 Ablation study (Sect. 6, Table 4)

```python
ablation = {}

# Baseline: XGBoost (default) on raw features
m = make_pipeline(XGBRegressor(**XGB_FIXED)).fit(X_tr_raw, y_tr)
ablation["Baseline (XGBoost)"] = evaluate(y_te, m.predict(X_te_raw))

# + Feature engineering
m = make_pipeline(XGBRegressor(**XGB_FIXED)).fit(X_tr_fe, y_tr)
ablation["+ Feature engineering"] = evaluate(y_te, m.predict(X_te_fe))

# + Polynomial features (degree-2 expansion on top of the domain features)
m = make_pipeline(XGBRegressor(**XGB_FIXED), poly=True).fit(X_tr_fe, y_tr)
ablation["+ Polynomial features"] = evaluate(y_te, m.predict(X_te_fe))

# + Hyperparameter optimization (polynomial features dropped, Sect. 4.2)
m = make_pipeline(build_xgb(p_xgb_free)).fit(X_tr_fe, y_tr)
ablation["+ Hyperparameter optimization"] = evaluate(y_te, m.predict(X_te_fe))

# + Physics constraints (monotone XGBoost)
m = make_pipeline(build_xgb(p_xgb, MONO_FE)).fit(X_tr_fe, y_tr)
ablation["+ Physics constraints"] = evaluate(y_te, m.predict(X_te_fe))

# + OOF stacking (proposed core)
ablation["+ OOF stacking"] = stack_metrics

# + Uncertainty modeling (full model) - point predictions unchanged
ablation["+ Uncertainty modeling (full model)"] = stack_metrics

table4 = pd.DataFrame(ablation).T.round(3)
print(table4)

fig, axes = plt.subplots(2, 1, figsize=(10, 9))
x, w = np.arange(len(table4)), 0.38
axes[0].bar(x - w/2, table4["MAE (MPa)"],  w, label="MAE",  edgecolor="k")
axes[0].bar(x + w/2, table4["RMSE (MPa)"], w, label="RMSE", edgecolor="k")
axes[0].set_xticks(x); axes[0].set_xticklabels(table4.index, rotation=25, ha="right")
axes[0].set_ylabel("Error (MPa)"); axes[0].legend(); axes[0].set_title("Ablation - error metrics")
axes[1].bar(x, table4["R2"], 0.5, color="seagreen", edgecolor="k")
axes[1].set_xticks(x); axes[1].set_xticklabels(table4.index, rotation=25, ha="right")
axes[1].set_ylim(0.85, 0.96); axes[1].set_ylabel("R$^2$"); axes[1].set_title("Ablation - R$^2$")
plt.tight_layout()
plt.savefig("ablation.png", dpi=300)
plt.show()
```

### A.15 Parity plot with 80 % prediction intervals

```python
order = np.argsort(y_te.values)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

lims = [y_te.min() - 3, y_te.max() + 3]
axes[0].scatter(y_te, y_pred_stack, alpha=0.7, edgecolor="k", linewidth=0.4)
axes[0].plot(lims, lims, "r--", lw=1.5)
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_xlabel("Actual CS (MPa)"); axes[0].set_ylabel("Predicted CS (MPa)")
axes[0].set_title(f"Parity plot (R$^2$ = {stack_metrics['R2']:.3f})")

idx = np.arange(len(order))
axes[1].fill_between(idx, y_lo[order], y_hi[order], alpha=0.3,
                     label=r"[$\hat y_{0.1}$, $\hat y_{0.9}$]")
axes[1].plot(idx, y_te.values[order], "k.", ms=4, label="Actual")
axes[1].plot(idx, y_pred_stack[order], "r-", lw=1, label="Stacking prediction")
axes[1].set_xlabel("Test samples (sorted by actual CS)"); axes[1].set_ylabel("CS (MPa)")
axes[1].set_title(f"Prediction intervals (PICP = {PICP:.3f}, MPIW = {MPIW:.2f} MPa)")
axes[1].legend()

plt.tight_layout()
plt.savefig("parity_and_intervals.png", dpi=300)
plt.show()
```

---

## Part B — Results Reported in the Paper

### B.1 Comparison with baseline models (Table 1)

| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| Linear regression | 5.214 | 7.102 | 0.812 |
| Random forest | 3.102 | 4.892 | 0.901 |
| XGBoost | 2.996 | 4.606 | 0.918 |
| LightGBM | 2.873 | 4.421 | 0.923 |
| CatBoost | 2.812 | 4.302 | 0.928 |
| **Proposed model** | **2.601** | **3.842** | **0.945** |

The CatBoost row is a standalone reimplementation on the 80:20 split with default settings and no extra feature engineering.

### B.2 Comparison with state-of-the-art on the UCI dataset (Table 2)

| Study | Model type | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|---|
| Zhang et al. [1] | CatBoost | 2.27 | 3.06 | 0.966 |
| Kumar et al. [28] | Optimized LightGBM | 2.446 | 3.371 | 0.961 |
| **Proposed model** | **Physics-informed stacking** | **2.601** | **3.842** | **0.945** |

The paper states that the proposed model does not beat the best published raw accuracy. Its stated contribution is the combination of physically consistent predictions, OOF stacking and uncertainty estimation.

### B.3 Uncertainty estimation (Table 3)

| Metric | Value |
|---|---|
| PICP (90 % interval) | 0.914 |
| MPIW (MPa) | 5.12 |

### B.4 Ablation study (Table 4)

| Model variant | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| Baseline (XGBoost) | 2.996 | 4.606 | 0.918 |
| + Feature engineering | 2.882 | 4.421 | 0.923 |
| + Polynomial features | 3.317 | 5.238 | 0.889 |
| + Hyperparameter optimization | 2.910 | 4.512 | 0.920 |
| + Physics constraints | 2.781 | 4.203 | 0.932 |
| + OOF stacking | 2.601 | 3.842 | 0.945 |
| **+ Uncertainty modeling (full model)** | **2.601** | **3.842** | **0.945** |

Key findings reported:

- Domain features help.
- Polynomial features hurt tree ensembles.
- Hyperparameter optimization gives only a marginal gain.
- Monotone constraints improve both accuracy and interpretability.
- OOF stacking gives the largest gain.
- Uncertainty modeling leaves the point metrics unchanged but adds confidence bounds.

### B.5 Other reported results

- **CatBoost monotonicity compliance rate** (w/c sweep, other features at median): **above 97 %**.
- **Paired t-test** of the proposed model against the best baseline (CatBoost): **p < 0.01**, statistically significant.
- **Computational complexity:** O(K · M · T · log n).
- **Setup:** 80:20 split, 10-fold CV on the training set only, seed 42, Intel Core i5 with 16 GB RAM and no GPU. Libraries: NumPy, Pandas, scikit-learn, XGBoost, LightGBM and CatBoost.

---

### Implementation notes

- **Unreported settings:** the paper does not give search ranges, the number of trials, the ElasticNet α/λ, or the quantile-regressor hyperparameters. The code uses standard choices for these and marks them as such.
- **Superplasticizer ratio:** Sect. 3 lists it among the derived features, but Eqs. 2–5 omit it. The code includes it as `sp_b`. Remove it from `add_domain_features` to match the equations exactly.
- **Interval level:** Sect. 4.6 defines the interval as [ŷ₀.₁, ŷ₀.₉], which is a nominal 80 % interval, but Table 3 labels PICP as a "90 % interval". The code follows Sect. 4.6. For a nominal 90 % interval, set `TAUS = [0.05, 0.5, 0.95]`.
- **Quantile branch:** the code trains it on the training-set OOF meta-features and evaluates it on the test-set meta-features. This is how "no additional data leakage" in Sect. 4.6 is read here.
