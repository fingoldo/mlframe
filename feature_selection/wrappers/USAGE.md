# RFECV Usage Guide

`mlframe.feature_selection.wrappers.RFECV` is a feature selector with multiple
algorithm paths. This guide shows recommended usage for common scenarios.

## Quick reference: which path do I want?

| Your situation | Recommended config |
|---|---|
| **Small data, single estimator, want recall** | `RFECV(estimator=lr)` (default) |
| **Small data, want robustness** | `RFECV(estimator=lr, stability_selection=True)` |
| **Multi-estimator voting (LR+RF+...)** | `RFECV(estimators=[lr, rf, ...], stability_selection=True)` |
| **High-cardinality features (categoricals)** | `importance_getter='permutation'` or `'knockoff'` |
| **Production data with potential leak** | `leakage_corr_threshold=0.95, leakage_action='exclude'` |
| **Large p (≥1000), p ≪ n** | `importance_getter='knockoff'` (much faster than baseline) |
| **One-hot encoded categoricals** | `feature_groups={'cat1': ['c1_a','c1_b']}` |
| **Force-keep specific features** | `must_include=['core_feature']` |
| **Force-exclude (e.g. data leak)** | `must_exclude=['user_id']` |

## Examples

### 1. Vanilla RFECV (most common)

```python
from sklearn.linear_model import LogisticRegression
from mlframe.feature_selection.wrappers import RFECV

rfecv = RFECV(
    estimator=LogisticRegression(max_iter=400, random_state=0),
    cv=3,
    max_refits=10,
    leakage_corr_threshold=0.95,    # auto-detect target leakage
    leakage_action='exclude',        # auto-drop high-corr columns
)
rfecv.fit(X_train, y_train)

selected = rfecv.get_feature_names_out()
X_train_selected = rfecv.transform(X_train)
```

### 2. Stability Selection (robust on small data)

```python
rfecv = RFECV(
    estimator=LogisticRegression(max_iter=400, random_state=0),
    stability_selection=True,
    stability_n_bootstrap=50,
    stability_threshold=0.6,         # feature in >=60% of bootstraps wins
)
rfecv.fit(X, y)

# Inspect per-feature bootstrap frequency
freqs = rfecv.stability_selection_freq_
for name, freq in zip(rfecv.feature_names_in_, freqs):
    print(f"{name}: {freq:.2%}")
```

### 3. Multi-estimator voting

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

rfecv = RFECV(
    estimators=[
        LogisticRegression(max_iter=400, random_state=0),
        RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1),
    ],
    stability_selection=True,        # bypasses MBH, robust
    stability_n_bootstrap=30,
)
rfecv.fit(X, y)
```

**Don't** parallelise across estimators - they each use native multi-threading
(CB/LGB/XGB/RF). Parallelism lives at the CV-fold level via `n_jobs`.

### 4. Knockoffs (production-scale, bias-free)

```python
from mlframe.feature_selection.wrappers import (
    knockoff_importance, select_features_fdr,
)

# Compute W statistic per feature
W = knockoff_importance(
    model_factory=lambda: LogisticRegression(max_iter=400, random_state=0),
    X=X, y=y, random_state=0,
)

# Select with provable FDR control
selected_names = select_features_fdr(W, q=0.1)  # FDR <= 10%
X_selected = X[selected_names]
```

Knockoffs are 10-50× faster than baseline RFECV on n>1000, p<5000 with
paritetic recall. Best for production retraining pipelines.

### 5. Feature groups (one-hot all-or-nothing)

```python
rfecv = RFECV(
    estimator=LogisticRegression(),
    feature_groups={
        'category_A': ['cat_A_1', 'cat_A_2', 'cat_A_3'],
        'category_B': ['cat_B_1', 'cat_B_2'],
    },
)
rfecv.fit(X, y)
# Either ALL of category_A's columns end up in support_, or none of them.
```

### 6. Permutation / SHAP importance

```python
# Permutation: model-agnostic, robust to FI bias on tree estimators
rfecv = RFECV(
    estimator=RandomForestClassifier(),
    importance_getter='permutation',
)

# SHAP: faster for tree models, but inherits the same high-cardinality bias
# as Gini/feature_importances_ - use permutation/knockoff for unbiased FI
rfecv = RFECV(
    estimator=RandomForestClassifier(),
    importance_getter='shap',
)
```

### 7. n_features_selection_rule for plateau handling

When the score curve is flat between many N values (common at n>>p):

```python
# Default: auto -> argmax for singular, one_se_max for multi-estimator
rfecv = RFECV(estimator=lr)  # auto

# Parsimonious (sklearn-canonical 1-SE rule, smallest N in band):
rfecv = RFECV(estimator=lr, n_features_selection_rule='one_se_min')

# Recall-focused (largest N in 1-SE band, plateau-resistant):
rfecv = RFECV(estimator=lr, n_features_selection_rule='one_se_max')

# Legacy (pure argmax, may pick random N on flat plateau):
rfecv = RFECV(estimator=lr, n_features_selection_rule='argmax')
```

### 8. Diagnostics after fit

```python
rfecv.fit(X, y)

# How stable is the selection across CV folds?
stability = rfecv.selection_stability_(metric='jaccard')

# What's the SE-band CI on the optimal N?
low, n, high = rfecv.n_features_bootstrap_ci_(n_bootstrap=200, ci=0.9)
print(f"Optimal n_features_: {n} (90% CI [{low}, {high}])")

# 1-SE rule: most parsimonious N within 1 SE of best
n_parsimonious = rfecv.n_features_one_se_()
```

## Performance benchmark (n=8000, p=200, 30 informative)

| Method | n_sel | recall | time | stability |
|---|---|---|---|---|
| baseline RFECV (LR) | 150 | 0.99 | 5.1s | 0.57 |
| stability_selection (LR) | 32 | 0.86 | 2.4s | 0.56 |
| stability + multi (LR+RF) | 31 | 0.86 | 31.4s | 0.57 |
| **knockoffs (LR, BC q=0.5)** | 52 | 0.86 | **0.54s** | NaN |

Knockoffs are 9.4× faster than baseline with paritetic recall on production-scale data.

## Edge-case safety

RFECV validates the inputs at fit entry to prevent silent corruption:

- Single-class y → `ValueError`
- NaN/Inf in y → `ValueError`
- Minority class < n_splits → `ValueError`
- `must_include` ∩ `must_exclude` → `ValueError`
- `transform` before `fit` → `NotFittedError`
- Column drift between fit and transform → `RuntimeError`
- Target leakage (Pearson > threshold) → `'warn'` / `'exclude'` / `'raise'` per `leakage_action`

## See also

- `feature_selection/wrappers/TODO.md` — deferred ML improvements
- `feature_selection/_benchmarks/bench_pr4_methods.py --large` — full bench
- `tests/feature_selection/test_wrappers_*.py` — comprehensive test suite
