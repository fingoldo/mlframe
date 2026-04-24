# Multi-Output Classification in mlframe

This document describes the design and usage of multi-output classification
support: `MULTICLASS_CLASSIFICATION` (K>2 single-label) and
`MULTILABEL_CLASSIFICATION` (K independent binary outputs).

## Target type matrix

| Target | y shape | Decision rule | Probability output | Library dispatch |
|---|---|---|---|---|
| `BINARY_CLASSIFICATION` | (N,) int{0,1} | `probs[:, 1] >= threshold` | (N, 2) | per library binary objective |
| `MULTICLASS_CLASSIFICATION` | (N,) int{0..K-1} | `argmax(probs, axis=1)` | (N, K), rows sum to 1 | softmax per library |
| `MULTILABEL_CLASSIFICATION` | (N, K) int{0,1} | `(probs >= threshold)` per label | (N, K), independent | native (CB) or wrapper |
| `REGRESSION` | (N,) float | identity | (N,) float | per library regression objective |

## Per-strategy dispatch

| Strategy | multiclass | multilabel default | multilabel notes |
|---|---|---|---|
| **CatBoost** | `loss_function='MultiClass'` | **NATIVE** `loss_function='MultiLogloss'` | single tree ensemble, returns (N, K). Best path. |
| **XGBoost** | `objective='multi:softprob' + num_class=K` | `MultiOutputClassifier` wrapper | Native `multi_strategy='multi_output_tree'` available via `MultilabelDispatchConfig.force_native_xgb_multilabel=True` (XGB 3.x experimental, marked WIP — opt-in only). |
| **LightGBM** | `objective='multiclass' + num_class=K` | `MultiOutputClassifier` wrapper | No native multilabel (issue #524 since 2017). |
| **HistGradientBoosting** | sklearn auto-detects | `MultiOutputClassifier` wrapper | sklearn `HistGradientBoostingClassifier` doesn't support multilabel natively. |
| **Linear (LogisticRegression)** | `multi_class='multinomial', solver='lbfgs'` | `MultiOutputClassifier` wrapper | LR has no native multilabel; OvR is the standard. |
| **Recurrent (LSTM/Transformer)** | `softmax` head + `MulticlassAUROC` | NotImplementedError | Multilabel needs separate sigmoid head — Session-3. |

## MultilabelDispatchConfig

```python
from mlframe.training import MultilabelDispatchConfig

cfg = MultilabelDispatchConfig(
    strategy="auto",         # "auto" | "wrapper" | "chain" | "native"
    n_chains=3,              # for chain strategy: ensemble of 3 random-ordered chains
    chain_order_strategy="random",  # "random" | "by_frequency" | "user"
    chain_order_user=None,   # only when chain_order_strategy="user"
    chain_seeds=None,        # default [0, 1, ..., n_chains-1]
    cv=5,                    # ClassifierChain.cv — cross-validates chain features
    per_label_thresholds=None,  # Optional[List[float]] for per-label decision rule
    wrapper_n_jobs="auto",   # MultiOutputClassifier.n_jobs (auto: min(K, cpu/2))
    allow_uncalibrated_multi=False,  # downgrade calib NotImplementedError to warn
    force_native_xgb_multilabel=False,  # XGB 3.x experimental multi_output_tree
)
```

### Strategy choices

- **`auto`** — let the strategy pick. CB → native MultiLogloss; everyone else → MultiOutputClassifier.
- **`wrapper`** — force `MultiOutputClassifier` even for CB (degrades CB native to OvR; useful for A/B comparison).
- **`chain`** — `_ChainEnsemble` of 3 random-ordered `ClassifierChain` instances; averages predict_proba. Empirically +0.5-2pp Jaccard on correlated labels (verified on `tests/training/test_bizvalue_classifier_chain.py` — +0.59pp mean over 5 seeds, positive in 5/5).
- **`native`** — assert strategy supports native multilabel; raise if not. For users who explicitly want CB MultiLogloss and want to fail loud if mis-configured.

### When to pick each strategy

| Scenario | Recommended | Rationale |
|---|---|---|
| CatBoost available, no label correlation | `auto` (→ native) | Fastest, single tree ensemble |
| LGB / XGB / HGB / Linear, no correlation | `auto` (→ wrapper) | Standard OvR — simple, well-understood |
| Strong label correlation (tagging, hierarchies) | `chain` | +0.5-5pp Jaccard at 3-5x training cost |
| A/B vs native CB | `wrapper` | Degrades CB to OvR for direct comparison |
| XGB 3.x stability acceptable | `force_native_xgb_multilabel=True` | Vector-output trees, integrated GPU/SHAP |

## Stratified splitting

```python
from mlframe.training import make_train_test_split
import numpy as np

# Multiclass — 1-D stratify_y triggers sklearn StratifiedShuffleSplit
y = np.array([0, 1, 2, 0, 1, 2, ...])
train, val, test, *_ = make_train_test_split(df, stratify_y=y, test_size=0.1)

# Multilabel — 2-D stratify_y triggers iterstrat.MultilabelStratifiedShuffleSplit
y_multi = np.array([[1, 0, 1], [0, 1, 1], ...])  # (N, K) binary
train, val, test, *_ = make_train_test_split(df, stratify_y=y_multi, test_size=0.1)
```

For multilabel stratification, install the optional dependency:
```
pip install iterative-stratification
```

Lazy-imported on first 2-D `stratify_y` use; helpful `ImportError` message
if missing.

## Probability surface contract

All classification estimators' `predict_proba` is canonicalised to `(N, K)` shape:

```python
from mlframe.training import canonical_predict_proba_shape, predict_from_probs, TargetTypes

# Whatever predict_proba returned (binary (N,2), 1-D sigmoid, list-of-arrays
# from MultiOutputClassifier, native CB (N,K)) → unified (N, K)
probs_NK = canonical_predict_proba_shape(probs, classes_=getattr(model, "classes_", None))

# Decision rule per target_type
preds = predict_from_probs(probs_NK, TargetTypes.MULTICLASS_CLASSIFICATION, classes_=model.classes_)
# returns (N,) int labels via argmax

preds = predict_from_probs(probs_NK, TargetTypes.MULTILABEL_CLASSIFICATION, threshold=0.5)
# returns (N, K) int{0,1} matrix via per-label threshold
```

Per-label thresholds for cost-sensitive multilabel:
```python
preds = predict_from_probs(probs_NK, TargetTypes.MULTILABEL_CLASSIFICATION,
                           threshold=np.array([0.5, 0.3, 0.7]))  # per-label
```

## Multilabel metrics

```python
from mlframe.metrics import (
    hamming_loss,           # mean fraction of incorrect labels
    subset_accuracy,        # exact-match (all labels correct per row)
    jaccard_score_multilabel,  # per-row averaged Jaccard, empty-union → 1.0
)

ham = hamming_loss(y_true, y_pred)
sub = subset_accuracy(y_true, y_pred)
jac = jaccard_score_multilabel(y_true, y_pred)
```

Numba-implemented — sequential + parallel variants. Parallel auto-selected
when `N*K > 1_000_000`. Bitmap popcount fast-path for K∈[16, 64] gives
8.6× speedup at K=64.

### Metrics registry (Session 4)

Built-in multilabel metrics dispatched automatically from
`report_probabilistic_model_perf` via `mlframe.training.metrics_registry`:

```python
from mlframe.training.metrics_registry import (
    register_metric, list_registered, iter_extra_metrics,
)
from mlframe.training.configs import TargetTypes

# Inspect registered metrics
print(list_registered(TargetTypes.MULTILABEL_CLASSIFICATION))
# ['hamming_loss', 'subset_accuracy', 'jaccard_samples']

# Register a domain-specific metric (no evaluation.py edit required)
def my_multilabel_score(y_true, probs_NK, preds_NK):
    # Custom scoring logic
    return some_value

register_metric(
    TargetTypes.MULTILABEL_CLASSIFICATION, "my_score", my_multilabel_score,
)
```

Failing metrics (any exception during compute) are silently skipped —
report keeps going. Idempotent registration; unregister via
`unregister_metric(target_type, name)`.

## Calibration

Post-hoc calibration (`post_calibrate_model`) dispatches by target_type:

| Target | Calibrator | Output |
|---|---|---|
| `BINARY_CLASSIFICATION` | univariate `IsotonicRegression` mapping `probs[:, 1]` | `(N, 2)` calibrated |
| `MULTICLASS_CLASSIFICATION` | **`_PerClassIsotonicCalibrator`** (K independent isotonics) + row-normalisation | `(N, K)`, rows sum to 1 |
| `MULTILABEL_CLASSIFICATION` | **`_PerClassIsotonicCalibrator`** (K independent isotonics, columns independent) | `(N, K)`, per-label in [0,1] |

Per-class isotonic semantics (Session 4):
- Each class k gets its own `IsotonicRegression.fit(probs[:, k], y_k)`
  where `y_k = (y == k)` for MULTICLASS or `y[:, k]` for MULTILABEL
- Constant-label columns (all 0 or all 1 in calibration set) → identity
  mapping (skipped, no crash)
- Output clipped to `[0, 1]`; MULTICLASS re-normalised row-wise to
  preserve softmax invariant; MULTILABEL columns stay independent
- Wrapped in `_PostHocMultiCalibratedModel` for transparent
  `predict_proba` / `predict` delegation; pickle-roundtrip verified

## FeaturesAndTargetsExtractor for multilabel

Standard FTE accepts a single `target_column`. For multilabel:
- **Polars**: target column is `pl.List(pl.Int8)` or `pl.Array(pl.Int8, K)`.
  Auto-unpacked to `(N, K)` ndarray by FTE.
- **Pandas**: target column is `object` dtype with list/tuple cells.
  Auto-stacked to `(N, K)`.
- **Native 2-D ndarray**: pass as-is.

```python
import polars as pl
df = pl.DataFrame({
    "feature_a": [1.0, 2.0, 3.0],
    "target": pl.Series([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=pl.List(pl.Int8)),
})
fte = SimpleFeaturesAndTargetsExtractor(
    target_column="target",
    target_type=TargetTypes.MULTILABEL_CLASSIFICATION,
)
result = fte.transform(df)
# result[1][TargetTypes.MULTILABEL_CLASSIFICATION]['target'].shape == (3, 3)
```

## Ensembling for multi-output

`mlframe.ensembling.ensemble_probabilistic_predictions` works on multi-output
shapes after the materialisation-dedup refactor (Session-1 landed) — single
`_preds_arr = np.asarray(preds)` cache, ~5× peak-memory reduction.

For prod-sized frames (N=9M+, M=6, K=5) the materialised (M, N, K) tensor
is 2.16 GB — uncomfortably close to Win32 4 GB ceiling. Use the streaming
accumulator API:

```python
from mlframe.ensembling import _WelfordAccumulator

acc = _WelfordAccumulator(shape=(N, K))
for model in models:
    acc.push(model.predict_proba(X_val))  # one (N, K) at a time
result = acc.result()
# {'mean': (N, K), 'std': (N, K), 'min': (N, K), 'max': (N, K), 'n': M}
```

Memory: O(N*K) regardless of M. For median/quantile aggregations,
P²-Quantile streaming sketch is planned (Session-3).

## Related docs

- `docs/NUMERICAL_STABILITY_REPORT.md` — Welford / Kahan numerical-stability audit + benchmarks
- `tests/training/COMBO_FUZZ.md` — combo-fuzz harness covering multi-output combos
- `tests/training/test_bizvalue_classifier_chain.py` — empirical justification for ChainEnsemble dispatch

## Out-of-scope (Session 3+)

- Recurrent / NeuralNet multilabel sigmoid head
- Per-class isotonic calibration for multi-output
- Polars-native `pl.Array(pl.Int8, K)` schema integration through full pipeline
- Native XGB 3.x multilabel as default (waiting for v3.1 stable)
- P²-Quantile / T-Digest streaming quantile accumulators
