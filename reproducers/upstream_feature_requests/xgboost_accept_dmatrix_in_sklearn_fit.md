# [RFC] `XGBClassifier.fit(X=DMatrix)` — accept a pre-built DMatrix to enable weight/label reuse across fits

**Target repo:** [dmlc/xgboost](https://github.com/dmlc/xgboost) — open as a new issue under "feature request" label.

---

## Motivation

When running a training suite that explores multiple sample-weight schemes (e.g. recency-weighted vs uniform) or sweeps multiple targets sharing the same feature matrix, the `XGBClassifier` / `XGBRegressor` sklearn wrappers rebuild the internal `DMatrix` (or `QuantileDMatrix`) on every `.fit()` call. On multi-GB datasets the rebuild dominates the wall-clock budget — one of our production workloads (9 018 479 × 118, cat-heavy) spends tens of seconds per rebuild, repeated N_models × N_weights times.

The native `xgb.train(params, dtrain)` API accepts a pre-built `DMatrix` and supports in-place `dtrain.set_label(new_label)` / `dtrain.set_weight(new_weight)` between fits, but callers using the sklearn interface (for early stopping, `predict_proba`, `feature_importances_`, pipelines, cross-validated estimators, MLflow integrations) have no clean way to reuse the DMatrix — the wrapper always rebuilds in `_create_dmatrix` at [xgboost/sklearn.py:1253-1262](https://github.com/dmlc/xgboost/blob/master/python-package/xgboost/sklearn.py).

## Precedent — CatBoost already supports this

**CatBoost's sklearn wrapper already does exactly what this RFC proposes:** `CatBoostClassifier.fit(X=Pool)` short-circuits rebuild via an `isinstance(X, Pool)` check in `_build_train_pool` (installed `catboost 1.2.10`, `catboost/core.py:1587-1590`):

```python
def _build_train_pool(X, y, cat_features, text_features, embedding_features, ...):
    if isinstance(X, Pool):
        train_pool = X
        ...  # label overwrite via _set_pool_label_with_overwrite_warning
```

Callers routinely build one `Pool` and call `fit()` multiple times with `pool.set_label` / `pool.set_weight` in between — this is idiomatic, documented in the CatBoost tutorials, and delivers a significant speed-up for grid-search / hyperparameter-sweep / multi-target workloads. Among the "big three" GBDT libraries, XGBoost is the only one whose sklearn wrapper forces a full rebuild on every fit. Closing this gap would make XGBoost more competitive in production training pipelines where the same feature matrix is re-used dozens of times.

## Proposal

Add a fast-path in `XGBModel._create_dmatrix` (`xgboost/sklearn.py` ~line 1253):

```python
def _create_dmatrix(self, ref: Optional[DMatrix], *, X, y=None, **kwargs) -> DMatrix:
    # NEW: accept a pre-built DMatrix as X. The caller guarantees
    # feature names / feature types / categorical layout match the
    # model's expectations; we skip the rebuild entirely.
    if isinstance(X, DMatrix):
        if y is not None:
            X.set_label(y)
        sample_weight = kwargs.get("weight")
        if sample_weight is not None:
            X.set_weight(sample_weight)
        base_margin = kwargs.get("base_margin")
        if base_margin is not None:
            X.set_base_margin(base_margin)
        return X
    # ...existing QuantileDMatrix / DMatrix construction...
```

No default-behaviour change — the `isinstance` gate is strictly additive. `feature_names` / `feature_types` round-trip through `DMatrix` already, so `feature_names_in_` stays consistent on the wrapper.

## Use-case snippet (with the fix)

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y_train_u, weight=w_uniform, enable_categorical=True)
dval   = xgb.DMatrix(X_val,   label=y_val,     enable_categorical=True)

for weight_scheme in ("uniform", "recency", "inverse_recency"):
    dtrain.set_weight(weight_for_scheme(weight_scheme))
    model = xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_estimators=1000)
    model.fit(dtrain, eval_set=[(dval, y_val)], early_stopping_rounds=50)
    ...  # predict_proba, feature_importances_, save_model — all sklearn API
```

## Backwards compatibility

Purely additive — existing `fit(X: array_like, y, ...)` paths are unchanged. The `DMatrix` branch is gated on `isinstance(X, DMatrix)`.

## Related existing threads

- [#4190](https://github.com/dmlc/xgboost/issues/4190) — access `DMatrix.set_base_margin` via sklearn wrapper. **Accepted**, implemented via PR #5151. Precedent that upstream adds DMatrix capabilities through the sklearn wrapper when there's a concrete use case.
- [#7817](https://github.com/dmlc/xgboost/issues/7817) — `categorical_features` param gap between sklearn wrapper and DMatrix.
- [#2000](https://github.com/dmlc/xgboost/issues/2000) — "How to feed test data to my xgb model?" (tangentially related).

## Alternatives considered

- **Using `xgb.train` directly + a thin sklearn-facade adapter in userspace:** 30+ lines per wrapper, duplicates `predict_proba` / `feature_importances_` logic, fragile on version bumps, breaks pipeline / grid-search integrations.
- **Monkey-patching `_create_dmatrix`:** brittle, tightly couples userspace to internal layout. Observed to break between 2.x and 3.x minor releases.

## Willingness to PR

Happy to submit the PR with unit tests covering:
- `fit(DMatrix, y)` sets label correctly.
- `fit(DMatrix)` with pre-set label.
- `set_weight` round-trip across fits preserves metric parity vs fit(array, y, sample_weight=...).
- `feature_names` preservation.
- Mismatched feature shapes rejected clearly.
- `eval_set=[(DMatrix, y)]` also short-circuits (scope extension).
