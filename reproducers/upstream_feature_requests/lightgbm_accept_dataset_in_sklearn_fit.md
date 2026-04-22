# [RFC] `LGBMModel.fit(X=Dataset)` — accept a pre-built Dataset to enable efficient weight/label change


## Motivation

Same root cause as [#5074](https://github.com/microsoft/LightGBM/issues/5074), re-opening with a concrete API proposal. On multi-GB / multi-weight training runs, `LGBMModel.fit(X, y, sample_weight=w)` rebuilds the `Dataset` on every call at [lightgbm/sklearn.py:978-987](https://github.com/microsoft/LightGBM/blob/master/python-package/lightgbm/sklearn.py). This affects all three sklearn wrappers — `LGBMClassifier`, `LGBMRegressor`, and `LGBMRanker` — since they all inherit `fit` from `LGBMModel`. With N weight schemes (recency / uniform / custom) across M models in a sklearn `Pipeline` or hyperparam sweep, that's N × M full-Dataset rebuilds for the same feature matrix — on our 200+ GB production dataset each completely unnecessary rebuild takes up to 10 minutes, and the aggregate dominates wall-clock.

The native `lightgbm.train(params, train_set)` API takes a pre-built `Dataset` and supports `train_set.set_label(new_label)` / `train_set.set_weight(new_weight)` in-place (both documented as returning `self`, in-place on the C++ handle via `set_field`). But the sklearn API's `early_stopping`, `callbacks`, `predict_proba`, `feature_importances_`, pipeline compatibility, and grid-search integration force users back to `LGBMModel.fit(X, y)` and block reuse.

## Precedent — CatBoost already supports this

**CatBoost's sklearn wrapper already does exactly what this RFC proposes:** `CatBoostClassifier.fit(X=Pool)` short-circuits rebuild via an `isinstance(X, Pool)` check in `_build_train_pool` (installed `catboost 1.2.10`, `catboost/core.py:1587-1590`):

```python
def _build_train_pool(X, y, cat_features, text_features, embedding_features, ...):
    if isinstance(X, Pool):
        train_pool = X
        ...  # label overwrite via _set_pool_label_with_overwrite_warning
```

Callers build one `Pool` and call `fit()` multiple times with `pool.set_label` / `pool.set_weight` in between. It's idiomatic, documented in CatBoost tutorials, and a major performance win for hyperparameter-sweep / multi-target / cross-validated workflows. LightGBM and XGBoost are the two libraries among the "big three" whose sklearn wrappers force a full rebuild on every fit; this RFC proposes the fix for LightGBM. Closing this gap would bring LGB in line with CatBoost for production training pipelines that re-use the same feature matrix across multiple fits.

## Proposal

Add a fast-path in `LGBMModel.fit` (`lightgbm/sklearn.py` ~line 948, before the `_LGBMValidateData` branch). Because the fix is in the base class, it covers `LGBMClassifier`, `LGBMRegressor`, and `LGBMRanker` with a single change:

```python
if isinstance(X, Dataset):
    # Pre-built Dataset; skip the sklearn-validator path (the Dataset
    # was validated at construction time) and reuse in place.
    train_set = X
    if y is not None:
        train_set.set_label(y)
    if sample_weight is not None:
        train_set.set_weight(sample_weight)
    # _X / _y are only used for eval_set construction and some sklearn
    # metadata (feature_names_in_, n_features_in_); populate from the
    # Dataset's own accessors.
    _X, _y = None, y  # subsequent branches gate on isinstance(_X, ...)
    feature_names_in_ = train_set.feature_name
    n_features_in_ = train_set.num_feature()
else:
    # existing path: _LGBMValidateData + Dataset(data=_X, label=_y, ...)
```

No default behaviour change for `fit(X: array_like, y, ...)`. The `isinstance` gate is strictly additive.

## Use-case snippet (with the fix)

```python
import lightgbm as lgb

# Build the Dataset once for the full 200+ GB feature matrix.
# set_label / set_weight write directly to the C++ handle via LGBM_DatasetSetField
# and work post-construct regardless of free_raw_data.
train_set = lgb.Dataset(X_train, label=y_churn, weight=w_uniform,
                        categorical_feature=cat_cols)
val_set   = lgb.Dataset(X_val, label=y_churn_val, reference=train_set)

# LGBMClassifier — vary weight schemes for a single target.
for weight_scheme in ("uniform", "recency", "inverse_recency"):
    train_set.set_weight(weight_for_scheme(weight_scheme))
    clf = lgb.LGBMClassifier(n_estimators=1000)
    clf.fit(train_set, eval_set=[(val_set,)], callbacks=[lgb.early_stopping(50)])
    ...  # predict_proba, feature_importances_, booster_ — all sklearn API

# LGBMClassifier / LGBMRegressor / LGBMRanker — vary targets across the same feature matrix.
targets = {
    "churn":              (lgb.LGBMClassifier(n_estimators=1000),              y_churn_train,    y_churn_val),
    "next_best_action":   (lgb.LGBMClassifier(n_estimators=1000),              y_nba_train,      y_nba_val),
    "best_discount_pct":  (lgb.LGBMRegressor(n_estimators=1000),               y_discount_train, y_discount_val),
    "job_rank":           (lgb.LGBMRanker(n_estimators=1000, objective="rank"), y_rank_train,     y_rank_val),
}
for target_name, (model, y_train, y_val) in targets.items():
    train_set.set_label(y_train)
    val_set.set_label(y_val)
    model.fit(train_set, eval_set=[(val_set,)], callbacks=[lgb.early_stopping(50)])
    ...
```

## Backwards compatibility

Purely additive — existing callers are unchanged. The `Dataset` branch is gated on `isinstance(X, Dataset)`.

## Relation to existing threads

- [#5074](https://github.com/microsoft/LightGBM/issues/5074) — "Fitting LGBMClassifier on a Dataset". The earlier issue asked about the performance cost of extracting `.data` / `.label` from a Dataset to feed the sklearn API; it was closed without documented resolution. This RFC proposes the concrete API to eliminate the extraction entirely.
- [#4965](https://github.com/microsoft/LightGBM/issues/4965) — Documents the Dataset lifecycle constraint that users of this RFC must understand: once `construct()` runs (internally on the first `train()` call), the feature layout is frozen — feature data is binned and cannot be changed. `set_label()` and `set_weight()` work post-construct because they write directly to the C++ handle via `LGBM_DatasetSetField`, not through the Python raw-data references. The practical rule: build the Dataset once with the final feature set; call `set_label`/`set_weight` freely between fits; never alter columns or dtypes.

## Alternatives considered

- **Using `lightgbm.train` directly + a thin sklearn-facade adapter in userspace:** the facade has to re-implement `predict_proba`, `feature_importances_`, `_le` label-encoder plumbing, early-stopping callbacks, pipeline compatibility. Non-trivial boilerplate that's fragile on version bumps.
- **Monkey-patching `LGBMModel.fit`:** brittle; breaks silently when `_LGBMValidateData` signature changes.

## Willingness to PR

Happy to submit the PR with unit tests covering:
- `LGBMClassifier.fit(Dataset, y)`, `LGBMRegressor.fit(Dataset, y)`, and `LGBMRanker.fit(Dataset, y)` set label correctly.
- `fit(Dataset)` with pre-set label.
- `set_weight` round-trip across fits preserves metric parity vs `fit(array, y, sample_weight=...)`.
- `fit(Dataset, eval_set=[(Dataset,)])` with both as Dataset.
- `set_label` works post-construct (after the C++ handle is built).
- `feature_name` / `categorical_feature` preservation.
- Pipeline / GridSearchCV compatibility via `feature_names_in_` accessor.
