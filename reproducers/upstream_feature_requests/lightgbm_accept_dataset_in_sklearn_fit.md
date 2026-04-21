# [RFC] `LGBMClassifier.fit(X=Dataset)` — accept a pre-built Dataset to enable weight/label reuse (follow-up to #5074)

**Target repo:** [microsoft/LightGBM](https://github.com/microsoft/LightGBM) — open as a new issue; reference #5074 and #2966 in the body.

---

## Motivation

Same root cause as [#5074](https://github.com/microsoft/LightGBM/issues/5074), re-opening with a concrete API proposal. On multi-GB / multi-weight training runs, `LGBMClassifier.fit(X, y, sample_weight=w)` rebuilds the `Dataset` on every call at [lightgbm/sklearn.py:978-987](https://github.com/microsoft/LightGBM/blob/master/python-package/lightgbm/sklearn.py). With N weight schemes (recency / uniform / custom) across M models in a sklearn `Pipeline` or hyperparam sweep, that's N × M full-Dataset rebuilds for the same feature matrix — in our 9M-row production workload, each rebuild is several seconds and the aggregate dominates wall-clock.

The native `lightgbm.train(params, train_set)` API takes a pre-built `Dataset` and supports `train_set.set_label(new_label)` / `train_set.set_weight(new_weight)` in-place (both documented as returning `self`, in-place on the C++ handle via `set_field`). But the sklearn API's `early_stopping`, `callbacks`, `predict_proba`, `feature_importances_`, pipeline compatibility, and grid-search integration force users back to `LGBMClassifier.fit(X, y)` and block reuse.

## Precedent — CatBoost already supports this

**CatBoost's sklearn wrapper already does exactly what this RFC proposes:** `CatBoostClassifier.fit(X=Pool)` short-circuits rebuild via an `isinstance(X, Pool)` check in `_build_train_pool` (installed `catboost 1.2.10`, `catboost/core.py:1587-1590`):

```python
def _build_train_pool(X, y, cat_features, text_features, embedding_features, ...):
    if isinstance(X, Pool):
        train_pool = X
        ...  # label overwrite via _set_pool_label_with_overwrite_warning
```

Callers build one `Pool` and call `fit()` multiple times with `pool.set_label` / `pool.set_weight` in between. It's idiomatic, documented in CatBoost tutorials, and a major performance win for hyperparameter-sweep / multi-target / cross-validated workflows. Among the "big three" GBDT libraries, LightGBM is the only one whose sklearn wrapper forces a full rebuild on every fit. Closing this gap would bring LGB in line with the other two and make the sklearn wrapper usable for production training pipelines that re-use the same feature matrix dozens of times.

## Proposal

Add a fast-path in `LGBMModel.fit` (`lightgbm/sklearn.py` ~line 948, before the `_LGBMValidateData` branch):

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

train_set = lgb.Dataset(X_train, label=y_train, weight=w_uniform,
                        categorical_feature=cat_cols, free_raw_data=False)
val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set, free_raw_data=False)

for weight_scheme in ("uniform", "recency", "inverse_recency"):
    train_set.set_weight(weight_for_scheme(weight_scheme))
    model = lgb.LGBMClassifier(n_estimators=1000)
    model.fit(train_set, eval_set=[(val_set,)], callbacks=[lgb.early_stopping(50)])
    ...  # predict_proba, feature_importances_, booster_ — all sklearn API
```

## Backwards compatibility

Purely additive — existing callers are unchanged. The `Dataset` branch is gated on `isinstance(X, Dataset)`.

## Relation to existing threads

- [#5074](https://github.com/microsoft/LightGBM/issues/5074) — "Fitting LGBMClassifier on a Dataset". The earlier issue asked about the performance cost of extracting `.data` / `.label` from a Dataset to feed the sklearn API; it was closed without documented resolution. This RFC proposes the concrete API to eliminate the extraction entirely.
- [#2966](https://github.com/microsoft/LightGBM/issues/2966) — "The sklearn wrapper is not really compatible with the sklearn ecosystem" — explicit acknowledgement that the wrapper diverges from the native API.
- [#4965](https://github.com/microsoft/LightGBM/issues/4965) — Dataset reuse constraints (`free_raw_data=False` trade-off); orthogonal but relevant for users adopting this RFC.

## Alternatives considered

- **Using `lightgbm.train` directly + a thin sklearn-facade adapter in userspace:** the facade has to re-implement `predict_proba`, `feature_importances_`, `_le` label-encoder plumbing, early-stopping callbacks, pipeline compatibility. 50+ lines per wrapper, fragile on version bumps.
- **Monkey-patching `LGBMModel.fit`:** brittle; breaks silently when `_LGBMValidateData` signature changes.

## Willingness to PR

Happy to submit the PR with unit tests covering:
- `fit(Dataset, y)` sets label correctly.
- `fit(Dataset)` with pre-set label.
- `set_weight` round-trip across fits preserves metric parity vs fit(array, y, sample_weight=...).
- `fit(Dataset, eval_set=[(Dataset,)])` with both as Dataset.
- `free_raw_data=True` edge case (Dataset.set_label must still work post-construct once the handle is built).
- `feature_name` / `categorical_feature` preservation.
- Pipeline / GridSearchCV compatibility via feature_names_in_ accessor.
