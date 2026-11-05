# Wave 1 (2026-11-05), attack class A10: API contract and configuration consistency

Read-only audit by one Opus 5.5 low-effort agent. Seven defects verified on the current tree. The most serious is in RFECV: once a grouped config object is passed, changing the same setting through `set_params` either makes `clone()` raise or has no effect. MRMR already handles this case; RFECV does not. The report is the agent's returned text, condensed; each finding is re-checked on HEAD before it is fixed.

## Findings

| ID | Severity | Location | Defect |
|---|---|---|---|
| P10-01 | Medium | `src/mlframe/feature_selection/wrappers/rfecv/__init__.py:450` (with `_overlay_explicit_config_fields`, line 78) | When a config object is set, `set_params(<flat field>=v)` followed by `clone()` raises `RuntimeError`, because `__init__` re-applies the config's explicit fields over the flat kwarg. |
| P10-02 | Medium | `src/mlframe/feature_selection/wrappers/rfecv/__init__.py:450`; no `set_params` override in `rfecv/` | `set_params(search_config=...)` (and `fi_config`, `robustness_config`) is stored but never expanded onto the flat attributes, so it is ignored. |
| P10-03 | Low | `src/mlframe/estimators/early_stopping.py:105-112` | Comment says monotonic-decline "Default-on at 5"; the signature default is 7, and the suite's `TrainingBehaviorConfig` uses 20 while claiming to mirror the wrapper paths. |
| P10-04 | Low | `src/mlframe/training/strategies/base.py:127-136` | `_InfToNaNTransformer._replace` only handles pandas frames and float ndarrays: a mixed-dtype polars frame passes through with inf untouched, and an all-float polars frame comes back as an ndarray. |
| P10-05 | Low | `src/mlframe/training/strategies/base.py:184-196` | `_NumericOnlyTransformer.fit` fits the constructor parameter `inner` in place instead of a clone, so `get_params()['inner']` comes back fitted. |
| P10-06 | Low | `src/mlframe/estimators/custom.py:665-676` | `IdentityEstimator.fit` never sets `n_features_in_` (or `feature_names_in_`). |
| P10-07 | Low | `docs/ENVIRONMENT_VARIABLES.md` vs code; `src/mlframe/feature_selection/filters/_hermite_fe_mi.py:147` | Five env switches the code reads are missing from the doc. A sixth, documented only in a source comment (`MLFRAME_FE_GPU_FUSE_MI=0`, "Default ON"), is read nowhere, and the fused kernel it refers to is never called by production code. |

### P10-01

Repro: `r = RFECV(estimator=lr, search_config=SearchConfig(max_refits=20)); r.set_params(max_refits=5); clone(r)`. `clone` builds `RFECV(max_refits=5, search_config=SearchConfig(max_refits=20))`. `__init__` then overwrites `max_refits` with 20, and sklearn's check that the constructor stored each parameter unchanged raises `RuntimeError`. Any `GridSearchCV` over a flat RFECV parameter that a passed config also sets fails this way. MRMR avoids it with `mrmr_set_params` and `invalidate_stale_mrmr_configs` (`_mrmr_config_dataclasses.py:262-319`); RFECV has neither.

### P10-02

Repro: `r.set_params(search_config=SearchConfig(max_refits=3))`, then `r.fit(...)`. `BaseEstimator.set_params` only assigns the `search_config` attribute. The overlay runs only inside `__init__`, so fit keeps using the old `max_refits`. Grid-searching over a config object therefore runs every candidate with the same settings. A `clone()` would apply it, so whether the config takes effect depends on whether the estimator was cloned.

### P10-03

The comment block above the parameter says "Default-on at 5 (... N=5 calibrated ...)", but the actual default is `monotonic_decline_patience: int = 7`. A user reading it expects stopping after 5 strictly worse iterations and gets 7. Separately, `_model_configs_behavior.py:191` sets 20 for the boosters and says it "mirrors the neural / sklearn-wrapper paths", which use different values. The same knob therefore has three documented values.

### P10-04

With a polars frame that has a string column, `np.asarray(X)` returns an object array, so `dtype.kind` is `"O"` and `X` is returned with +-inf still in it. The class docstring promises "inf -> NaN -> imputed" so the linear and MLP pipelines no longer depend on `fix_infinities`. For an all-float polars frame the inf replacement works, but the result is an ndarray rather than a frame, unlike the pandas path. Whether the suite ever hands polars to this pipeline step was not verified.

### P10-05

`self.inner.fit(...)` mutates the object the caller passed in. If one instance is reused or shared outside `clone`, a refit overwrites it, and the caller's original imputer or scaler is left fitted. This breaks the sklearn rule that fitted state lives only in trailing-underscore attributes.

### P10-06

`IdentityRegressor().fit(X, y)` leaves no `n_features_in_`, so sklearn's `check_estimator` and its `n_features_in_` checks fail on the regressor and classifier variants.

### P10-07

The code reads these switches, none of which appear in the doc: `MLFRAME_ODDS_COMBINE_BACKEND` (`calibration/_ktc_dispatch.py:46`), `MLFRAME_LOGICAL_CONSTRAINTS_BACKEND` (`inference/_ktc_dispatch.py:32`), `MLFRAME_COMPOSITE_CORR_BACKEND` and `MLFRAME_COMPOSITE_COLLINEAR_BACKEND` (`training/composite/discovery/_ktc_dispatch.py:60-61`), and `MLFRAME_FLEURET_PRANGE` (`_confirm_predictor_helpers.py:196`). In the other direction, every `MLFRAME_*` name in the doc appears in source. The source comment's `MLFRAME_FE_GPU_FUSE_MI=0` is read nowhere, and `_get_fused_mi_term` is called only by its test and is listed in the uncalled-functions meta-test allowlist, so setting that variable does nothing and the "Default ON" fusion is not active.

## Checked and clean

- MRMR's `set_params`/`clone` handling of its nested configs.
- RFECV re-runs its `__init__` validation in `fit` (`_fit.py:83-87`), so values set later through `set_params` are still checked.
- No mismatch between `__all__` and the names defined, in every `__init__.py` without a lazy `__getattr__` or star import.
- No `__init__` in calibration, preprocessing, inference, models, training/composite or training/strategies transforms or raises on a parameter (the scan only sees direct `self.x = ...` assignments; classes that store parameters via `store_params_in_object` were not covered).
- `EstimatorWithEarlyStopping` and `EarlyStoppingWrapper` return `self` from `fit` and set `n_features_in_`.

## Not covered

- The `train_mlframe_models_suite` config objects in depth (docstring vs field defaults beyond `monotonic_decline_patience`).
- `__all__` correctness in packages that use a lazy `__getattr__` or star imports, such as `training/__init__`.
- Each calibration class's `fit`/`predict` contract and whether `fit` is idempotent.
- `ShapProxiedFS`, `BorutaShap`, `StabilityMRMR` and the `functional_adapters` selectors.
- Pandas vs polars vs numpy output parity for MRMR and RFECV `transform`.
- Whether `predict` changes state anywhere in `inference/`.
