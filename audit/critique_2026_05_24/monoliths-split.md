# Monolith split plan — `mlframe/` files > 900 LOC (audit 2026-05-24)

NOTE: Persisted from inline agent output (Plan-type subagent ran without Write tool).

## Summary

Eighteen modules currently exceed the 900-LOC threshold; eleven exceed 1k. The dominant carve-friendly pattern in this codebase is already established (see `composite_transforms.py` re-exporting `_composite_transforms_linear` / `_nonlinear`, and `wrappers/_rfecv_fit.py` carved as a single bound method) — most siblings live in the same package directory with `_<domain>.py` naming and parents do bottom-of-file `from ._sibling import name1, name2  # noqa: F401`. Six of the eighteen are tight-coupled single-class monoliths (`_composite_target_estimator.py`, `neural/base.py`, `neural/recurrent.py`, `neural/ranker.py`, `boruta_shap.py`, `_target_distribution_analyzer.py`) where carving by method does not preserve `self` semantics cleanly — those need method-extraction-into-module-level-functions, not raw carve. The remaining twelve are mixed function collections that fit the sibling pattern directly. AST-audit gate (CLAUDE.md "Monolith split: AST-audit sibling for unresolved names BEFORE commit") is mandatory for every split.

## Per-file split plans

### `training/composite_transforms.py` — 1142 LOC

Already partially split (linear + nonlinear siblings exist). Residual surface: registry construction, simple transforms (diff/ratio/additive/median/y_quantile_clip/rolling_quantile_ratio), Pack-J unary adapters, Pack-K chain entries, and the naming helpers (`compose_target_name`, `is_composite_target_name`, `list_transforms`, `_TRANSFORMS_REGISTRY`).

| Sibling (new) | Lines from parent | Contains |
|---|---|---|
| `_composite_transforms_simple.py` | 145-352, 506-545 | diff/additive_residual/median_residual/y_quantile_clip/ratio/rolling_quantile_ratio fit/forward/inverse/domain + `_MEDIAN_RESIDUAL_N_BINS`, `_Y_QUANTILE_CLIP_*`, `_ROLLING_QUANTILE_DEFAULT_K` |
| `_composite_transforms_registry.py` | 595-1028 (adapter factory + `_TRANSFORMS_REGISTRY` dict) | `_make_unary_registry_adapter`, per-unary adapters, `_TRANSFORMS_REGISTRY` |
| `_composite_transforms_naming.py` | 1041-1129 | `TRANSFORM_NAME_SHORT`, `compose_target_name`, `_COMPOSITE_NAME_FRAGMENTS`, `is_composite_target_name`, `list_transforms`, `get_transform` |

Re-export contract (bottom of parent):
```python
from ._composite_transforms_simple import *  # noqa: F401, F403  -- explicit __all__ inside
from ._composite_transforms_registry import _TRANSFORMS_REGISTRY, _make_unary_registry_adapter  # noqa: F401
from ._composite_transforms_naming import (  # noqa: F401
    TRANSFORM_NAME_SHORT, compose_target_name, get_transform,
    is_composite_target_name, list_transforms,
)
```

Risks:
- `_TRANSFORMS_REGISTRY` references functions across SIX modules (`_composite_transforms_linear`, `_nonlinear`, `_simple`, `_registry`, plus this parent's `Transform` class). The registry module must import `Transform` from parent lazily inside the dict literal (or use a `register(name, ...)` builder that the simple/linear/nonlinear modules call themselves). Init-order: registry import must happen AFTER all four functional modules are loaded.
- `_make_unary_registry_adapter` consumes raw `cbrt_y_*` etc. from `composite_unary_transforms`; that import line stays at parent top (or moves into `_registry`).
- `compose_target_name` is called from many sites (verified by `composite_transforms` import-graph in `composite.py`); re-export at parent bottom is mandatory.

Sensor test (`tests/training/test_composite_transforms_split.py`):
```python
def test_composite_transforms_identity_preserved():
    from mlframe.training import composite_transforms as parent
    from mlframe.training import _composite_transforms_naming as naming
    assert parent.compose_target_name is naming.compose_target_name
    assert parent.is_composite_target_name is naming.is_composite_target_name
    assert parent.list_transforms() == sorted(parent._TRANSFORMS_REGISTRY)

def test_composite_transforms_facade_loc_budget():
    p = Path(parent.__file__); assert len(p.read_text().splitlines()) <= 600

def test_composite_transforms_smoke_fit_forward_inverse():
    t = parent.get_transform("linear_residual")
    y = np.array([1.,2.,3.,4.]); b = np.array([1.,2.,3.,4.])
    params = t.fit(y, b); T = t.forward(y, b, params); y2 = t.inverse(T, b, params)
    assert np.allclose(y, y2, atol=1e-9)
```

### `training/core/_phase_composite_post.py` — 1129 LOC

Four top-level symbols (`_LagPredictDeployableModel`, `recover_composite_y_scale_metrics`, `_run_suite_end_dummy_baselines_summary`, `run_composite_post_processing`); `run_composite_post_processing` alone is 819 LOC (311->1129). Natural carve by phase.

| Sibling (new) | Lines from parent | Contains |
|---|---|---|
| `_phase_composite_post_lag_predict.py` | 51-122 | `_LagPredictDeployableModel` class |
| `_phase_composite_post_y_scale.py` | 124-170 | `recover_composite_y_scale_metrics` |
| `_phase_composite_post_dummy_summary.py` | 172-310 | `_run_suite_end_dummy_baselines_summary` |
| `_phase_composite_post_runner.py` | 311-1129 | `run_composite_post_processing` (consider second-level carve of cross-target-ensemble loop into `_phase_composite_post_xt_ensemble.py` if function exceeds 500 LOC after the first carve) |

Re-export contract:
```python
from ._phase_composite_post_lag_predict import _LagPredictDeployableModel  # noqa: F401
from ._phase_composite_post_y_scale import recover_composite_y_scale_metrics  # noqa: F401
from ._phase_composite_post_dummy_summary import _run_suite_end_dummy_baselines_summary  # noqa: F401
from ._phase_composite_post_runner import run_composite_post_processing  # noqa: F401
```

Risks:
- `run_composite_post_processing` imports `_LagPredictDeployableModel`, `recover_composite_y_scale_metrics`, AND `_run_suite_end_dummy_baselines_summary`. Its sibling needs explicit `from . import _phase_composite_post as _parent` lazy-import OR direct `from ._phase_composite_post_lag_predict import _LagPredictDeployableModel` siblings-to-siblings (preferred — no cycle).
- `_WATCHDOG_RELATIVE_THRESHOLD`, `_DEFAULT_OOF_RANDOM_STATE`, `_PROB_NORM_EPS` constants — keep at parent top + re-export to runners that need them.
- AST audit critical: per CLAUDE.md, this is exactly the file class that surfaces NameError-via-broad-except WARN spam.

Sensor test: identity check (`parent.run_composite_post_processing is runner.run_composite_post_processing`), facade LOC <= 100, plus a smoke test invoking `run_composite_post_processing` with a minimal `TrainingContext` mock — import alone is insufficient per CLAUDE.md "split sensor must exercise at least one code path".

### `metrics/core.py` — 1064 LOC

Mixed numba kernels + scoring helpers + warmup. Natural carve by function family.

| Sibling (new) | Lines from parent | Contains |
|---|---|---|
| `_core_numba_warmup.py` | 39-432 (warmup + numba prewarm) | `numba_warmup`, `_assert_numba_nogil_active`, `prewarm_numba_cache`, `_prewarm_numba_cache_body` |
| `_core_cb_logits.py` | 434-531 | `_cb_logits_to_probs_binary_*`, `cb_logits_to_probs_binary`, `_cb_logits_to_probs_multiclass_*`, `cb_logits_to_probs_multiclass` |
| `_core_auc_brier.py` | 533-1042 (selected: fast_roc_auc, fast_aucs, fast_brier, brier_and_precision, scorer factory) | `fast_roc_auc`, `fast_numba_auc_nonw`, `fast_aucs`, `fast_numba_aucs`, `_fast_brier_score_loss_*`, `fast_brier_score_loss`, `brier_and_precision_score`, `make_brier_precision_scorer` |
| `_core_precision_mape.py` | 598-815 (precision report + MAPE kernels) | `fast_precision`, `fast_classification_report`, `_max_abs_pct_error_kernel*`, `maximum_absolute_percentage_error` |

Re-export at parent bottom:
```python
from ._core_numba_warmup import numba_warmup, prewarm_numba_cache  # noqa: F401
from ._core_cb_logits import (  # noqa: F401
    cb_logits_to_probs_binary, cb_logits_to_probs_multiclass,
)
from ._core_auc_brier import (  # noqa: F401
    fast_roc_auc, fast_aucs, fast_brier_score_loss,
    brier_and_precision_score, make_brier_precision_scorer,
)
from ._core_precision_mape import (  # noqa: F401
    fast_precision, fast_classification_report,
    maximum_absolute_percentage_error,
)
```

Risks:
- Heavy external surface: `compute_probabilistic_multiclass_error`, `robust_mlperf_metric`, `ICE` are imported from `metrics.core` by `training/helpers.py:43-47` (verified above) AND many other modules. Whichever sibling owns each name MUST be re-exported at parent — easy NameError trap.
- numba JIT decorators on these functions: keep `cache=True` paths intact across the carve so `.pyc` cache doesn't get re-invalidated unnecessarily.

Sensor test: identity preserved, facade LOC <= 150, smoke: `fast_roc_auc(np.array([0,1,0,1]), np.array([0.1,0.9,0.2,0.8]))` returns a finite float.

### `training/neural/base.py` — 1057 LOC

**Carve-pattern WARNING**: `PytorchLightningEstimator` (291->845, ~555 LOC) is tight-coupled single class. Method extraction into module-level functions taking `self` as first arg works only if the methods don't share much short-name local state. The remaining surface IS carve-friendly:

| Sibling (new) | Lines from parent | Contains |
|---|---|---|
| `_base_logging.py` | 26-186 | `_LightningRankZeroNoiseFilter`, `suppress_lightning_workers_warning`, `_rmse_metric`, `MetricSpec` |
| `_base_tensor_helpers.py` | 205-285 | `custom_collate_fn`, `to_tensor_any`, `to_numpy_safe`, `_ensure_numpy` |
| `_base_callbacks.py` | 893-1057 | `NetworkGraphLoggingCallback`, `AggregatingValidationCallback`, `ValLossDivergenceCallback`, `BestEpochModelCheckpoint`, `PeriodicLearningRateFinder` |
| (keep) `PytorchLightningEstimator` + subclasses in parent | 286-892 | — |

Re-export contract:
```python
from ._base_logging import (  # noqa: F401
    _LightningRankZeroNoiseFilter, suppress_lightning_workers_warning, MetricSpec,
)
from ._base_tensor_helpers import (  # noqa: F401
    custom_collate_fn, to_tensor_any, to_numpy_safe,
)
from ._base_callbacks import (  # noqa: F401
    NetworkGraphLoggingCallback, AggregatingValidationCallback,
    ValLossDivergenceCallback, BestEpochModelCheckpoint, PeriodicLearningRateFinder,
)
```

Risks:
- `PytorchLightningEstimator` references `to_tensor_any`, `to_numpy_safe`, `MetricSpec`, callbacks inside `_fit_common` / `_predict_raw`. Sibling import at parent top is fine (no cycle — siblings only consume torch/numpy/lightning).
- `MetricSpec` extends pydantic `BaseModel`; if a downstream test does `isinstance(x, MetricSpec)`, the carve must preserve class identity (re-export the same class object, NEVER redefine).

Alternative for the 555-LOC class: leave the class in parent; the post-carve parent drops to ~580 LOC which is acceptable. No further refactor needed.

Sensor: identity (`parent.MetricSpec is _base_logging.MetricSpec`), facade <= 700 LOC, smoke = construct `PytorchLightningRegressor` instance.

### `training/core/_setup_helpers.py` — 1047 LOC

Mixed pipeline-disk-cache + outlier detection + multiple setup builders. The biggest target is `_apply_outlier_detection_global` (249->469, ~220 LOC) and `_build_pre_pipelines` (538->673, ~135 LOC).

| Sibling (new) | Lines from parent | Contains |
|---|---|---|
| `_setup_helpers_pipeline_cache.py` | 53-234 | `_PIPELINE_JSON_*` globals, `_pipeline_disk_cache_path`, `_pipeline_disk_cache_version_tag`, `_load_pipeline_disk_cache_into_memory`, `_persist_pipeline_disk_cache`, `_PolarsDsPipelineJsonProxy`, `_polars_ds_pipeline_from_json` |
| `_setup_helpers_outliers.py` | 249-468 | `_apply_outlier_detection_global` |
| `_setup_helpers_pre_pipelines.py` | 538-673 | `_build_pre_pipelines` (uses lazy MRMR import — keep that pattern) |
| `_setup_helpers_metadata.py` | 852-947 | `_create_initial_metadata`, `_initialize_training_defaults`, `_finalize_and_save_metadata` |

Re-export everything cited in tests / sibling modules. Risks:
- `_finalize_and_save_metadata` consumes `_PIPELINE_JSON_DISK_CACHE_*` and `_polars_ds_pipeline_from_json` — the metadata sibling must import from `_setup_helpers_pipeline_cache`, not from the parent (avoid cycle).
- `_build_pre_pipelines` does a deferred MRMR import (CLAUDE.md mentions ~10-25s startup cost) — preserve that pattern verbatim in the sibling.
- `if TYPE_CHECKING` ring imports for `TrainingContext`, `PreprocessingConfig` etc. — copy these into each sibling.

Sensor: identity + smoke `_pipeline_disk_cache_path()` returns a string; facade <= 350 LOC.

### `training/_target_distribution_analyzer.py` — 1017 LOC

Two big public functions (`analyze_target_distribution` 384->654 = 270 LOC, `analyze_feature_distribution` 846->1017 = 171 LOC) + many helpers + two dataclasses.

| Sibling (new) | Lines from parent | Contains |
|---|---|---|
| `_target_distribution_analyzer_stats.py` | 132-282 | `_excess_kurtosis`, `_skewness`, `_lag1_autocorr`, `_lag_autocorr`, `_max_abs_lag_autocorr`, `_lag1_autocorr_grouped`, `_check_within_group_ordering` |
| `_target_distribution_analyzer_modes.py` | 284-371 | `_detect_multi_modal`, `_within_between_group_variance_ratio`, `_classify_target_type` |
| `_target_distribution_analyzer_target_fn.py` | 384-654 | `analyze_target_distribution` |
| `_target_distribution_analyzer_features.py` | 655-1017 | `FeatureDistributionReport`, `_pairwise_redundant_features`, `_normalise_X`, `analyze_feature_distribution` |

Risks: `analyze_target_distribution` calls every helper in `_stats` and `_modes` — sibling-to-sibling top-level imports work. `TargetDistributionReport` dataclass stays at parent top (referenced by tests as `from mlframe.training._target_distribution_analyzer import TargetDistributionReport`).

Sensor: identity for the two `analyze_*` functions, facade <= 250 LOC, smoke = call `analyze_target_distribution(np.random.randn(1000))`.

### `feature_selection/wrappers/_rfecv_fit.py` — 998 LOC

**Carve-pattern PARTIALLY APPLIES**: this file is already a single carved function (`fit`, 68->998, ~930 LOC). It is a single monolithic method bound onto `RFECV`. The standard carve doesn't help further — the only path is to extract intra-method phases into module-level helpers in the same package and have `fit` call them. Phases observable from code shape (the file already has helper imports for `_sanitize_X_inputs`, `_resolve_cv_and_val_cv`, etc., so the pattern is established):

| Sibling (new) | Source slice | Contains |
|---|---|---|
| `_rfecv_fit_init.py` | sample_weight validation + polars-time-series hint + early `_sanitize_X_inputs` (lines 68-260 approx) | `_init_fit_state(self, X, y, groups, sample_weight, fit_params)` returning a `FitState` dataclass |
| `_rfecv_fit_cv_loop.py` | the `_eval_fold` closure (line 560) and its surrounding cv-fold body | `_run_outer_cv_loop(self, state, ...)` |
| `_rfecv_fit_iter.py` | feature-elimination iteration around the cv loop | `_run_elimination_iter(self, state, ...)` |

Re-export contract: identical — bind `fit` onto `RFECV` at parent bottom. The new helpers are NOT re-exported (they're internal-only).

Risks:
- Single-method monoliths often share many closure-captured locals; lifting them to module-level functions requires passing 10-15 locals as an explicit `FitState` dataclass. That is a real refactor (not a pure carve) and falls outside "behavioural-only" constraint UNLESS each lifted phase is tested with both the original and the new layout producing identical outputs on a fixed seed.
- `_eval_fold` is a closure over `current_features`, `scores`, `nfold`, etc. — extracting it makes those args explicit; behavioural-equivalence test is mandatory.

**Alternative recommendation**: rather than rush this split now, prioritise the other carve-friendly files first. Add a regression sensor that asserts `_rfecv_fit.py` LOC stays <= 1100 to prevent further growth, and treat the carve as a separate dedicated PR with full RFECV biz_value coverage.

Sensor (if carved): identity (`RFECV.fit is _rfecv_fit.fit`), smoke = full RFECV fit on tiny synthetic with both `n_features_selection_rule="best"` and `"smallest_within_threshold"`.

### `training/_composite_target_estimator.py` — 998 LOC

**Carve-pattern DOES NOT APPLY cleanly**: single class `CompositeTargetEstimator` (55->998, ~940 LOC). Methods share `self.fitted_params_`, `self._transform`, `self._base_column`, and many runtime-stats fields. Standalone carve would degrade to "every method takes `self` as arg" which is just procedural rewrite without modularity gain.

Better alternative — extract the FOUR methods that don't share much state and can become module-level functions:

| Sibling (new) | Source method | Contains |
|---|---|---|
| `_composite_target_estimator_utils.py` | `_subset_rows`, `_drop_columns`, `_require_fitted`, `feature_importances_`, `coef_`, `intercept_`, `get_booster`, `booster_`, `n_features_in_` (lines 817-872) | static / property accessors that don't mutate self |
| `_composite_target_estimator_update.py` | `update`, `get_buffer_state` (lines 918-1000) | streaming-update path (already isolated semantically) |
| `_composite_target_estimator_predict.py` | `_predict_unclipped`, `predict_pre_clip`, `predict`, `predict_quantile` (lines 533-816) | predict family |

The remaining `__init__`, `from_fitted_inner`, `_resolve_base_columns`, `_extract_base_for_transform`, `fit` stays in parent.

Re-export at parent bottom binds the carved methods back onto the class via assignment:
```python
from . import _composite_target_estimator_predict as _pred
from . import _composite_target_estimator_update as _upd
CompositeTargetEstimator.predict = _pred.predict
CompositeTargetEstimator.predict_pre_clip = _pred.predict_pre_clip
CompositeTargetEstimator.predict_quantile = _pred.predict_quantile
CompositeTargetEstimator._predict_unclipped = _pred._predict_unclipped
CompositeTargetEstimator.update = _upd.update
CompositeTargetEstimator.get_buffer_state = _upd.get_buffer_state
```
(Same pattern as `RFECV.fit`.)

Risks:
- `isinstance` check tripwire (CLAUDE.md): consumers do `isinstance(model, CompositeTargetEstimator)`. Method-rebinding does not change class identity — safe.
- `_predict_unclipped` returns `tuple[np.ndarray, int, dict]` and is called by both `predict` and `predict_pre_clip`; co-locate all three in the same sibling.

Sensor: identity (`CompositeTargetEstimator.predict is _pred.predict`), full fit + predict round-trip, facade LOC <= 700.

### `training/helpers.py` — 993 LOC

Three top-level symbols visible (`parse_catboost_devices`, `get_training_configs` 156->912 = ~756 LOC, `compute_cb_text_processing`). The dominant mass is one function.

| Sibling (new) | Source | Contains |
|---|---|---|
| `_helpers_cb_devices.py` | 94-155 | `parse_catboost_devices` |
| `_helpers_training_configs.py` | 156-912 | `get_training_configs` |
| `_helpers_cb_text.py` | 913-993 | `compute_cb_text_processing` |

Re-export at parent bottom. Risks:
- `get_training_configs` does a deferred `import torch + mlframe.lightninglib` (line 28 comment); preserve that lazy import inside the sibling, not at sibling top.
- `compute_cb_text_processing` is imported by `feature_selection/wrappers/_rfecv_fit.py:44`. Re-export at parent bottom is mandatory.

Sensor: identity for the three functions, facade <= 100 LOC, smoke = `get_training_configs(iterations=10)` returns a dict.

### `training/neural/recurrent.py` — 963 LOC

Three classes (`_RecurrentWrapperBase` 158->553 ~395 LOC, `RecurrentClassifierWrapper` 554->740 ~186 LOC, `RecurrentRegressorWrapper` 741->893 ~152 LOC) + `_monitor_mode` helper + `extract_sequences` / `extract_sequences_chunked` (894->963 = end).

Carve-pattern applies cleanly because subclasses are independent:

| Sibling (new) | Source | Contains |
|---|---|---|
| `_recurrent_classifier.py` | 554-740 | `RecurrentClassifierWrapper` |
| `_recurrent_regressor.py` | 741-893 | `RecurrentRegressorWrapper` |
| `_recurrent_sequence_extract.py` | 894-963 | `extract_sequences`, `extract_sequences_chunked` |

Parent keeps `_monitor_mode` + `_RecurrentWrapperBase` (the abstract base both subclasses extend).

Risks:
- Subclasses inherit from `_RecurrentWrapperBase` — they MUST top-import from parent: `from .recurrent import _RecurrentWrapperBase`. Parent re-exports the subclasses at bottom AFTER they are defined in siblings. No cycle (subclass module imports parent base, parent module imports finished subclass back) — but use late import at parent bottom to avoid partial-module-import error at sibling load time.
- Alternative without cycle risk: move `_RecurrentWrapperBase` to a third sibling `_recurrent_base.py`, then both classifier and regressor siblings (and the parent re-export) import from it.

Sensor: identity, facade <= 600 LOC, smoke fit+predict on a small sequence.

### `feature_selection/boruta_shap.py` — 952 LOC

Single class `BorutaShap` (52->910, ~858 LOC) + `load_data` helper. **Carve-pattern DOES NOT APPLY** directly — see `_composite_target_estimator.py` for the recommended alternative pattern (carve methods to module-level functions, rebind at bottom).

Method clusters identified:
- Setup/validate: `check_model`, `_ordinal_encode_object_cols_inplace`, `check_X`, `missing_values_y`, `check_missing_values`, `Check_if_chose_train_or_test_and_train_model` — lines 126-297
- Train/feature-importance: `Train_model`, `create_importance_history`, `update_importance_history`, `store_feature_importance`, `results_to_csv`, `remove_features_if_rejected`, `feature_importance` — lines 298-635
- Statistics helpers: `binomial_H0_test`, `bonferoni_corrections`, `calculate_Zscore`, `average_of_list`, `flatten_list`, `find_index_of_true_in_array`, `symetric_difference_between_two_arrays`, `get_5_percent`, `get_5_percent_splits`, `find_sample` — lines 482-712
- Test/decide: `test_features`, `TentativeRoughFix`, `Subset` — lines 714-796
- Plotting: `plot`, `box_plot`, `create_list`, `filter_data`, `hasNumbers`, `check_if_which_features_is_correct`, `create_mapping_of_features_to_attribute`, `to_dictionary` — lines 797-910

| Sibling (new) | Cluster |
|---|---|
| `_boruta_shap_stats.py` | statistics helpers (most are `@staticmethod`-eligible) |
| `_boruta_shap_plot.py` | plotting cluster |
| `_boruta_shap_train.py` | train/feature-importance cluster |

Rebind methods onto `BorutaShap` at parent bottom. `_boruta_shap_fit_explain.py` sibling already exists in the directory — extend that pattern.

Risks:
- Many helpers are `@staticmethod` already (e.g. `average_of_list`, `flatten_list`); these are TRIVIAL to lift to module-level — no `self` capture.
- Plot/seaborn import is lazy; keep that.

Sensor: identity of every rebound method, facade <= 500 LOC, smoke = fit on 200-row synthetic.

### `training/target_temporal_audit.py` — 949 LOC

Mixed: `coerce_timestamps_for_audit` (82-230, ~148 LOC), `TimeBin` + `TemporalAuditResult` dataclasses, `_pick_granularity` (406-473), polars aggregation helpers (474-613), pandas aggregation (570-614), `audit_target_over_time` (644-755, ~111 LOC), `audit_targets_over_time` (756-925, ~169 LOC), `format_temporal_audit_report` (926-end).

| Sibling (new) | Source | Contains |
|---|---|---|
| `_target_temporal_audit_coerce.py` | 59-229 | `_import_ruptures`, `coerce_timestamps_for_audit` |
| `_target_temporal_audit_aggregate.py` | 406-613 | `_pick_granularity`, `_polars_rate_expr`, `_aggregate_by_time_polars`, `_aggregate_by_time_polars_multi`, `_aggregate_by_time_pandas`, `_format_bin_label` |
| `_target_temporal_audit_runners.py` | 644-925 | `audit_target_over_time`, `audit_targets_over_time` |
| `_target_temporal_audit_format.py` | 926-949 | `format_temporal_audit_report` |

Risks: `audit_target_over_time` calls `coerce_timestamps_for_audit` + `_aggregate_by_time_polars` + `_pick_granularity` — sibling-to-sibling direct import. `TimeBin` + `TemporalAuditResult` dataclasses stay at parent top (referenced as `audit.TimeBin` from callers).

Sensor: identity for runners + format, facade <= 350 LOC, smoke = `audit_target_over_time` on a small pandas df with a timestamp column.

### `training/core/_phase_helpers.py` — 948 LOC

Several NamedTuples + ~12 phase-helper functions. The biggest is `_phase_pandas_conversion_and_cat_prep` (397->614, ~217 LOC) and `_phase_load_and_preprocess` (770->852, ~82 LOC).

| Sibling (new) | Source | Contains |
|---|---|---|
| `_phase_helpers_named_tuples.py` | 61-143 | `_models_need_pandas_cat_prep`, `TrainValTestSplitResult`, `FitPipelineResult`, `PolarsCategoricalFixesResult` |
| `_phase_helpers_plot_style.py` | 144-221 | `_apply_plot_style_overrides` |
| `_phase_helpers_pandas_cat.py` | 397-614 | `_phase_pandas_conversion_and_cat_prep` |
| `_phase_helpers_load.py` | 770-888 | `_phase_load_and_preprocess`, `_build_suite_common_params_dict`, `_maybe_dispatch_to_ltr_ranker_suite` |

Parent retains: `_defensive_copy_and_expand_multilabel_regression`, `_init_composite_discovery_metadata`, `_phase_global_outlier_detection`, `_log_cardinality_and_drift_snapshot` (these are mid-sized and tightly used together).

Risks: existing siblings `_phase_helpers_fit_pipeline.py` and `_phase_helpers_fit_split.py` already follow this pattern — copy their conventions exactly.

Sensor: identity, facade <= 600 LOC, smoke = each carved function with mocked context.

### `training/baseline_diagnostics.py` — 942 LOC

Single class `BaselineDiagnostics` (236->898, ~662 LOC) + 3 dataclasses + 4 helpers + `format_baseline_diagnostics_report`.

Same alternative-pattern issue as `boruta_shap.py` / `_composite_target_estimator.py`. Method-level carve:

| Sibling (new) | Source method | Contains |
|---|---|---|
| `_baseline_diagnostics_quick_model.py` | `_make_quick_model`, `_fit_quick_and_score` (435-571) | quick-model factory + scoring |
| `_baseline_diagnostics_ablation.py` | `_run_ablation` (572-676) | ablation loop |
| `_baseline_diagnostics_init_score.py` | `_fit_init_score_baseline` (677-827) | init-score baseline path |
| `_baseline_diagnostics_recommend.py` | `_build_recommendation` (828-893) | recommendation builder |

Parent keeps dataclasses + `BaselineDiagnostics.__init__` + `fit_and_report` + `_skipped` + `_sample` + the 3 small helpers (`_coerce_to_pandas`, `_select_metric`, `_compute_metric`, `_delta_pct`) + `format_baseline_diagnostics_report`.

Rebind methods onto `BaselineDiagnostics` at parent bottom (same pattern as `RFECV.fit`).

Risks: methods share `self.config`, `self._report`, `self._cache_dir`; carve preserves `self` via first-arg passing.

Sensor: identity of rebound methods, facade <= 450 LOC, smoke = `BaselineDiagnostics(cfg).fit_and_report(X, y, target_type="regression")` on a tiny synthetic.

### `training/train_eval.py` — 941 LOC

Seven top-level functions; `select_target` (227->602, ~375 LOC) and `process_model` (690->941, ~251 LOC) dominate.

| Sibling (new) | Source | Contains |
|---|---|---|
| `_train_eval_validate.py` | 48-150 | `_extract_polars_cat_columns`, `_validate_cached_model_schema`, `_n_classes_from_target` |
| `_train_eval_optimize.py` | 176-226 | `optimize_model_for_storage` |
| `_train_eval_select_target.py` | 227-602 | `select_target` |
| `_train_eval_process_model.py` | 603-941 | `_call_train_evaluate_with_configs`, `process_model` |

Re-export contract: `select_target` is the public name imported broadly across the suite — re-export at parent bottom is mandatory.

Risks: `process_model` calls into `_call_train_evaluate_with_configs` which calls `select_target` — keep these in `_process_model` sibling and use sibling-to-sibling import `from ._train_eval_select_target import select_target`. NO parent re-import inside the sibling (avoids the AST-audit NameError trap).

Sensor: identity, facade <= 80 LOC, smoke = `select_target` with the smallest fixture from existing `tests/training/`.

### `training/neural/flat.py` — 927 LOC

Four top-level: `MLPNeuronsByLayerArchitecture` enum, `get_valid_num_groups`, `generate_mlp` (46->417, ~371 LOC), `MLPTorchModel` (418->927, ~509 LOC).

| Sibling (new) | Source | Contains |
|---|---|---|
| `_flat_arch_enum.py` | 31-45 | `MLPNeuronsByLayerArchitecture`, `get_valid_num_groups` |
| `_flat_generate_mlp.py` | 46-417 | `generate_mlp` |
| `_flat_torch_module.py` | 418-927 | `MLPTorchModel` (LightningModule subclass) |

Risks:
- `MLPTorchModel` calls `generate_mlp` inside `__init__`/`configure_optimizers` — sibling-to-sibling import.
- Lightning subclass identity: `isinstance(net, MLPTorchModel)` must work — re-export at parent.

Sensor: identity, facade <= 50 LOC, smoke = construct `MLPTorchModel(n_inputs=4, n_outputs=1)`.

### `training/extractors.py` — 940 LOC

Mix of helpers (42-381) + Protocol + two classes (`FeaturesAndTargetsExtractor` 405-653, `SimpleFeaturesAndTargetsExtractor` 654-940). Two classes are independent → clean carve.

| Sibling (new) | Source | Contains |
|---|---|---|
| `_extractors_dtype_helpers.py` | 42-227 | `get_dataframe_info`, `_smallest_safe_int_dtype`, `_safe_int_cast_numpy`, `intize_targets`, `get_sample_weights_by_recency` |
| `_extractors_showcase.py` | 228-381 | `showcase_features_and_targets` |
| `_extractors_simple.py` | 654-940 | `SimpleFeaturesAndTargetsExtractor` |

Parent keeps the `FeaturesAndTargetsExtractorProtocol` + base `FeaturesAndTargetsExtractor` (subclassed by Simple).

Risks: `SimpleFeaturesAndTargetsExtractor` extends `FeaturesAndTargetsExtractor`; sibling imports base from parent.

Sensor: identity of every re-export + `isinstance(simple_inst, FeaturesAndTargetsExtractor)` returns True, facade <= 500 LOC.

### `training/neural/ranker.py` — 919 LOC

Multiple classes: `GroupBatchSampler` (230-336), `_RankerDataset` (338-475), `MLPRankerLightningModule` (530-588), `_SamplerSetEpochCallback` (595-617), `MLPRanker` (619-919, ~300 LOC).

| Sibling (new) | Source | Contains |
|---|---|---|
| `_ranker_losses.py` | 56-228 | `_ranknet_pair_cache_clear`, `ranknet_pairwise_loss`, `_ranknet_loss_precomputed_core`, `ranknet_pairwise_loss_precomputed`, `listnet_top1_loss` |
| `_ranker_dataset.py` | 230-503 | `GroupBatchSampler`, `_RankerDataset`, `_ranker_passthrough_collate`, `_import_lightning` |
| `_ranker_lightning_module.py` | 530-617 | `MLPRankerLightningModule`, `_make_lightning_module`, `_SamplerSetEpochCallback` |

Parent keeps `MLPRanker` (the public estimator).

Risks: `MLPRanker.fit` uses `GroupBatchSampler`, `_RankerDataset`, `_make_lightning_module`, `_SamplerSetEpochCallback` — sibling-to-sibling imports.

Sensor: identity, facade <= 400 LOC, smoke = MLPRanker fit on n=200 / 20-group synthetic.

## Превентивный список (900-1000 LOC, рядом)

Files that have not yet crossed >1k but are close — bake the carve plan now to prevent next quarter's spike:

| Path | LOC | Pre-emptive carve hint |
|---|---|---|
| `training/core/_phase_train_one_target_body.py` | 920 | check function inventory and split body of largest function into `_phase_train_one_target_body_pre.py` / `_post.py` |
| `training/core/_main_train_suite.py` | 911 | split into `_main_train_suite_setup.py` / `_main_train_suite_dispatch.py` / `_main_train_suite_finalize.py` |
| `training/core/_phase_train_one_target.py` | 883 | mirror `_phase_train_one_target_body` carve plan |
| `training/composite_discovery.py` | 878 | likely splits cleanly along discovery phases (screening / fit / select) |
| `training/strategies.py` | 869 | per-estimator strategy classes — natural carve |
| `training/trainer.py` | 832 | central; rev-share with the team before carving |
| `feature_selection/filters/mrmr.py` | (check; likely >900 with FE polynom support added 2026-04-22) | already partially split (`_mrmr_fit_impl.py`, `_mrmr_fingerprints.py`, etc.) — extend the pattern |

Each of these should get a sensor today asserting `len(file.read_text().splitlines()) <= 1100` so future PRs that push them over are flagged in CI.

## Приоритезация (top-3-5 to split first)

Ranked by leverage (size × call-site density × cycle-risk):

1. **`training/core/_phase_composite_post.py` (1129 LOC)** — single 819-LOC function `run_composite_post_processing` is the highest cyclomatic complexity in the codebase by an order of magnitude; carve is cleanly mechanical AND eliminates the most likely NameError-via-broad-except trap (CLAUDE.md cites the WARN spam pattern from waves 92-107 here as the canonical regression). Direct biz_value coverage exists.
2. **`training/composite_transforms.py` (1142 LOC)** — already partially split; the remaining work (simple-transforms + registry + naming carve) is the lowest-risk-per-line in the list because the imports are already organised and `Transform` is a frozen dataclass with stable identity. Doing it now closes the carve pattern cleanly for future Pack-L additions.
3. **`metrics/core.py` (1064 LOC)** — wide downstream surface (`training/helpers.py`, RFECV wrapper, every scoring path) → carve unlocks fast small-targeted-pytest cycles for AUC / Brier work, which currently re-imports the whole numba warmup tree. Identity preservation is the only real risk.
4. **`training/core/_setup_helpers.py` (1047 LOC)** — the pipeline-disk-cache section (~180 LOC) is purely IO + cache logic with no overlap with the rest; carving it first reduces parent to ~870 LOC and removes the heaviest TYPE_CHECKING import burden from suite startup.
5. **`training/_target_distribution_analyzer.py` (1017 LOC)** — two large public functions plus self-contained helper clusters; carve is one-PR-sized and unblocks the upcoming feature-distribution add-ons.

The four "tight-coupled single-class" files (`neural/base.py`, `neural/recurrent.py`, `neural/ranker.py`, `_composite_target_estimator.py`, `boruta_shap.py`, `baseline_diagnostics.py`) should be sequenced AFTER the above five — they require the alternative method-rebinding pattern which carries higher behavioural-equivalence risk and warrants its own dedicated reviewer pass per class.

## Не нашёл (files that do not fit the carve pattern)

These four monoliths require a non-carve approach. Documenting explicitly per the requirement "richness-first, behavioral only":

1. **`feature_selection/wrappers/_rfecv_fit.py` (998 LOC)** — single carved function `fit` bound onto `RFECV`. Further carve requires intra-method phase extraction into a `FitState` dataclass and explicit phase functions, which is a real refactor, not a behavioural-only carve. Recommendation: ship an LOC-cap sensor (1100), defer the refactor to a dedicated PR with full RFECV biz_value coverage (`test_biz_val_wrappers_rfecv.py`) verifying byte-identical fold scores pre/post.
2. **`training/_composite_target_estimator.py` (998 LOC)** — single class with shared `self` state. The method-rebinding alternative (predict + update siblings) DOES work; documented in its section above. NOT pure carve, but accepted as the established codebase convention (see `RFECV.fit`).
3. **`feature_selection/boruta_shap.py` (952 LOC)** — single class. Same alternative pattern as above. Bonus risk: many `@staticmethod` helpers that are trivially extractable; do those first to reduce class LOC quickly.
4. **`training/neural/base.py` (1057 LOC)** — `PytorchLightningEstimator` is 555 LOC of tightly-coupled training-loop scaffolding. The carve I proposed extracts ONLY callbacks + tensor helpers + logging filter; the class itself stays in parent and parent drops to ~580 LOC, which is acceptable. Do NOT split the class internals.

For files (2)-(4), the rule "не предлагать удалять функционал ради LOC budget" applies — the LOC count is a symptom of legitimate richness (many params, many subclass-aware behaviours); the right answer is the method-rebinding pattern, not feature deletion.
