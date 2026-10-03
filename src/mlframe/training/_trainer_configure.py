"""``configure_training_params`` carved out of ``mlframe.training.trainer``.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.trainer import configure_training_params``
resolves transparently.
"""
from __future__ import annotations

import copy
import threading
from timeit import default_timer as timer
from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from ._configs_base import TargetTypes
    from ._model_configs import LinearModelConfig, MultilabelDispatchConfig

import numpy as np
import pandas as pd

# Heavy optional deps: defer failures to first actual use so `import mlframe.training` stays cheap and does not crash when a given backend is not installed.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from mlframe.metrics.core import fast_mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

# Optional model backends: lazy/tolerant of missing deps.
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostRegressor = CatBoostClassifier = None
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore[assignment,misc]
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore[assignment,misc]

from ._predict_guards import _CB_VAL_POOL_CACHE  # noqa: F401
from mlframe.training.pipeline.shared import (  # noqa: F401
    PRE_PIPELINE_CACHE as _PRE_PIPELINE_CACHE,
    PRE_PIPELINE_CACHE_LOCK as _PRE_PIPELINE_CACHE_LOCK,
    PRE_PIPELINE_CACHE_MAX as _PRE_PIPELINE_CACHE_MAX,
    apply_pre_pipeline_transforms as _apply_pre_pipeline_transforms,
    extract_feature_selector as _extract_feature_selector,
    is_fitted as _is_fitted,
    multilabel_target_to_1d_for_supervised_encoders as _multilabel_target_to_1d_for_supervised_encoders,
    passthrough_cols_fit_transform as _passthrough_cols_fit_transform,
    pipeline_signature_for_cache as _pipeline_signature_for_cache,
    pre_pipeline_cache_clear as _pre_pipeline_cache_clear,
    pre_pipeline_cache_get as _pre_pipeline_cache_get,
    pre_pipeline_cache_set as _pre_pipeline_cache_set,
    prepare_test_split as _prepare_test_split,
)
from mlframe.training.cb.shared import (  # noqa: F401
    cached_gpu_info as _cached_gpu_info,
    maybe_get_or_build_cb_pool as _maybe_get_or_build_cb_pool,
    maybe_rewrite_eval_set_as_cb_pool as _maybe_rewrite_eval_set_as_cb_pool,
    polars_df_has_null_in_categorical as _polars_df_has_null_in_categorical,
    polars_fill_null_in_categorical as _polars_fill_null_in_categorical,
    polars_nullable_categorical_cols as _polars_nullable_categorical_cols,
    polars_schema_diagnostic as _polars_schema_diagnostic,
    predict_with_fallback as _predict_with_fallback,
)
from ._eval_helpers import (  # noqa: F401
    _align_xgb_cat_categories, _append_split_rate_suffix,
    _compute_split_metrics, _decategorise_float_cat_columns,
    _filter_categorical_features, run_confidence_analysis,
)
from ._training_loop import (  # noqa: F401
    _SigmoidAdapter, _PostHocCalibratedModel,
    _PostHocMultiCalibratedModel, _PerClassIsotonicCalibrator,
    _maybe_apply_posthoc_calibration, _train_model_with_fallback,
)
from ._data_helpers import (  # noqa: F401
    _setup_eval_set, _setup_early_stopping_callback,
)
from ._model_factories import (  # noqa: F401
    GPU_VRAM_SAFE_FREE_LIMIT_GB, GPU_VRAM_SAFE_SATURATION_LIMIT,
    MODELS_SUBDIR, USE_LGB_DATASET_REUSE_SHIM, USE_XGB_DMATRIX_REUSE_SHIM,
    _get_neural_components,
    _lgb_classifier_cls as _lgb_classifier_cls_factory,
    _lgb_regressor_cls as _lgb_regressor_cls_factory,
    _patch_dataset_constructors_with_logging,
    _patch_lgb_feature_names_in_setter,
    _xgb_classifier_cls as _xgb_classifier_cls_factory,
    _xgb_regressor_cls as _xgb_regressor_cls_factory,
)

# Optional model backends mirrored from parent: defaults to None when the
# library is not installed; downstream branches gate on that.
try:
    from ngboost import NGBClassifier, NGBRegressor
except ImportError:
    NGBClassifier = None
    NGBRegressor = None


# A5#4/#16 session-level memo for ``get_training_configs``. The function is called
# twice per ``select_target`` invocation (CPU + GPU), and ``select_target`` runs
# once per (target, pre_pipeline, model) in the suite -- ten targets x three
# pre_pipelines x five models = 300 calls to ``get_training_configs`` per suite.
# Most of those calls share identical ``config_params`` content because the
# suite-level ``hyperparams_config.model_dump()`` is computed once and never
# changes between targets; only ``subgroups`` may differ when per-target OD
# filtering changes ``train_idx`` / ``val_idx``.
#
# The cache is capped via FIFO eviction at 16 entries to bound memory; in
# practice 2-4 entries cover the canonical suite (CPU + GPU x at most a few
# distinct ``subgroups`` shapes). Falls through to a direct call when the
# kwargs contain unhashable values (callable scorers, large polars-categorical
# dicts) so the contract stays "memo when safe; direct otherwise".
_GTC_CACHE_MAX = 16
_GTC_CACHE: "dict[tuple, Any]" = {}
# Guards _GTC_CACHE and its pin dict together: concurrent fits could otherwise evict a key between another
# thread's insert into one dict and the other, leaving a pin without its entry or an entry without its pin.
_GTC_CACHE_LOCK = threading.Lock()
# Pins the exact ``subgroups`` object whose id() was folded into a live cache key, keyed by that same
# key. CPython can reuse a freed object's id() for a brand-new, unrelated object -- without this pin, a
# short-lived ``indexed_subgroups`` dict from one target being garbage-collected and a DIFFERENT
# target's freshly-built dict landing at the same id() would produce a stale cache HIT mixing one
# target's fairness-subgroup-derived calibration metric config into another target's result. Holding a
# strong reference here for as long as the cache entry lives guarantees id() cannot be reused by
# anything else while that entry is still a valid hit; evicted together with its cache entry below.
_GTC_CACHE_SUBGROUPS_PIN: "dict[tuple, Any]" = {}


from ._trainer_configure_helpers import (  # noqa: F401  -- carved helpers
    logger,
    _configure_training_common_params_none,
    _configure_training_ngb_params_none,
    _configure_training_xgb_rfecv_rfecv,
    _configure_training_multilabel_post_hoc_calibration,
    _configure_training_use_regression_2,
    _configure_training_subgroups_none_fairness_features,
    _configure_training_val_df_size_bytes,
    _configure_training_prefer_gpu_configs_cb,
    _configure_training_lazy_model_creation_only,
    _configure_training_should_create_model_mlp,
    _configure_training_should_create_model_ngb,
    _configure_training_gated_outlier_train_target,
    _configure_training_mlframe_models_default_allowlist,
    _configure_training_learning_rate_float_trees,
    _configure_training_use_regression,
    _configure_training_minimal_linear_params_shape,
)


def _hashable_or_none(v: Any):
    """Return ``v`` when it is hash-stable across calls (suitable for a dict key),
    else ``None`` to signal the cache should bail out for this call.

    Hash-stable: built-in immutables, tuples of hash-stable items, frozensets,
    and ``type`` objects. ``id(v)`` is intentionally NOT used as a fallback --
    two structurally-identical ``indexed_subgroups`` dicts built on different
    calls would have different ``id()`` and defeat the memo on every call.
    """
    if v is None or isinstance(v, (bool, int, float, str, bytes)):
        return v
    if isinstance(v, type):
        return v
    if isinstance(v, tuple):
        out = tuple(_hashable_or_none(x) for x in v)
        return None if any(x is None and v[i] is not None for i, x in enumerate(out)) else out
    if isinstance(v, frozenset):
        return v
    return None


def _get_training_configs_cached(**kwargs):
    """Memoised wrapper for ``get_training_configs``. Falls through to a direct
    call when the kwargs contain any value that ``_hashable_or_none`` cannot
    safely hash (callable scorers, dicts, polars Series). The cache returns a
    ``copy.deepcopy`` so callers can mutate the returned SimpleNamespace
    without poisoning sibling-target entries.
    """
    from .trainer import get_training_configs
    items: list[tuple[str, Any]] = []
    cacheable = True
    _subgroups_obj = None
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if k == "subgroups":
            # ``indexed_subgroups`` is a dict-of-arrays per fairness recipe; key on
            # ``id()`` here because within the suite loop the same dict object is
            # reused across ``has_gpu`` flag toggles and across same-target sibling
            # ``get_training_configs`` calls, so ``id()`` is stable enough for the
            # 2-call CPU+GPU pair. ``None`` (default) stays a hash-stable sentinel.
            # The object itself is pinned below (_GTC_CACHE_SUBGROUPS_PIN) so this
            # id() cannot be silently reused by a different target's dict for as
            # long as the resulting cache entry stays alive.
            items.append((k, None if v is None else id(v)))
            _subgroups_obj = v
            continue
        hv = _hashable_or_none(v)
        if hv is None and v is not None:
            cacheable = False
            break
        items.append((k, hv))
    if not cacheable:
        return get_training_configs(**kwargs)
    key = tuple(items)
    with _GTC_CACHE_LOCK:
        hit = _GTC_CACHE.get(key)
    if hit is not None:
        return copy.deepcopy(hit)
    res = get_training_configs(**kwargs)
    _stored = copy.deepcopy(res)
    with _GTC_CACHE_LOCK:
        if len(_GTC_CACHE) >= _GTC_CACHE_MAX:
            # FIFO eviction (Python dict insertion order). Cheap; the cache is small
            # so an LRU dance via OrderedDict would add complexity without measurable
            # gain at maxsize=16.
            _evicted_key = next(iter(_GTC_CACHE))
            _GTC_CACHE.pop(_evicted_key)
            _GTC_CACHE_SUBGROUPS_PIN.pop(_evicted_key, None)
        _GTC_CACHE[key] = _stored
        if _subgroups_obj is not None:
            _GTC_CACHE_SUBGROUPS_PIN[key] = _subgroups_obj
    return res


def configure_training_params(
    df: Optional[pd.DataFrame] = None,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    target: Optional[pd.Series] = None,
    target_label_encoder: Optional[object] = None,
    train_target: Optional[pd.Series] = None,
    test_target: Optional[pd.Series] = None,
    val_target: Optional[pd.Series] = None,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    cat_features: list | None = None,
    text_features: list | None = None,
    embedding_features: list | None = None,
    fairness_features: Sequence | None = None,
    cont_nbins: int = 6,
    fairness_min_pop_cat_thresh: float | int = 1000,
    use_robust_eval_metric: bool = False,
    sample_weight: np.ndarray | None = None,
    prefer_gpu_configs: bool = True,
    nbins: int = 10,
    use_regression: bool = False,
    verbose: bool = True,
    rfecv_model_verbose: bool = True,
    prefer_cpu_for_lightgbm: bool = True,
    prefer_cpu_for_xgboost: bool = False,
    xgboost_verbose: int | bool = False,
    cb_fit_params: dict | None = None,
    prefer_calibrated_classifiers: bool = True,
    default_regression_scoring: dict | None = None,
    default_classification_scoring: dict | None = None,
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    group_ids: np.ndarray | None = None,
    model_name: str = "",
    common_params: dict | None = None,
    config_params: dict | None = None,
    metamodel_func: Callable | None = None,
    _precomputed_fairness_subgroups: dict | None = None,
    mlframe_models: list | None = None,
    mlframe_models_is_default_allowlist: bool = False,
    linear_model_config: LinearModelConfig | None = None,
    callback_params: dict | None = None,
    train_df_size_bytes: float | None = None,
    val_df_size_bytes: float | None = None,
    target_type: TargetTypes | None = None,
    n_classes: int | None = None,
    multilabel_dispatch_config: MultilabelDispatchConfig | None = None,
    # TrainingBehaviorConfig field; accepted here as a no-op so the caller's ``**effective_behavior_params`` splat (train_eval.py:592) doesn't fail with
    # 'unexpected keyword'. The cache bound is consumed in _pipeline_helpers via behavior_config attached to common_params.
    pre_pipeline_cache_max: int = 4,
    # Catch-all for the rest of TrainingBehaviorConfig: train_eval.py splats every
    # behavior field as **effective_behavior_params and most of them are consumed
    # downstream via the behavior_config object attached to common_params, NOT via
    # this signature. Without **_unused_behavior_kwargs every new behavior knob
    # would break the splat with TypeError. Bind the splat catch-all so the suite
    # stays forward-compatible with new TrainingBehaviorConfig fields.
    **_unused_behavior_kwargs,
):
    """Configure training parameters for all model types.

    Parameters
    ----------
    mlframe_models : list, optional
        List of model types to create. If None, all models are created.
        Used for lazy model creation to save memory.
    linear_model_config : LinearModelConfig, optional
        Configuration for linear models. If provided, applies shared settings
        to all linear model types.
    train_df_size_bytes : float, optional
        Precomputed RAM usage of train_df in bytes (e.g. from Polars
        ``.estimated_size()`` taken BEFORE pandas conversion). When
        provided, skips the pandas ``memory_usage`` call entirely. The
        value only feeds GPU-RAM-fit heuristics; Polars estimated_size
        is accurate enough and O(cols).
    val_df_size_bytes : float, optional
        Same as ``train_df_size_bytes`` for the validation split.
    """
    # Lazy import of parent-resident helpers: ``.trainer`` re-imports
    # this sibling at its bottom, so a top-level ``from .trainer
    # import ...`` would create a hard cycle the meta-test flags.
    from .trainer import LINEAR_MODEL_TYPES, RFECV, _configure_lightgbm_params, _configure_xgboost_params, create_fairness_subgroups_indices, fast_roc_auc, get_df_memory_consumption

    def _identity(x):
        """Default ``metamodel_func`` when the caller doesn't supply one: passes the model through unchanged."""
        return x

    # Helper for lazy model creation
    models_set = set(mlframe_models) if mlframe_models else None

    def _should_create_model(name: str) -> bool:
        """Check if a model should be created based on mlframe_models filter."""
        return models_set is None or name in models_set

    if metamodel_func is None:
        metamodel_func = _identity

    if default_regression_scoring is None:
        default_regression_scoring = dict(score_func=fast_mean_absolute_error, response_method="predict", greater_is_better=False)

    if default_classification_scoring is None:
        default_classification_scoring = dict(score_func=fast_roc_auc, response_method="predict_proba", greater_is_better=True)

    cb_fit_params, common_params, config_params, fairness_features, prefer_calibrated_classifiers = _configure_training_common_params_none(common_params, config_params, fairness_features, cb_fit_params, target_type, prefer_calibrated_classifiers, multilabel_dispatch_config, n_classes, mlframe_models)

    _configure_training_use_regression_2(use_regression, config_params, target, target_type)

    subgroups = _precomputed_fairness_subgroups
    subgroups = _configure_training_subgroups_none_fairness_features(subgroups, fairness_features, df, train_df, cont_nbins, fairness_min_pop_cat_thresh)

    if use_robust_eval_metric and subgroups is not None and train_idx is not None and val_idx is not None and test_idx is not None:
        indexed_subgroups = create_fairness_subgroups_indices(
            subgroups=subgroups, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, group_weights={}, cont_nbins=cont_nbins
        )
    else:
        indexed_subgroups = None

    # Per-section timers. Three candidate hot-spots: get_training_configs (called twice - CPU + GPU), get_df_memory_consumption(deep=False), and the GPU probe (cached nvidia-smi subprocess). The timers below localise the spend so the operator can see the breakdown without instrumenting by hand.
    #
    # A5#4/#16 partial memo: ``_get_training_configs_cached`` is a thin session-cache wrapper around ``get_training_configs`` keyed on the suite-invariant kwargs (``has_gpu``, the ``config_params`` content hash, and the ``indexed_subgroups`` identity). Across a multi-target suite, ``config_params`` is derived from ``hyperparams_config.model_dump()`` once and never changes per target, so the second-target call hits the cache and skips the CB / LGB / XGB defaults assembly. Memoization is a no-op when the kwargs contain unhashable values (callable scorers, polars-categorical dicts) -- the wrapper falls through to a direct call in that case.
    _t0_cfg = timer()
    cpu_configs = _get_training_configs_cached(has_gpu=False, subgroups=indexed_subgroups, **config_params)
    _t_cpu_cfg = timer() - _t0_cfg
    _t0_cfg = timer()
    gpu_configs = _get_training_configs_cached(has_gpu=None, subgroups=indexed_subgroups, **config_params)
    _t_gpu_cfg = timer() - _t0_cfg

    # Prefer caller-supplied size (typically computed on the Polars frame
    # BEFORE pandas conversion via .estimated_size() -- O(cols), microseconds).
    # Fall back to get_df_memory_consumption with deep=False -- O(cols) for
    # pandas too. Explicit deep=False avoids the O(rows) deep scan that used
    # to block this site for 3 minutes on frames with millions of unique
    # object-column strings. pyutilz default stays deep=True (back-compat);
    # mlframe opts out at this specific heuristic-only call site.
    _t0_mem = timer()
    if train_df_size_bytes is not None:
        train_df_size = float(train_df_size_bytes)
    else:
        train_df_size = get_df_memory_consumption(train_df, deep=False)
    val_df_size = _configure_training_val_df_size_bytes(val_df_size_bytes, val_df)
    data_size_gb = (train_df_size + val_df_size) / (1024**3)
    _t_mem = timer() - _t0_mem

    # Skip expensive GPU probe (nvidia-smi subprocess ~0.5s, also pulls GPUtil
    # ~50ms transitive distutils import) when GPU configs are unreachable. Three
    # opt-out conditions, any one enough:
    #   - prefer_gpu_configs=False (caller explicit opt-out)
    #   - cb_kwargs.task_type == "CPU" (CatBoost forced to CPU)
    #   - No GPU-eligible model in mlframe_models: no CatBoost AND
    #     (no XGBoost OR prefer_cpu_for_xgboost). LightGBM is excluded
    #     because prefer_cpu_for_lightgbm=True by default and lgb GPU uses
    #     OpenCL, not the CUDA topology this probe reports.
    _t0_gpu = timer()
    # ``cb_kwargs`` may be present-but-None (an explicit ``cb_kwargs=None`` in config_params),
    # so ``dict.get(..., {})`` is not enough -- it only substitutes the default for a MISSING
    # key, not for a present None value. Coerce to {} before the nested ``.get`` to avoid an
    # AttributeError: 'NoneType' object has no attribute 'get' (observed on the binary-imbalanced
    # edge-case path where the strategy left cb_kwargs unset to None).
    _cb_kwargs = config_params.get("cb_kwargs") or {}
    cb_task_type = _cb_kwargs.get("task_type")
    cb_devices = _cb_kwargs.get("devices")
    _cb_requested = models_set is None or "cb" in models_set
    _xgb_gpu_eligible = (models_set is None or "xgb" in models_set) and not prefer_cpu_for_xgboost
    _no_gpu_model_needed = not (_cb_requested or _xgb_gpu_eligible)
    data_fits_cb_gpu_ram, data_fits_gpu_ram = _configure_training_prefer_gpu_configs_cb(prefer_gpu_configs, cb_task_type, _no_gpu_model_needed, data_size_gb, cb_devices)
    _t_gpu = timer() - _t0_gpu

    logger.info("data_fits_gpu_ram=%s, data_fits_cb_gpu_ram=%s, cb_devices=%s", data_fits_gpu_ram, data_fits_cb_gpu_ram, cb_devices)
    if (_t_cpu_cfg + _t_gpu_cfg + _t_mem + _t_gpu) > 0.5:
        logger.info(
            "configure_training_params timing breakdown: " "cpu_configs=%.2fs, gpu_configs=%.2fs, mem_probe=%.2fs, gpu_probe=%.2fs (total %.2fs)",
            _t_cpu_cfg,
            _t_gpu_cfg,
            _t_mem,
            _t_gpu,
            _t_cpu_cfg + _t_gpu_cfg + _t_mem + _t_gpu,
        )

    configs = gpu_configs if (prefer_gpu_configs and data_fits_gpu_ram) else cpu_configs
    cb_configs = gpu_configs if (prefer_gpu_configs and data_fits_cb_gpu_ram) else cpu_configs

    common_params_result = dict(
        nbins=nbins,
        subgroups=subgroups,
        sample_weight=sample_weight,
        df=df,
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        target=target,
        train_target=train_target,
        test_target=test_target,
        val_target=val_target,
        train_idx=train_idx,
        test_idx=test_idx,
        val_idx=val_idx,
        target_label_encoder=target_label_encoder,
        custom_ice_metric=configs.integral_calibration_error,
        custom_rice_metric=configs.final_integral_calibration_error,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
        group_ids=group_ids,
        model_name=model_name,
        callback_params=callback_params,
        # Thread target_type through so the ensemble path (score_ensemble -> _process_single_ensemble_method -> _build_configs_from_params) can gate render_multi_target_panels via DataConfig.target_type. Without this the ensemble report block goes through report_model_perf with target_type=None and auto_dispatch falls back to firing LTR / multilabel / multiclass panels for any target with group_ids set, which is wrong on regression.
        target_type=str(target_type) if target_type is not None else None,
    )
    if common_params:
        common_params_result.update(common_params)
    common_params = common_params_result

    # Lazy model creation - only create models that are in mlframe_models (or all if None)
    cb_params = None
    cb_params = _configure_training_lazy_model_creation_only(_should_create_model, use_regression, metamodel_func, cb_configs, prefer_calibrated_classifiers, verbose, cat_features, text_features, embedding_features, cb_fit_params, cb_params)

    # Per-strategy multilabel-wrap helper. Strategies without native (N, K) target support (HGB, XGB-via-MultiOutputClassifier, LGB, Linear) need MultiOutputClassifier when target is multilabel. Inner-estimator early_stopping that depends on eval_set must be disabled because the outer wrapper doesn't slice eval_set per label; without an eval_set the inner fit would crash ("at least one dataset and eval metric is required for evaluation").
    def _wrap_for_multilabel_if_needed(estimator, strategy_cls):
        """For strategies without native ``(N, K)`` target support, wrap ``estimator`` in ``strategy_cls().wrap_multilabel`` on a MULTILABEL_CLASSIFICATION target; first strips any eval_set-dependent early-stopping params, since the multilabel wrapper doesn't slice eval_set per label. No-op for regression or non-multilabel targets."""
        if use_regression or target_type is None or not hasattr(target_type, "name") or target_type.name != "MULTILABEL_CLASSIFICATION":
            return estimator
        # Disable eval_set-dependent early stopping on the inner estimator.
        try:
            params = estimator.get_params()
        except Exception as e:
            logger.debug("estimator.get_params() failed, treating params as empty: %s", e)
            params = {}
        _patch: dict = {}
        if "early_stopping_rounds" in params and params.get("early_stopping_rounds") is not None:
            _patch["early_stopping_rounds"] = None
        # XGB sklearn >=2 uses callbacks for early stopping too; strip them.
        if "callbacks" in params and params.get("callbacks"):
            _patch["callbacks"] = None
        if _patch:
            try:
                estimator.set_params(**_patch)
            except Exception as e:
                logger.debug("swallowed exception in _trainer_configure.py: %s", e)
                pass
        return strategy_cls().wrap_multilabel(
            estimator,
            target_type,
            multilabel_config=multilabel_dispatch_config,
            n_labels=n_classes,
        )

    hgb_params = None
    if _should_create_model("hgb"):
        from .strategies import HGBStrategy

        _hgb_est = (
            HistGradientBoostingRegressor(**configs.HGB_GENERAL_PARAMS)
            if use_regression
            else _wrap_for_multilabel_if_needed(
                HistGradientBoostingClassifier(**configs.HGB_GENERAL_PARAMS),
                HGBStrategy,
            )
        )
        hgb_params = dict(model=metamodel_func(_hgb_est))

    xgb_params = None
    if _should_create_model("xgb"):
        xgb_params = _configure_xgboost_params(
            configs=configs,
            cpu_configs=cpu_configs,
            use_regression=use_regression,
            prefer_cpu_for_xgboost=prefer_cpu_for_xgboost,
            prefer_calibrated_classifiers=prefer_calibrated_classifiers,
            xgboost_verbose=xgboost_verbose,
            metamodel_func=metamodel_func,
        )
        # XGB sklearn wrapper rejects 2-D y unless we use multi_strategy='multi_output_tree' (WIP in 3.x). Default to MultiOutputClassifier instead.
        from .strategies import XGBoostStrategy

        xgb_params["model"] = _wrap_for_multilabel_if_needed(xgb_params["model"], XGBoostStrategy)

    lgb_params = None
    if _should_create_model("lgb"):
        lgb_params = _configure_lightgbm_params(
            configs=configs,
            cpu_configs=cpu_configs,
            use_regression=use_regression,
            prefer_cpu_for_lightgbm=prefer_cpu_for_lightgbm,
            prefer_calibrated_classifiers=prefer_calibrated_classifiers,
            metamodel_func=metamodel_func,
        )
        # LGB has no native multilabel -- wrap with MultiOutputClassifier.
        from .strategies import TreeModelStrategy

        lgb_params["model"] = _wrap_for_multilabel_if_needed(lgb_params["model"], TreeModelStrategy)

    mlp_params = None
    mlp_params = _configure_training_should_create_model_mlp(_should_create_model, train_df, train_target, configs, config_params, use_regression, metamodel_func, target_type, mlp_params)

    bagging_params, composite_classification_params, gated_outlier_params, ngb_params = _configure_training_ngb_params_none(_should_create_model, configs, use_regression, target_type, config_params, metamodel_func, train_target, common_params, target, train_idx, mlframe_models_is_default_allowlist)

    # Linear models - only create variants that are needed
    linear_model_params: dict[Any, Any] = {}
    linear_models_needed = LINEAR_MODEL_TYPES & models_set if models_set else LINEAR_MODEL_TYPES
    # Keys that have incompatible meanings between tree and linear models
    # (e.g., learning_rate is float for trees but string schedule for linear SGD)
    linear_config_excluded_keys = {"learning_rate"}
    _configure_training_learning_rate_float_trees(linear_models_needed, config_params, linear_config_excluded_keys, linear_model_config, use_regression, _wrap_for_multilabel_if_needed, metamodel_func, linear_model_params)

    # RFECV setup
    rfecv_params = configs.COMMON_RFECV_PARAMS.copy()
    cb_rfecv_params = cb_configs.COMMON_RFECV_PARAMS.copy()

    if not common_params.get("show_perf_chart", True):
        rfecv_params["optimizer_plotting"] = "No"
        cb_rfecv_params["optimizer_plotting"] = "No"

    if "rfecv_params" in common_params:
        custom_rfecv_params = common_params.pop("rfecv_params")
        rfecv_params.update(custom_rfecv_params)
        cb_rfecv_params.update(custom_rfecv_params)

    rfecv_scoring = _configure_training_use_regression(use_regression, default_regression_scoring, prefer_calibrated_classifiers, configs, rfecv_model_verbose, default_classification_scoring)

    params = (cb_configs.CB_REGR if use_regression else (cb_configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else cb_configs.CB_CLASSIF)).copy()

    cb_rfecv = RFECV(
        estimator=(metamodel_func(CatBoostRegressor(**params)) if use_regression else CatBoostClassifier(**params)),
        fit_params=dict(plot=rfecv_model_verbose > 1),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **cb_rfecv_params,
    )

    lgb_fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}

    lgb_rfecv = RFECV(
        estimator=(metamodel_func(LGBMRegressor(**configs.LGB_GENERAL_PARAMS)) if use_regression else LGBMClassifier(**configs.LGB_GENERAL_PARAMS)),
        fit_params=lgb_fit_params,
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    models_params, xgb_rfecv = _configure_training_xgb_rfecv_rfecv(use_regression, metamodel_func, configs, prefer_calibrated_classifiers, cat_features, rfecv_scoring, rfecv_params, cb_params, lgb_params, xgb_params, hgb_params, mlp_params, ngb_params, gated_outlier_params, bagging_params, composite_classification_params)
    # Add linear models (already filtered to only needed ones)
    models_params.update(linear_model_params)

    # Generic estimator-instance path. Beyond the built-in string tags (cb/lgb/xgb/hgb/mlp/ngb/linear),
    # ``mlframe_models`` may carry sklearn-compatible estimator INSTANCES or ``(name, estimator)`` tuples
    # (``get_strategy`` already dispatches both, MRO-based). The per-target loop keys ``models_params`` and
    # ``strategy_by_model`` by the entry object / ``id()``, so the entry itself is the key here. Mirror the
    # minimal linear params shape (``dict(model=...)``); the training body reads every other key via ``.get``.
    _configure_training_minimal_linear_params_shape(mlframe_models, metamodel_func, models_params)

    return (
        common_params,
        models_params,
        cb_rfecv,
        lgb_rfecv,
        xgb_rfecv,
        cpu_configs,
        gpu_configs,
    )
