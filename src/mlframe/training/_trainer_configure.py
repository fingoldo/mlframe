"""``configure_training_params`` carved out of ``mlframe.training.trainer``.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.trainer import configure_training_params``
resolves transparently.
"""
from __future__ import annotations

import copy
import logging
from timeit import default_timer as timer
from typing import Any, Callable, Sequence, TYPE_CHECKING
if TYPE_CHECKING:
    from ._configs_base import TargetTypes
    from ._model_configs import LinearModelConfig, MultilabelDispatchConfig

import numpy as np
import pandas as pd

from pyutilz.system import compute_total_gpus_ram

# Heavy optional deps: defer failures to first actual use so `import mlframe.training` stays cheap and does not crash when a given backend is not installed.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from sklearn.metrics import (
    make_scorer,
)
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
from .pipeline import (  # noqa: F401
    _PRE_PIPELINE_CACHE, _PRE_PIPELINE_CACHE_LOCK, _PRE_PIPELINE_CACHE_MAX,
    _apply_pre_pipeline_transforms, _extract_feature_selector,
    _is_fitted, _multilabel_target_to_1d_for_supervised_encoders,
    _passthrough_cols_fit_transform, _pipeline_signature_for_cache,
    _pre_pipeline_cache_clear, _pre_pipeline_cache_get,
    _pre_pipeline_cache_set, _prepare_test_split,
)
from .cb import (  # noqa: F401
    _cached_gpu_info, _maybe_get_or_build_cb_pool,
    _maybe_rewrite_eval_set_as_cb_pool,
    _polars_df_has_null_in_categorical,
    _polars_fill_null_in_categorical,
    _polars_nullable_categorical_cols,
    _polars_schema_diagnostic,
    _predict_with_fallback,
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


logger = logging.getLogger("mlframe.training.trainer")


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
    for k in sorted(kwargs.keys()):
        v = kwargs[k]
        if k == "subgroups":
            # ``indexed_subgroups`` is a dict-of-arrays per fairness recipe; key on
            # ``id()`` here because within the suite loop the same dict object is
            # reused across ``has_gpu`` flag toggles and across same-target sibling
            # ``get_training_configs`` calls, so ``id()`` is stable enough for the
            # 2-call CPU+GPU pair. ``None`` (default) stays a hash-stable sentinel.
            items.append((k, None if v is None else id(v)))
            continue
        hv = _hashable_or_none(v)
        if hv is None and v is not None:
            cacheable = False
            break
        items.append((k, hv))
    if not cacheable:
        return get_training_configs(**kwargs)
    key = tuple(items)
    hit = _GTC_CACHE.get(key)
    if hit is not None:
        return copy.deepcopy(hit)
    res = get_training_configs(**kwargs)
    if len(_GTC_CACHE) >= _GTC_CACHE_MAX:
        # FIFO eviction (Python dict insertion order). Cheap; the cache is small
        # so an LRU dance via OrderedDict would add complexity without measurable
        # gain at maxsize=16.
        _GTC_CACHE.pop(next(iter(_GTC_CACHE)))
    _GTC_CACHE[key] = copy.deepcopy(res)
    return res


def configure_training_params(
    df: pd.DataFrame = None,
    train_df: pd.DataFrame = None,
    test_df: pd.DataFrame = None,
    val_df: pd.DataFrame = None,
    target: pd.Series = None,
    target_label_encoder: object = None,
    train_target: pd.Series = None,
    test_target: pd.Series = None,
    val_target: pd.Series = None,
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
    linear_model_config: LinearModelConfig | None = None,
    callback_params: dict | None = None,
    train_df_size_bytes: float | None = None,
    val_df_size_bytes: float | None = None,
    target_type: TargetTypes | None = None,
    n_classes: int | None = None,
    multilabel_dispatch_config: MultilabelDispatchConfig | None = None,
    # TrainingBehaviorConfig field; accepted here as a no-op so the caller's
    # ``**effective_behavior_params`` splat (train_eval.py:592) doesn't fail
    # with 'unexpected keyword'. The cache bound is consumed in
    # _pipeline_helpers via behavior_config attached to common_params.
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
    from .trainer import LINEAR_MODEL_TYPES, LinearModelConfig, RFECV, TargetTypes, _configure_lightgbm_params, _configure_mlp_params, _configure_xgboost_params, create_fairness_subgroups, create_fairness_subgroups_indices, create_linear_model, fast_roc_auc, get_df_memory_consumption, parse_catboost_devices

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

    if common_params is None:
        common_params = {}
    if config_params is None:
        config_params = {}
    if fairness_features is None:
        fairness_features = []
    if cb_fit_params is None:
        cb_fit_params = {}

    # Multilabel + post-hoc calibration safety gate. ``CalibratedClassifierCV`` is single-output only; combining it with a MULTILABEL target silently fails inside the wrapper (label-list shape mismatch deep in sklearn). Honour ``MultilabelDispatchConfig.allow_uncalibrated_multi``: when False (default, strict), refuse the combo loudly so the misconfiguration is visible at config time; when True, drop the calibration request with a warning and continue. No-op when target is not multilabel or no MultilabelDispatchConfig was supplied.
    if target_type is not None and prefer_calibrated_classifiers and multilabel_dispatch_config is not None:
        if target_type == TargetTypes.MULTILABEL_CLASSIFICATION:
            if not multilabel_dispatch_config.allow_uncalibrated_multi:
                raise NotImplementedError(
                    "prefer_calibrated_classifiers=True is incompatible with "
                    "MULTILABEL_CLASSIFICATION (CalibratedClassifierCV is "
                    "single-output only). Set MultilabelDispatchConfig."
                    "allow_uncalibrated_multi=True to drop calibration with a "
                    "warning instead of raising."
                )
            logger.warning(
                "Multilabel target + prefer_calibrated_classifiers=True; "
                "dropping calibration (MultilabelDispatchConfig."
                "allow_uncalibrated_multi=True). Trained models will be "
                "uncalibrated."
            )
            prefer_calibrated_classifiers = False

    # Route target_type / n_classes into get_training_configs so per-strategy classification dispatch (CB MultiLogloss, XGB multi:softprob+num_class, LGB multiclass+num_class) gets injected. Without this, multilabel targets reach CB without loss_function set and CB's _get_loss_function_for_train tries len(set(label)) on the 2-D ndarray and crashes with TypeError: unhashable type: 'numpy.ndarray'.
    if target_type is not None and "target_type" not in config_params:
        config_params["target_type"] = target_type
    if n_classes is not None and "n_classes" not in config_params:
        config_params["n_classes"] = n_classes
    # Thread mlframe_models -> get_training_configs so the MLP config block (and its ~14s pytorch / lightning import on first call) is skipped when no neural model is requested.
    if mlframe_models is not None and "enabled_models" not in config_params:
        config_params["enabled_models"] = list(mlframe_models)

    if not use_regression:
        if "catboost_custom_classif_metrics" not in config_params:
            # Multi-output safe label count: 2-D multilabel uses n_columns;
            # 1-D binary/multiclass uses unique value count.
            target_arr = np.asarray(target) if target is not None else None
            # Multilabel detection: explicit 2-D, OR 1-D object dtype where
            # each cell is itself an array (the polars ``pl.List(pl.Int8)``
            # roundtrip lands here). Without the second clause,
            # ``np.unique(target_arr)`` raised ``truth value of array
            # ambiguous`` on the per-cell-array comparison (cb / pandas / multilabel target).
            _is_object_of_arrays = False
            if target_arr is not None and target_arr.dtype == object and target_arr.ndim == 1 and target_arr.shape[0] > 0:
                _first = target_arr[0]
                _is_object_of_arrays = hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
            if target_arr is not None and target_arr.ndim == 2:
                nlabels = target_arr.shape[1] + 1  # treat as ">2" -> multiclass-style metrics
            elif _is_object_of_arrays:
                try:
                    assert target_arr is not None
                    _first = target_arr[0]
                    nlabels = (len(_first) if hasattr(_first, "__len__") else int(np.asarray(_first).size)) + 1
                except Exception:
                    nlabels = 3
            elif target_arr is not None:
                nlabels = len(np.unique(target_arr))
            else:
                nlabels = 2
            # When multilabel: AUC is incompatible with MultiLogloss (CB rejects
            # it at fit time). Skip the AUC/PRAUC defaults and let the per-strategy
            # multilabel dispatch in helpers.py pick a compatible eval_metric.
            if target_type is not None and getattr(target_type, "name", None) == "MULTILABEL_CLASSIFICATION":
                catboost_custom_classif_metrics = []
            elif nlabels > 2:
                catboost_custom_classif_metrics = ["AUC", "PRAUC:hints=skip_train~true"]
            else:
                catboost_custom_classif_metrics = ["AUC", "PRAUC:hints=skip_train~true", "BrierScore"]
            config_params["catboost_custom_classif_metrics"] = catboost_custom_classif_metrics

    subgroups = _precomputed_fairness_subgroups
    if subgroups is None and fairness_features:
        for next_df in (df, train_df):
            if next_df is not None:
                subgroups = create_fairness_subgroups(
                    next_df,
                    features=fairness_features,
                    cont_nbins=cont_nbins,
                    min_pop_cat_thresh=fairness_min_pop_cat_thresh,
                )
                break

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
    if val_df_size_bytes is not None:
        val_df_size = float(val_df_size_bytes)
    elif val_df is not None:
        val_df_size = get_df_memory_consumption(val_df, deep=False)
    else:
        val_df_size = 0
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
    if not prefer_gpu_configs or cb_task_type == "CPU" or _no_gpu_model_needed:
        all_gpus: list = []
        data_fits_gpu_ram = False
        data_fits_cb_gpu_ram = False
    else:
        all_gpus = _cached_gpu_info()
        single_gpu_limits = compute_total_gpus_ram(all_gpus)
        data_fits_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < single_gpu_limits.get("gpu_max_ram_total", 0)
        if cb_devices:
            multi_gpu_limits = compute_total_gpus_ram(parse_catboost_devices(cb_devices, all_gpus=all_gpus))
            data_fits_cb_gpu_ram = (GPU_VRAM_SAFE_SATURATION_LIMIT * data_size_gb + GPU_VRAM_SAFE_FREE_LIMIT_GB) < multi_gpu_limits.get("gpus_ram_total", 0)
        else:
            data_fits_cb_gpu_ram = data_fits_gpu_ram
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
    if _should_create_model("cb"):
        if use_regression:
            _cb_model = metamodel_func(CatBoostRegressor(**cb_configs.CB_REGR))
        else:
            _cb_classif_params = cb_configs.CB_CALIB_CLASSIF if prefer_calibrated_classifiers else cb_configs.CB_CLASSIF
            _cb_model = CatBoostClassifier(**_cb_classif_params)
        # Defensively pre-set the polars-fastpath sticky flag. ``_predict_with_fallback`` lazily flips this attribute to True after the FIRST polars-fastpath dispatch miss, so the short-circuit fires only on the SECOND predict call onward. That works for re-using a single fitted model (VAL -> TEST), but in a suite each weight-schema iteration calls ``sklearn.clone()`` on this base ``_cb_model`` and clone strips non-param attrs, giving every fresh CB instance a blank flag.
        # CB 1.2.x's ``_set_features_order_data_polars_categorical_column`` has dispatch gaps on our nullable-Categorical / Enum schema, so opting CB into pandas at predict time bypasses the doomed retry on success and costs nothing on failure. Set on the base instance so ``clone()`` carries the param-equivalent state forward; for the attr to survive clone we also re-assert it inside ``train_eval.py:process_model``'s clone call.
        try:
            _cb_model._mlframe_polars_fastpath_broken = True
        except Exception:  # nosec B110 - non-trivial body
            # CB Python class is permissive about attributes; slot-only forks could refuse - degrade to "pay first-call retry".
            pass
        cb_params = dict(
            model=_cb_model,
            fit_params=dict(
                plot=verbose,
                cat_features=cat_features,
                **({"text_features": text_features} if text_features else {}),
                **({"embedding_features": embedding_features} if embedding_features else {}),
                **cb_fit_params,
            ),
        )

    # Per-strategy multilabel-wrap helper. Strategies without native (N, K) target support (HGB, XGB-via-MultiOutputClassifier, LGB, Linear) need MultiOutputClassifier when target is multilabel. Inner-estimator early_stopping that depends on eval_set must be disabled because the outer wrapper doesn't slice eval_set per label; without an eval_set the inner fit would crash ("at least one dataset and eval metric is required for evaluation").
    def _wrap_for_multilabel_if_needed(estimator, strategy_cls):
        """For strategies without native ``(N, K)`` target support, wrap ``estimator`` in ``strategy_cls().wrap_multilabel`` on a MULTILABEL_CLASSIFICATION target; first strips any eval_set-dependent early-stopping params, since the multilabel wrapper doesn't slice eval_set per label. No-op for regression or non-multilabel targets."""
        if use_regression or target_type is None or not hasattr(target_type, "name") or target_type.name != "MULTILABEL_CLASSIFICATION":
            return estimator
        # Disable eval_set-dependent early stopping on the inner estimator.
        try:
            params = estimator.get_params()
        except Exception:
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
    if _should_create_model("mlp"):
        # Pass training rowcount so _configure_mlp_params can auto-reduce
        # network depth on small datasets where a 4-layer LeakyReLU MLP
        # over-fits the few-thousand-row train split and catastrophically
        # extrapolates on the small test split (regression-collapse-sensor
        # documented this mode for 6k-row mixed-scale features 2026-05-23).
        _n_train_for_mlp = None
        try:
            if train_df is not None:
                _n_train_for_mlp = len(train_df)
            elif train_target is not None:
                _n_train_for_mlp = len(train_target)
        except (TypeError, ValueError):
            _n_train_for_mlp = None
        mlp_params = _configure_mlp_params(
            configs=configs,
            config_params=config_params,
            use_regression=use_regression,
            metamodel_func=metamodel_func,
            target_type=target_type,
            n_train=_n_train_for_mlp,
        )

    ngb_params = None
    if _should_create_model("ngb"):
        # Target-type-aware Dist for NGBClassifier. Default ``Dist=Bernoulli`` (binary only) crashes on K>2 with ``IndexError: index out of bounds``; for multiclass we need ``Dist=k_categorical(K)``. NGBoost has no native multilabel / ranker, so those target types fall through to the default (likely with a downstream error if reached - they should be filtered earlier when the suite checks per-strategy multilabel / ranking flags).
        ngb_init_kwargs = dict(configs.NGB_GENERAL_PARAMS)
        if not use_regression and target_type == TargetTypes.MULTICLASS_CLASSIFICATION:
            try:
                from ngboost.distns import k_categorical

                # n_classes pulled from the actual y - NGB needs the exact K to size the categorical Dist's internal parameter array. Fall back to inspecting train_target via config_params (where train_target lives at this call layer).
                _train_target = config_params.get("train_target")
                if _train_target is not None:
                    _y = np.asarray(_train_target).ravel()
                    _K = int(_y.max()) + 1 if len(_y) else 2
                else:
                    _K = max(2, int(config_params.get("n_classes", 2)))
                ngb_init_kwargs["Dist"] = k_categorical(_K)
            except ImportError:
                pass  # ngboost.distns missing -> default Dist crashes loudly downstream

        ngb_params = dict(
            model=(
                metamodel_func(
                    (NGBRegressor(**ngb_init_kwargs) if use_regression else NGBClassifier(**ngb_init_kwargs)),
                )
            ),
            fit_params=({} if config_params.get("early_stopping_rounds") is None else dict(early_stopping_rounds=config_params.get("early_stopping_rounds"))),
        )

    # Linear models - only create variants that are needed
    linear_model_params = {}
    linear_models_needed = LINEAR_MODEL_TYPES & models_set if models_set else LINEAR_MODEL_TYPES
    # Keys that have incompatible meanings between tree and linear models
    # (e.g., learning_rate is float for trees but string schedule for linear SGD)
    linear_config_excluded_keys = {"learning_rate"}
    for model_type in linear_models_needed:
        # Build config by merging: config_params -> linear_model_config -> model_type
        # This allows config_params_override["iterations"] to work for linear models
        linear_config_kwargs: dict[str, Any] = {"model_type": model_type}
        # Apply config_params first (includes iterations from config_params_override)
        if config_params:
            # Only include keys that LinearModelConfig recognizes
            linear_config_fields = set(LinearModelConfig.model_fields.keys()) - linear_config_excluded_keys
            # Also include 'iterations' which gets mapped to max_iter by the validator
            linear_config_fields.add("iterations")
            for key, value in config_params.items():
                if key in linear_config_fields:
                    linear_config_kwargs[key] = value
        # Override with explicit linear_model_config if provided
        if linear_model_config:
            linear_config_kwargs.update(linear_model_config.model_dump(exclude={"model_type"}))
        config = LinearModelConfig(**linear_config_kwargs)
        _linear_est = create_linear_model(model_type, config, use_regression=use_regression)
        # Linear classifiers reject 2-D y -> MultiOutputClassifier wrapper for multilabel.
        from .strategies import LinearModelStrategy

        _linear_est = _wrap_for_multilabel_if_needed(_linear_est, LinearModelStrategy)
        linear_model_params[model_type] = dict(model=metamodel_func(_linear_est))

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

    if use_regression:
        rfecv_scoring = make_scorer(**default_regression_scoring)
    else:
        if prefer_calibrated_classifiers:

            def fs_and_hpt_integral_calibration_error(*args, **kwargs):
                """RFECV scorer for calibrated classifiers: forwards to ``configs.fs_and_hpt_integral_calibration_error`` with the closure-captured ``verbose`` level threaded in."""
                return configs.fs_and_hpt_integral_calibration_error(*args, **kwargs, verbose=rfecv_model_verbose)

            rfecv_scoring = make_scorer(
                score_func=fs_and_hpt_integral_calibration_error,
                response_method="predict_proba",
                greater_is_better=False,
            )
        else:
            rfecv_scoring = make_scorer(**default_classification_scoring)

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

    xgb_rfecv = RFECV(
        estimator=(
            metamodel_func(XGBRegressor(**configs.XGB_GENERAL_PARAMS))
            if use_regression
            else XGBClassifier(**(configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.XGB_GENERAL_CLASSIF))
        ),
        fit_params=dict(verbose=False),
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    # Build models_params dict, only including models that were created
    models_params = {}
    if cb_params is not None:
        models_params["cb"] = cb_params
    if lgb_params is not None:
        models_params["lgb"] = lgb_params
    if xgb_params is not None:
        models_params["xgb"] = xgb_params
    if hgb_params is not None:
        models_params["hgb"] = hgb_params
    if mlp_params is not None:
        models_params["mlp"] = mlp_params
    if ngb_params is not None:
        models_params["ngb"] = ngb_params
    # Add linear models (already filtered to only needed ones)
    models_params.update(linear_model_params)

    # Generic estimator-instance path. Beyond the built-in string tags (cb/lgb/xgb/hgb/mlp/ngb/linear),
    # ``mlframe_models`` may carry sklearn-compatible estimator INSTANCES or ``(name, estimator)`` tuples
    # (``get_strategy`` already dispatches both, MRO-based). The per-target loop keys ``models_params`` and
    # ``strategy_by_model`` by the entry object / ``id()``, so the entry itself is the key here. Mirror the
    # minimal linear params shape (``dict(model=...)``); the training body reads every other key via ``.get``.
    if mlframe_models:
        for _entry in mlframe_models:
            if isinstance(_entry, str):
                continue
            _est = _entry[1] if (isinstance(_entry, tuple) and len(_entry) == 2) else _entry
            models_params[_entry] = dict(model=metamodel_func(_est))

    return (
        common_params,
        models_params,
        cb_rfecv,
        lgb_rfecv,
        xgb_rfecv,
        cpu_configs,
        gpu_configs,
    )
