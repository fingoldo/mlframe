"""``train_and_evaluate_model`` carved out of ``mlframe.training.trainer``.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.trainer import train_and_evaluate_model``
resolves transparently.
"""
from __future__ import annotations

import copy
import inspect
import logging
import re
import pickle
from timeit import default_timer as timer
from functools import partial
import os
from os import sep as os_sep
from os.path import join, exists
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, TYPE_CHECKING, Tuple, Union
if TYPE_CHECKING:
    from ._reporting_configs import ConfidenceAnalysisConfig, NamingConfig, PredictionsContainer, ReportingConfig
    from ._training_runtime_configs import DataConfig, MetricsConfig, OutputConfig, TrainingControlConfig

import numpy as np
import pandas as pd
import polars as pl
import joblib

from pyutilz.system import compute_total_gpus_ram
from mlframe.metrics.core import compute_probabilistic_multiclass_error
from .phases import phase
from .utils import maybe_clean_ram_adaptive as _maybe_clean_ram

# Heavy optional deps: defer failures to first actual use so `import mlframe.training` stays cheap and does not crash when a given backend is not installed.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

from sklearn.base import ClassifierMixin, RegressorMixin, TransformerMixin, is_classifier
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error,
    max_error,
    r2_score,
    root_mean_squared_error,
    make_scorer,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# Optional model backends: lazy/tolerant of missing deps.
try:
    from catboost import CatBoostRegressor, CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostRegressor = CatBoostClassifier = None  # type: ignore[assignment]
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore[assignment]
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore[assignment]

from ._predict_guards import _CB_VAL_POOL_CACHE  # noqa: E402,F401
from ._pipeline_helpers import (  # noqa: E402,F401
    _PRE_PIPELINE_CACHE, _PRE_PIPELINE_CACHE_LOCK, _PRE_PIPELINE_CACHE_MAX,
    _apply_pre_pipeline_transforms, _extract_feature_selector,
    _is_fitted, _multilabel_target_to_1d_for_supervised_encoders,
    _passthrough_cols_fit_transform, _pipeline_signature_for_cache,
    _pre_pipeline_cache_clear, _pre_pipeline_cache_get,
    _pre_pipeline_cache_set, _prepare_test_split,
)
from ._cb_pool import (  # noqa: E402,F401
    _cached_gpu_info, _maybe_get_or_build_cb_pool,
    _maybe_rewrite_eval_set_as_cb_pool,
    _polars_df_has_null_in_categorical,
    _polars_fill_null_in_categorical,
    _polars_nullable_categorical_cols,
    _polars_schema_diagnostic,
    _predict_with_fallback,
)
from ._eval_helpers import (  # noqa: E402,F401
    _align_xgb_cat_categories, _append_split_rate_suffix,
    _compute_split_metrics, _decategorise_float_cat_columns,
    _filter_categorical_features, run_confidence_analysis,
)
from ._feature_name_sanitize import sanitize_frame_columns as _sanitize_frame_columns  # noqa: E402
from ._training_loop import (  # noqa: E402,F401
    _SigmoidAdapter, _PostHocCalibratedModel,
    _PostHocMultiCalibratedModel, _PerClassIsotonicCalibrator,
    _maybe_apply_posthoc_calibration, _train_model_with_fallback,
)
from ._data_helpers import (  # noqa: E402,F401
    _setup_eval_set, _setup_early_stopping_callback,
)
from ._model_factories import (  # noqa: E402,F401
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
from ._model_factories import (
    LGBMClassifierWithDatasetReuse as _LGBMClassifierWithDatasetReuse,
    LGBMRegressorWithDatasetReuse as _LGBMRegressorWithDatasetReuse,
    XGBClassifierWithDMatrixReuse as _XGBClassifierWithDMatrixReuse,
    XGBRegressorWithDMatrixReuse as _XGBRegressorWithDMatrixReuse,
)
import lightgbm as _lgb_for_factory


logger = logging.getLogger("mlframe.training.trainer")

def train_and_evaluate_model(
    model: object,
    data: DataConfig,
    control: TrainingControlConfig,
    metrics: MetricsConfig,
    reporting: ReportingConfig,
    naming: NamingConfig,
    output: OutputConfig | None = None,
    confidence: ConfidenceAnalysisConfig | None = None,
    predictions: PredictionsContainer | None = None,
    train_od_idx: np.ndarray | None = None,
    val_od_idx: np.ndarray | None = None,
    trainset_features_stats: dict | None = None,
    trusted_root: str | None = None,
    oof_n_splits: int = 0,
):
    """Train and evaluate a machine learning model with comprehensive metrics and optional caching.

    ``oof_n_splits=0`` (default, changed from 5 in fuzz iter#195) opts the K-fold OOF prediction
    pass OUT by default. The pass runs ``cross_val_predict`` with K refits of the model and at
    1M rows on a single HGB/LGB classifier this costs ~60-120 s -- pure waste when the suite is
    not running ``score_ensemble`` (use_mlframe_ensembles=False) or any level-1 stacker
    (max_ensembling_level=1). The downstream consumers tolerate missing OOF:
      - ``score_ensemble`` at ``max_ensembling_level=1`` falls back to in-sample train preds for
        single-level aggregation (ensembling.py:1097-1098); only ``max_ensembling_level>=2``
        requires OOF and raises a clear error when missing (ensembling.py:1524).
      - ``post_calibrate_model`` accepts a separate ``calib_probs`` argument or a caller-reserved
        calibration slice; the OOF-probs path is opt-in ("preferred", evaluation.py:393/430).
    Callers that need OOF (level-1 stacking or OOF-preferred calibration) MUST pass
    ``oof_n_splits>=2`` explicitly.

    Parameters
    ----------
    model : object
        The model to train (sklearn estimator, Pipeline, etc.).
    data : DataConfig
        Input data configuration (DataFrames, targets, indices).
    control : TrainingControlConfig
        Training control flags (verbose, cache, metrics computation).
    metrics : MetricsConfig
        Metrics configuration (nbins, custom metrics, subgroups).
    reporting : ReportingConfig
        Reporting / display configuration (figsize, plot settings, title-metrics template, histogram subplot, feature-importance config).
    naming : NamingConfig
        Model naming configuration.
    confidence : ConfidenceAnalysisConfig, optional
        Confidence analysis configuration.
    predictions : PredictionsContainer, optional
        Pre-computed predictions (for just_evaluate mode).
    train_od_idx : np.ndarray, optional
        Training outlier detection indices.
    val_od_idx : np.ndarray, optional
        Validation outlier detection indices.
    trainset_features_stats : dict, optional
        Pre-computed feature statistics from training set.

    Returns
    -------
    tuple
        (result_namespace, train_df, val_df, test_df) where result_namespace contains
        model, predictions, metrics, and other training artifacts.
    """
    # Lazy import of parent-resident helpers: ``.trainer`` re-imports
    # this sibling at its bottom, so a top-level ``from .trainer
    # import ...`` would create a hard cycle the meta-test flags.
    from .trainer import ConfidenceAnalysisConfig, DataConfig, FeatureImportanceConfig, MetricsConfig, NamingConfig, OutputConfig, PredictionsContainer, ReportingConfig, TrainingControlConfig, _compute_oof_preds, _disable_xgboost_early_stopping_if_needed, _extract_targets_from_indices, _prepare_train_df_for_fitting, _setup_model_info_and_paths, _setup_sample_weight, _subset_dataframe, _update_model_name_after_training, _validate_infinity_and_columns, _validate_target_values, _validate_trusted_path
    from IPython.display import display as ipython_display

    # Initialize optional configs with defaults
    if confidence is None:
        confidence = ConfidenceAnalysisConfig()
    if predictions is None:
        predictions = PredictionsContainer()

    df = data.df
    train_df = data.train_df
    val_df = data.val_df
    test_df = data.test_df
    target = data.target
    train_target = data.train_target
    val_target = data.val_target
    test_target = data.test_target
    train_idx = data.train_idx
    val_idx = data.val_idx
    test_idx = data.test_idx
    group_ids = data.group_ids
    sample_weight = data.sample_weight
    drop_columns = list(data.drop_columns) if data.drop_columns else []
    target_label_encoder = data.target_label_encoder
    skip_infinity_checks = data.skip_infinity_checks
    n_features = data.n_features

    verbose = control.verbose
    use_cache = control.use_cache
    just_evaluate = control.just_evaluate
    compute_trainset_metrics = control.compute_trainset_metrics
    compute_valset_metrics = control.compute_valset_metrics
    compute_testset_metrics = control.compute_testset_metrics
    pre_pipeline = control.pre_pipeline
    skip_pre_pipeline_transform = control.skip_pre_pipeline_transform
    skip_preprocessing = control.skip_preprocessing
    fit_params = control.fit_params
    callback_params = control.callback_params
    model_category = control.model_category

    nbins = metrics.nbins
    custom_ice_metric = metrics.custom_ice_metric
    custom_rice_metric = metrics.custom_rice_metric
    subgroups = metrics.subgroups
    train_details = metrics.train_details
    val_details = metrics.val_details
    test_details = metrics.test_details

    figsize = reporting.figsize
    print_report = reporting.print_report
    show_perf_chart = reporting.show_perf_chart
    show_fi = reporting.show_fi
    fi_config = reporting.feature_importance_config or FeatureImportanceConfig()
    fi_kwargs = dict(
        figsize=fi_config.figsize,
        num_factors=fi_config.num_factors,
        positive_fi_only=fi_config.positive_fi_only,
        show_plots=fi_config.show_plots,
        max_zero_fi_to_plot=getattr(fi_config, "max_zero_fi_to_plot", 4),
    )
    display_sample_size = reporting.display_sample_size
    show_feature_names = reporting.show_feature_names
    show_prob_histogram = reporting.show_prob_histogram
    prob_histogram_yscale = reporting.prob_histogram_yscale
    show_inline_population_labels = reporting.show_inline_population_labels
    title_metrics_tokens = reporting.title_metrics_tokens
    plot_outputs = reporting.plot_outputs
    plot_dpi = reporting.plot_dpi
    multiclass_panels = reporting.multiclass_panels
    multilabel_panels = reporting.multilabel_panels
    ltr_panels = reporting.ltr_panels
    quantile_panels = reporting.quantile_panels
    # ``quantile_alphas`` arrives via fit_params (per-fit context), not via ReportingConfig - it depends on which alphas the model was trained on, not on display preference. Resolved at the _compute_split_metrics call site.
    quantile_alphas = None
    if hasattr(model, "_mlframe_quantile_alphas"):
        quantile_alphas = getattr(model, "_mlframe_quantile_alphas", None)

    if output is None:
        output = OutputConfig()
    plot_file = output.plot_file
    data_dir = output.data_dir
    models_subdir = output.models_dir

    model_name = naming.model_name
    model_name_prefix = naming.model_name_prefix

    include_confidence_analysis = confidence.include
    confidence_analysis_use_shap = confidence.use_shap
    confidence_analysis_max_features = confidence.max_features
    confidence_analysis_cmap = confidence.cmap
    confidence_analysis_alpha = confidence.alpha
    confidence_analysis_ylabel = confidence.ylabel
    confidence_analysis_title = confidence.title
    confidence_model_kwargs = dict(confidence.model_kwargs) if confidence.model_kwargs else {}

    train_preds = predictions.train_preds
    train_probs = predictions.train_probs
    val_preds = predictions.val_preds
    val_probs = predictions.val_probs
    test_preds = predictions.test_preds
    test_probs = predictions.test_probs

    _maybe_clean_ram()

    columns = []
    best_iter = None

    _orig_train_df = train_df
    _orig_val_df = val_df
    _orig_test_df = test_df

    real_drop_columns = _validate_infinity_and_columns(
        df=df,
        train_df=train_df,
        skip_infinity_checks=skip_infinity_checks,
        drop_columns=drop_columns,
    )

    if not custom_ice_metric:
        custom_ice_metric = partial(compute_probabilistic_multiclass_error, nbins=nbins)

    model_obj, model_type_name, model_name, plot_file, model_file_name = _setup_model_info_and_paths(
        model=model,
        model_name=model_name,
        model_name_prefix=model_name_prefix,
        plot_file=plot_file,
        data_dir=data_dir,
        models_subdir=models_subdir,
    )

    if use_cache and exists(model_file_name):
        logger.info("Loading model from file %s", model_file_name)
        # Security: only load pickles from an explicitly trusted directory root.
        # Default `trusted_root` to the model file's parent dir when not provided,
        # preserving backward compat for in-process trained-then-loaded flows.
        _root = trusted_root if trusted_root is not None else os.path.dirname(os.path.abspath(model_file_name))
        _validate_trusted_path(model_file_name, _root)
        try:
            model, *_, pre_pipeline = joblib.load(model_file_name)
        except (EOFError, OSError, ModuleNotFoundError, pickle.UnpicklingError, AttributeError):
            # Wave 41 (2026-05-20): retraining is expensive; preserve traceback so the
            # operator can distinguish pickle-version mismatch / torch attribute drift /
            # disk corruption rather than re-investigating after each fallback.
            logger.warning("Failed to load cached model from %s; will retrain instead.", model_file_name, exc_info=True)
            # Continue to training - model remains as originally passed

    if fit_params is None:
        fit_params = {}
    else:
        fit_params = copy.copy(fit_params)

    train_target, val_target, test_target = _extract_targets_from_indices(target, train_idx, val_idx, test_idx, train_target, val_target, test_target)

    if (df is not None) or (train_df is not None):
        if train_df is None:
            train_df = _subset_dataframe(df, train_idx, real_drop_columns)
        if val_df is None and val_idx is not None:
            val_df = _subset_dataframe(df, val_idx, real_drop_columns)

    # Decategorise float-typed pandas categorical columns BEFORE the pre_pipeline runs (RFECV inner CB / XGB inside the pre_pipeline would otherwise reject them; see helper docstring).
    train_df, val_df, test_df = _decategorise_float_cat_columns(
        train_df,
        val_df=val_df,
        test_df=test_df,
    )

    # Thread group_ids into the pre_pipeline fit so RFECV(cv=GroupKFold())
    # and grouped MRMR receive the same sample-grouping signal the suite-level
    # callers already pass into trainer.fit. Only forwarded on train+val sample
    # range (no test). fix audit row FS-P1-1.
    _pre_pipeline_groups = None
    if group_ids is not None:
        try:
            _gi = np.asarray(group_ids)
            if train_idx is not None and len(_gi) >= int(np.max(np.asarray(train_idx))) + 1:
                _pre_pipeline_groups = _gi[np.asarray(train_idx)]
            elif train_df is not None and hasattr(train_df, "shape") and len(_gi) == train_df.shape[0]:
                _pre_pipeline_groups = _gi
        except (TypeError, ValueError, IndexError):
            _pre_pipeline_groups = None

    # Extract train-subset sample_weight BEFORE FS runs so weight-aware MRMR / RFECV (when stamped with the
    # _mlframe_use_sample_weights_in_fs_ marker by _build_pre_pipelines) can receive it via fit_params.
    # _setup_sample_weight runs AFTER FS at L730 and writes to the model's fit_params dict; the FS-side
    # forwarding happens through _apply_pre_pipeline_transforms -> _passthrough_cols_fit_transform.
    _pre_pipeline_sample_weight = None
    if sample_weight is not None:
        try:
            if isinstance(sample_weight, (pd.Series, pd.DataFrame)):
                if train_idx is not None:
                    _pre_pipeline_sample_weight = np.asarray(sample_weight.iloc[train_idx].values, dtype=np.float64)
                else:
                    _pre_pipeline_sample_weight = np.asarray(sample_weight.values, dtype=np.float64)
            else:
                _sw_arr = np.asarray(sample_weight, dtype=np.float64)
                if train_idx is not None:
                    _pre_pipeline_sample_weight = _sw_arr[train_idx]
                else:
                    _pre_pipeline_sample_weight = _sw_arr
        except (TypeError, ValueError, IndexError):
            _pre_pipeline_sample_weight = None

    train_df, val_df = _apply_pre_pipeline_transforms(
        model=model,
        pre_pipeline=pre_pipeline,
        train_df=train_df,
        val_df=val_df,
        train_target=train_target,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        use_cache=use_cache,
        model_file_name=model_file_name,
        verbose=verbose,
        selector_passthrough_cols=(list(fit_params.get("text_features") or []) + list(fit_params.get("embedding_features") or [])) or None,
        groups=_pre_pipeline_groups,
        sample_weight=_pre_pipeline_sample_weight,
    )

    # The pre-pipeline may add engineered interaction columns whose names embed
    # JSON-structural characters (e.g. ``mul(log(f2),sin(f3))``); LightGBM/XGBoost
    # reject those at fit time. Rename the model-facing labels via a pure
    # deterministic map -- train/val here and test below map identically, so fit
    # and predict stay consistent. No-op when every name is already clean.
    train_df = _sanitize_frame_columns(train_df)
    val_df = _sanitize_frame_columns(val_df)

    # Check if feature selection removed all features
    if train_df is not None and train_df.shape[1] == 0:
        logger.warning(
            f"Feature selection removed all features for {model_name} - skipping training. "
            "This typically means no features had predictive power for the target."
        )
        return (
            SimpleNamespace(
                model=None,
                test_preds=None,
                test_probs=None,
                test_target=None,
                val_preds=None,
                val_probs=None,
                val_target=None,
                train_preds=None,
                train_probs=None,
                train_target=None,
                oof_preds=None,
                oof_probs=None,
                metrics={"train": {}, "val": {}, "test": {}, "best_iter": None},
                columns=[],
                pre_pipeline=pre_pipeline,
                train_od_idx=train_od_idx,
                val_od_idx=val_od_idx,
                trainset_features_stats=trainset_features_stats,
            ),
            None,
            None,
            None,
        )

    if model is not None and pre_pipeline and not skip_pre_pipeline_transform:
        _orig_train_df = train_df
        if val_df is not None:
            _orig_val_df = val_df

    if val_df is not None:
        if isinstance(val_target, pl.Series):
            val_target = val_target.to_numpy()

        # Slice-stable ES integration (opt-in; ``control.slice_stable_es`` is None for legacy path).
        # When enabled we build per-shard eval-sets from the val frame, inject the slice-aggregator
        # knobs into ``callback_params`` so the UniversalCallback aggregates positionally, and
        # strip the booster's native early_stopping_rounds so it doesn't race the callback.
        # HGB/NGB use a single (X_val, y_val) pair (``value_format='separate'``); when slice-ES
        # is configured for these we fall back to the legacy single-val path per ``on_unsupported``.
        extra_eval_sets = None
        _slice_cfg = getattr(control, "slice_stable_es", None)
        # Slice ES infrastructure activates when either ``enabled`` (drives ES decisions) or
        # ``diagnostic_only`` (per-shard logging + Pareto plot without changing ES) is set.
        _slice_active = _slice_cfg is not None and (
            getattr(_slice_cfg, "enabled", False) or getattr(_slice_cfg, "diagnostic_only", False)
        )
        _slice_diag_only = _slice_cfg is not None and getattr(_slice_cfg, "diagnostic_only", False) and not getattr(_slice_cfg, "enabled", False)
        if _slice_active:
            from ._slice_helpers import build_slice_eval_sets
            _supports_multi_eval = model_category in {"cb", "lgb", "xgb"}
            _policy = getattr(_slice_cfg, "on_unsupported", "posthoc")
            if _supports_multi_eval:
                try:
                    val_target_arr = val_target.values if hasattr(val_target, "values") else val_target
                    _sw_val = sample_weight[val_idx] if (sample_weight is not None and val_idx is not None) else None
                    _grp_val = group_ids[val_idx] if (group_ids is not None and val_idx is not None) else None
                    extra_eval_sets = build_slice_eval_sets(
                        val_df, val_target_arr,
                        source=getattr(_slice_cfg, "source", "random"),
                        k=int(getattr(_slice_cfg, "k", 5)),
                        min_rows_per_shard=int(getattr(_slice_cfg, "min_rows_per_shard", 100)),
                        random_state=int(getattr(_slice_cfg, "random_state", 42)),
                        sample_weight=_sw_val,
                        group_ids=_grp_val,
                    )
                    if extra_eval_sets:
                        callback_params = dict(callback_params or {})
                        callback_params.update({
                            "slice_k": len(extra_eval_sets),
                            "slice_aggregate_mode": getattr(_slice_cfg, "aggregate", "mean"),
                            "slice_aggregate_alpha": float(getattr(_slice_cfg, "alpha", 1.0)),
                            "slice_aggregate_confidence": float(getattr(_slice_cfg, "confidence", 0.9)),
                            "slice_aggregate_quantile_level": float(getattr(_slice_cfg, "quantile_level", 0.9)),
                            "slice_correlation_inflation": float(getattr(_slice_cfg, "correlation_inflation", 1.5)),
                            "slice_min_delta_in_se": getattr(_slice_cfg, "min_delta_in_se", None),
                            "slice_persist_history": bool(getattr(_slice_cfg, "pareto_plot_enabled", True))
                                                     or bool(getattr(_slice_cfg, "pareto_best_iter_selection", False))
                                                     or bool(getattr(_slice_cfg, "pareto_persist_shard_history", False)),
                            "slice_diagnostic_only": _slice_diag_only,
                        })
                        # Strip native ES rounds only when the slice callback OWNS the stop
                        # decision (``enabled=True``). In ``diagnostic_only`` mode the booster's
                        # native ES path stays intact -- slice ES is logging-only here.
                        if model_obj is not None and not _slice_diag_only:
                            try:
                                if "early_stopping_rounds" in model_obj.get_params():
                                    model_obj.set_params(early_stopping_rounds=None)
                            except Exception:
                                pass  # not every estimator exposes set_params for this key
                except Exception as _slice_err:
                    logger.warning(
                        "slice-stable ES wiring failed for %s (%s); falling back to single-val path",
                        model_type_name, _slice_err,
                    )
                    extra_eval_sets = None
            else:
                if _policy == "raise":
                    raise ValueError(
                        f"slice-stable ES not supported for model_category={model_category!r}; "
                        f"set TrainingConfig.slice_stable_es.on_unsupported='posthoc' or 'skip'"
                    )
                if verbose:
                    logger.info(
                        "slice-stable ES skipped for model_category=%s (uses separate X_val/y_val kwargs); "
                        "on_unsupported=%s",
                        model_category, _policy,
                    )
        _setup_eval_set(
            model_type_name, fit_params, val_df, val_target, callback_params, model_obj, model_category,
            extra_eval_sets=extra_eval_sets,
            sample_weight_val=sample_weight[val_idx] if (sample_weight is not None and val_idx is not None) else None,
            group_ids_val=group_ids[val_idx] if (group_ids is not None and val_idx is not None) else None,
        )

        # Auto-wrap models whose category isn't natively wired into ``_setup_eval_set``
        # (linear / ridge / lasso / elasticnet / huber / sgd / ransac) in PartialFitESWrapper
        # so val drives ES via partial_fit (SGD-family) or dichotomic budget search (the
        # iterative-solver linear-family models). The wrapper transparently delegates
        # attribute access to the underlying estimator via ``__getattr__`` so downstream
        # feature-importance / calibration / SHAP code continues to read ``.coef_`` etc.
        # unchanged. Closed-form models with no usable budget knob (plain LinearRegression)
        # pass through untouched -- no ES is structurally possible there.
        from ._data_helpers import maybe_wrap_for_partial_fit_es
        _behavior_kwargs: dict[str, Any] = {}
        _beh = getattr(getattr(control, "behavior", None), "__dict__", None)
        if _beh is None:
            # ``control`` is a TrainingControlConfig; suite-level TrainingBehaviorConfig may
            # live a level up. Re-try via the explicit attribute names we care about.
            for k in ("early_stop_on_worsening", "early_stop_on_worsening_coeff",
                      "early_stop_on_worsening_min_iters"):
                v = getattr(control, k, None)
                if v is not None:
                    _behavior_kwargs[k] = v
            _auto_wrap = getattr(control, "auto_wrap_partial_fit_es", True)
        else:
            for k in ("early_stop_on_worsening", "early_stop_on_worsening_coeff",
                      "early_stop_on_worsening_min_iters"):
                if k in _beh:
                    _behavior_kwargs[k] = _beh[k]
            _auto_wrap = _beh.get("auto_wrap_partial_fit_es", True)
        # ``auto_wrap_partial_fit_es`` (TrainingBehaviorConfig field, default True)
        # gates the wrap entirely. False reaches the underlying estimator unchanged
        # -- intended for parity bench / off-switch use, not perf.
        if _auto_wrap:
            _wrapped, _did_wrap = maybe_wrap_for_partial_fit_es(
                model_obj if model is None else (model_obj or model),
                model_category=model_category or "",
                X_val=val_df, y_val=val_target,
                is_classification=("Classifier" in (model_type_name or "")),
                behavior_kwargs=_behavior_kwargs,
            )
        else:
            _wrapped, _did_wrap = None, False
        if _did_wrap:
            logger.info("Auto-wrapped %s in PartialFitESWrapper for val-driven ES "
                         "(model_category=%s)", model_type_name, model_category)
            # Replace both ``model`` (used downstream for predict / metrics / FI) and
            # ``model_obj`` (used for set_params / get_params probes upstream) with the wrapper.
            # The wrapper's __getattr__ forwards everything to the underlying estimator.
            model = _wrapped
            model_obj = _wrapped
        _maybe_clean_ram()
    else:
        _disable_xgboost_early_stopping_if_needed(model_type_name, model_obj)

    if model is not None and fit_params:
        # Two-phase coupling with FS (FS runs at the _apply_pre_pipeline_transforms call upstream):
        # FS may drop columns from ``train_df``, so the ``cat_features`` declared in ``fit_params``
        # can now reference columns that no longer exist. ``_filter_categorical_features`` reconciles
        # the cat list against the post-FS frame; reordering this block relative to the FS call
        # would silently feed CatBoost/LightGBM stale cat_features and trigger
        # "feature_name not found" at fit time.
        _filter_categorical_features(fit_params, train_df, val_df=val_df, test_df=test_df)

    if model is not None:
        if (not use_cache) or (not exists(model_file_name)):
            _setup_sample_weight(sample_weight, train_idx, model_obj, fit_params)
            if verbose:
                logger.info("training dataset shape: %s", train_df.shape)

            if display_sample_size:
                from ._reporting import _style_with_caption
                ipython_display(_style_with_caption(train_df.head(display_sample_size), f"{model_name} features head"))
                ipython_display(_style_with_caption(train_df.tail(display_sample_size), f"{model_name} features tail"))

            if train_df is not None:
                report_title = f"Training {model_name} model on {train_df.shape[1]} feature(s)"
                if show_feature_names:
                    report_title += ": " + ", ".join(list(train_df.columns))
                report_title += f", {len(train_df):_} records"

            train_df, fit_params = _prepare_train_df_for_fitting(train_df, model, model_type_name, fit_params)

            _maybe_clean_ram()
            if verbose:
                logger.info("Training the model...")

            if isinstance(train_target, pl.Series):
                train_target = train_target.to_numpy()

            # Detect classification vs regression from the model type
            # name suffix (covers all four GBM backends + sklearn linear
            # + MultiOutputClassifier + ClassifierChain). Used by
            # ``_validate_target_values`` to flag single-class collapse
            # before the per-backend C++ crash.
            _is_clf = "Classifier" in model_type_name or model_type_name in ("ClassifierChain", "_ChainEnsemble")
            _validate_target_values(train_target, "train", is_classification=_is_clf)
            if val_target is not None:
                _validate_target_values(val_target, "val", is_classification=_is_clf)

            # XGB cat-category alignment (no-op for non-XGB models): align the ``categories`` list across train / val / test so val/test rows whose category wasn't seen in train don't trip XGBoost's ``Found a category not in the training set`` rejection at predict time. Done AFTER pre_pipeline so the alignment uses the actual cat layout the model.fit + model.predict will see (pre_pipeline can rename / re-cast cat columns; aligning before that would be undone).
            train_df, val_df, test_df = _align_xgb_cat_categories(
                model_type_name,
                train_df,
                val_df=val_df,
                test_df=test_df,
            )

            if not just_evaluate:
                # Nest Lightning checkpoints + CSV logger output under the per-model directory (``{dirname(model_file_name)}/{basename_no_ext}/``) so different (target, model, schema_hash) combos don't collide in a shared project-root ``logs/`` folder. Only applies to TTR-wrapped Lightning regressors; tree models ignore this attribute. Set on the inner regressor (under TTR's ``.regressor``) when present, falling back to the model itself for direct Lightning regressors.
                try:
                    if model_file_name:
                        _ckpt_dir = os.path.splitext(model_file_name)[0]
                        _inner = getattr(model, "regressor", model)
                        if hasattr(_inner, "trainer_params"):
                            _inner.checkpoint_dir_override = _ckpt_dir
                except Exception:
                    pass
                model, best_iter = _train_model_with_fallback(
                    model=model,
                    model_obj=model_obj,
                    model_type_name=model_type_name,
                    train_df=train_df,
                    train_target=train_target,
                    fit_params=fit_params,
                    verbose=verbose,
                )

                # Handle failed model training (e.g., dtype incompatibility)
                if model is None:
                    logger.warning(f"Model {model_type_name} training failed - skipping evaluation")
                    return (
                        SimpleNamespace(
                            model=None,
                            test_preds=None,
                            test_probs=None,
                            test_target=None,
                            val_preds=None,
                            val_probs=None,
                            val_target=None,
                            train_preds=None,
                            train_probs=None,
                            train_target=None,
                            oof_preds=None,
                            oof_probs=None,
                            metrics={"train": {}, "val": {}, "test": {}, "best_iter": None},
                            columns=[],
                            pre_pipeline=pre_pipeline,
                            train_od_idx=train_od_idx,
                            val_od_idx=val_od_idx,
                            trainset_features_stats=trainset_features_stats,
                        ),
                        None,
                        None,
                        None,
                    )

            model_name = _update_model_name_after_training(model_name, len(train_df), train_details, best_iter)

            # K-fold OOF predictions for level-1 stacking. The in-sample ``train_preds`` (computed by ``_compute_split_metrics``
            # below for the "train" split) leak: every row was seen by the model during fit, so a meta-learner trained on those
            # predictions learns the residual structure of the in-sample fit, not the generalisation behaviour. OOF preds
            # produced by holding each row out via K-fold CV are the canonical replacement. Attached to the model object so
            # ``score_ensemble`` can pick them up at level-1 aggregation time without changing the public return signature.
            if oof_n_splits and oof_n_splits >= 2 and not just_evaluate:
                _is_clf_for_oof = "Classifier" in model_type_name or model_type_name in ("ClassifierChain", "_ChainEnsemble")
                # Multi-output / multi-label paths skip OOF here; their stackers will raise if level-2 is requested.
                _y_arr = np.asarray(train_target) if train_target is not None else None
                _is_multi_output_target = _y_arr is not None and _y_arr.ndim == 2
                if not _is_multi_output_target:
                    _oof_preds, _oof_probs = _compute_oof_preds(
                        model=model,
                        train_df=train_df,
                        train_target=train_target,
                        is_classifier_model=_is_clf_for_oof,
                        n_splits=int(oof_n_splits),
                        random_seed=42,
                        group_ids=group_ids[train_idx] if (group_ids is not None and train_idx is not None) else None,
                    )
                    try:
                        if _oof_preds is not None:
                            model.oof_preds = _oof_preds
                        if _oof_probs is not None:
                            model.oof_probs = _oof_probs
                            # Stamp the train-aligned target so post-hoc OOF
                            # calibration pairs each OOF prob with its OWN row's
                            # label. cross_val_predict returns predictions in
                            # train-row order, so train_target (train_idx order)
                            # is the row-for-row match. Without this,
                            # post_calibrate_model fell back to a POSITIONAL
                            # target_series slice that is correct only when train
                            # is the leading contiguous block (wrong under
                            # shuffled / group-aware splits).
                            model.oof_target = _y_arr
                    except AttributeError:
                        # Some frozen-attribute estimators (sklearn 1.4+ slots) refuse new attrs; stamp on the wrapper carrier instead.
                        pass

    metrics = {"train": {}, "val": {}, "test": {}, "best_iter": best_iter}

    if compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics:
        t0_metrics = timer()
        if verbose:
            logger.info("Computing model's performance...")

        # Compute train-target envelope stats ONCE per (model, target) and
        # forward to every split's metrics call so the prediction-envelope
        # clip (in ``report_regression_model_perf``) gets the TRAIN bound
        # rather than falling back to the per-split eval bound (which is
        # a defensive net but not the conceptually correct domain).
        # ``train_target`` here is the y-scale target for this model;
        # for composite-target estimators (CompositeTargetEstimator) the
        # inner T-scale bound is computed by the wrapper itself, the
        # outer y-scale report sees y_train and gets the right bound.
        _y_train_envelope_stats = None
        if train_target is not None:
            try:
                from ._prediction_envelope_clip import compute_train_envelope_stats
                _y_train_envelope_stats = compute_train_envelope_stats(train_target)
            except Exception as _env_err:
                logger.debug(
                    "Could not compute train envelope stats: %s. Per-split "
                    "eval-fallback envelope still applies in the reporter.",
                    _env_err,
                )

        common_metrics_params = dict(
            # ReportingConfig is forwarded so report_regression_model_perf
            # can read overrides like regression_title_metrics_tokens; before
            # this wiring the function referenced ``reporting_config`` as a
            # free variable and the custom config was silently ignored
            # (the try/except just fell back to defaults via NameError).
            reporting_config=reporting,
            model=model,
            model_type_name=model_type_name,
            model_name=model_name,
            group_ids=group_ids,
            target_label_encoder=target_label_encoder,
            figsize=figsize,
            nbins=nbins,
            print_report=print_report,
            plot_file=plot_file,
            show_perf_chart=show_perf_chart,
            show_fi=show_fi,
            fi_kwargs=fi_kwargs,
            subgroups=subgroups,
            custom_ice_metric=custom_ice_metric,
            custom_rice_metric=custom_rice_metric,
            n_features=n_features,
            show_prob_histogram=show_prob_histogram,
            prob_histogram_yscale=prob_histogram_yscale,
            show_inline_population_labels=show_inline_population_labels,
            title_metrics_tokens=title_metrics_tokens,
            plot_outputs=plot_outputs,
            plot_dpi=plot_dpi,
            multiclass_panels=multiclass_panels,
            multilabel_panels=multilabel_panels,
            ltr_panels=ltr_panels,
            quantile_panels=quantile_panels,
            quantile_alphas=quantile_alphas,
            # Authoritative target_type — gates auto_dispatch's
            # render_multi_target_panels so regression+group_ids doesn't
            # incorrectly render LTR/multilabel/multiclass panels.
            target_type=getattr(data, "target_type", None),
            # Forwarded to ``report_regression_model_perf`` so the
            # prediction-envelope clip uses the TRAIN bound. None for
            # classification / degenerate train targets; the reporter
            # auto-falls back to the per-split eval envelope in that case.
            y_train_envelope_stats=_y_train_envelope_stats,
        )

        has_val = (val_idx is not None and len(val_idx) > 0) or val_df is not None
        has_test = (test_idx is not None and len(test_idx) > 0) or test_df is not None

        splits_config = [
            (
                "train",
                train_df,
                train_target,
                train_idx,
                train_preds,
                train_probs,
                train_details,
                compute_trainset_metrics and (train_idx is not None or train_df is not None),
            ),
            (
                "val",
                val_df,
                val_target,
                val_idx,
                val_preds,
                val_probs,
                val_details,
                compute_valset_metrics and ((val_idx is not None and len(val_idx) > 0) or val_df is not None),
            ),
        ]

        # Train runs sequentially (may feed into val/test setup); val+test parallelize later.
        for split_name, split_df, split_target, split_idx, split_preds, split_probs, split_details, should_compute in splits_config:
            if should_compute and split_name == "train":
                preds_result, probs_result, columns = _compute_split_metrics(
                    split_name=split_name,
                    df=split_df,
                    target=split_target,
                    idx=split_idx,
                    metrics_dict=metrics[split_name],
                    preds=split_preds,
                    probs=split_probs,
                    details=split_details,
                    has_other_splits=has_val or has_test,
                    **common_metrics_params,
                )
                train_preds, train_probs = preds_result, probs_result

        _val_cfg = next((c for c in splits_config if c[0] == "val" and c[-1]), None)
        _run_test = compute_testset_metrics and ((test_idx is not None and len(test_idx) > 0) or test_df is not None)

        if _run_test and ((df is not None) or (test_df is not None)):
            try:
                if train_df is not None:
                    del train_df
            except NameError:
                pass
            _maybe_clean_ram()

        if _run_test:
            test_df, test_target, columns = _prepare_test_split(
                df=df,
                test_df=test_df,
                test_idx=test_idx,
                test_target=test_target,
                target=target,
                real_drop_columns=real_drop_columns,
                model=model,
                pre_pipeline=pre_pipeline,
                skip_pre_pipeline_transform=skip_pre_pipeline_transform,
                skip_preprocessing=skip_preprocessing,
                selector_passthrough_cols=(list(fit_params.get("text_features") or []) + list(fit_params.get("embedding_features") or [])) or None,
            )
            # Same engineered-name sanitization as the train/val frames above:
            # the test frame is transformed here by the same fitted pipeline, so
            # the pure map reproduces the identical label rename and predict
            # matches the model's fitted feature names.
            test_df = _sanitize_frame_columns(test_df)
            if test_df is not None:
                _orig_test_df = test_df

        # Parallelize val and test metric computation -- numba kernels release GIL,
        # Agg matplotlib is thread-safe. Pure-Python parts still block, but the
        # heavy cumtime (binning, AUC, calibration plot save) runs concurrently.
        def _run_val():
            if _val_cfg is None:
                return None
            _, sdf, starg, sidx, spreds, sprobs, sdet, _sc = _val_cfg
            return _compute_split_metrics(
                split_name="val",
                df=sdf,
                target=starg,
                idx=sidx,
                metrics_dict=metrics["val"],
                preds=spreds,
                probs=sprobs,
                details=sdet,
                has_other_splits=has_test,
                **common_metrics_params,
            )

        def _run_test_metrics():
            if not _run_test:
                return None
            return _compute_split_metrics(
                split_name="test",
                df=test_df,
                target=test_target,
                idx=test_idx,
                metrics_dict=metrics["test"],
                preds=test_preds,
                probs=test_probs,
                details=test_details,
                has_other_splits=False,
                **common_metrics_params,
            )

        # Concurrent ThreadPoolExecutor was tried but matplotlib figure creation from concurrent threads races on pyplot's shared state even with Agg backend, producing "Argument must be an image or collection" errors in calibration plots. Sequential path is correct.
        with phase("compute_split_metrics", split="val"):
            val_res = _run_val()
        with phase("compute_split_metrics", split="test"):
            test_res = _run_test_metrics()

        if val_res is not None:
            val_preds, val_probs, columns = val_res
        if test_res is not None:
            test_preds, test_probs, columns = test_res

        if _run_test:
            if include_confidence_analysis and test_df is not None:
                run_confidence_analysis(
                    test_df=test_df,
                    test_target=test_target,
                    test_probs=test_probs,
                    cat_features=fit_params.get("cat_features") if fit_params else None,
                    text_features=fit_params.get("text_features") if fit_params else None,
                    embedding_features=fit_params.get("embedding_features") if fit_params else None,
                    confidence_model_kwargs=confidence_model_kwargs,
                    fit_params=fit_params if model_type_name == "CatBoostRegressor" else None,
                    use_shap=confidence_analysis_use_shap,
                    max_features=confidence_analysis_max_features,
                    cmap=confidence_analysis_cmap,
                    alpha=confidence_analysis_alpha,
                    title=confidence_analysis_title,
                    ylabel=confidence_analysis_ylabel,
                    figsize=figsize,
                    verbose=verbose,
                )

    if (compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics) and verbose:
        logger.info("  Metrics computation done -- %.1fs", timer() - t0_metrics)

    _maybe_clean_ram()

    # OOF preds/probs were stamped on ``model`` right after training (see ``_compute_oof_preds`` call above). Mirror them onto
    # the returned namespace so ensemble member shapes carry the OOF signal alongside the per-split predictions without
    # callers having to fish them off the model object.
    _oof_preds_out = getattr(model, "oof_preds", None) if model is not None else None
    _oof_probs_out = getattr(model, "oof_probs", None) if model is not None else None

    return (
        SimpleNamespace(
            model=model,
            test_preds=test_preds,
            test_probs=test_probs,
            test_target=test_target,
            val_preds=val_preds,
            val_probs=val_probs,
            val_target=val_target,
            train_preds=train_preds,
            train_probs=train_probs,
            train_target=train_target,
            oof_preds=_oof_preds_out,
            oof_probs=_oof_probs_out,
            metrics=metrics,
            columns=columns,
            pre_pipeline=pre_pipeline,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            trainset_features_stats=trainset_features_stats,
        ),
        _orig_train_df,
        _orig_val_df,
        _orig_test_df,
    )
