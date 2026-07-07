"""
Core training and evaluation functions.

This module contains:
- train_and_evaluate_model: Main training function
- train_and_evaluate_model_v2: Config-based wrapper
- configure_training_params: Training parameter configuration
- _build_configs_from_params: Config object builder
"""
from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Tuple, Callable, Any

import numpy as np
import pandas as pd

# Heavy optional deps: defer failures to first actual use so `import mlframe.training` stays cheap and does not crash when a given backend is not installed.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

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
from .pipeline import (  # noqa: E402,F401
    _PRE_PIPELINE_CACHE, _PRE_PIPELINE_CACHE_LOCK, _PRE_PIPELINE_CACHE_MAX,
    _apply_pre_pipeline_transforms, _extract_feature_selector,
    _is_fitted, _multilabel_target_to_1d_for_supervised_encoders,
    _passthrough_cols_fit_transform, _pipeline_signature_for_cache,
    _pre_pipeline_cache_clear, _pre_pipeline_cache_get,
    _pre_pipeline_cache_set, _prepare_test_split,
)
from .cb import (  # noqa: E402,F401
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


def _lgb_classifier_cls() -> type:
    """Trainer-local wrapper around the factory. Reads
    ``USE_LGB_DATASET_REUSE_SHIM`` from THIS module so test monkeypatches on
    ``trainer.USE_LGB_DATASET_REUSE_SHIM`` flip dispatch as documented."""
    if USE_LGB_DATASET_REUSE_SHIM and _LGBMClassifierWithDatasetReuse is not None:
        return _LGBMClassifierWithDatasetReuse
    return _lgb_for_factory.LGBMClassifier


def _lgb_regressor_cls() -> type:
    """Trainer-local wrapper, mirror of ``_lgb_classifier_cls``."""
    if USE_LGB_DATASET_REUSE_SHIM and _LGBMRegressorWithDatasetReuse is not None:
        return _LGBMRegressorWithDatasetReuse
    return _lgb_for_factory.LGBMRegressor


def _xgb_classifier_cls() -> type:
    """Trainer-local wrapper around the XGB factory. Reads
    ``USE_XGB_DMATRIX_REUSE_SHIM`` from THIS module so test monkeypatches on
    ``trainer.USE_XGB_DMATRIX_REUSE_SHIM`` flip dispatch as documented (the
    factory's own constant is a different binding)."""
    if USE_XGB_DMATRIX_REUSE_SHIM and _XGBClassifierWithDMatrixReuse is not None:
        return _XGBClassifierWithDMatrixReuse
    return XGBClassifier


def _xgb_regressor_cls() -> type:
    """Trainer-local wrapper, mirror of ``_xgb_classifier_cls``."""
    if USE_XGB_DMATRIX_REUSE_SHIM and _XGBRegressorWithDMatrixReuse is not None:
        return _XGBRegressorWithDMatrixReuse
    return XGBRegressor
from mlframe.metrics.core import create_fairness_subgroups, create_fairness_subgroups_indices, fast_roc_auc  # noqa: F401 -- re-exported for ._trainer_configure's runtime `from .trainer import ...`
from mlframe.feature_selection.wrappers import RFECV  # noqa: F401 -- re-exported, see above
from pyutilz.pandaslib import get_df_memory_consumption  # noqa: F401 -- re-exported, see above
from .models import create_linear_model  # noqa: F401 -- re-exported, see above

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except (ImportError, OSError):  # OSError covers Windows DLL load failures (WinError 127 etc.)
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:
    from ngboost import NGBClassifier, NGBRegressor
except ImportError:
    NGBClassifier = None  # type: ignore[assignment]
    NGBRegressor = None  # type: ignore[assignment]

from .configs import (
    DataConfig, TrainingControlConfig, MetricsConfig, ReportingConfig,
    FeatureImportanceConfig, OutputConfig, NamingConfig,
    ConfidenceAnalysisConfig, PredictionsContainer,
    LinearModelConfig, VALID_LINEAR_MODEL_TYPES as LINEAR_MODEL_TYPES, TargetTypes,  # noqa: F401 -- re-exported for ._trainer_configure's runtime `from .trainer import ...`
    MultilabelDispatchConfig,  # noqa: F401 -- re-exported for ._trainer_configure's runtime `from .trainer import ...`
)
from .helpers import get_training_configs, parse_catboost_devices  # noqa: F401 -- re-exported for ._trainer_configure's runtime `from .trainer import ...`

from ._data_helpers import (  # noqa: E402,F401
    _disable_xgboost_early_stopping_if_needed, _extract_target_subset,
    _extract_targets_from_indices, _initialize_mutable_defaults,
    _normalize_multilabel_target, _prepare_df_for_model,
    _prepare_train_df_for_fitting, _setup_model_info_and_paths,
    _setup_sample_weight, _strip_internal_model_suffixes,
    _subset_dataframe, _update_model_name_after_training,
    _validate_infinity_and_columns, _validate_target_values,
    _validate_trusted_path, get_function_param_names,
)

# Backward-compat re-export. The live CatBoost-train Pool cache lives in
# mlframe.training._cb_pool; previously this module had its own dead stub at this
# slot that nothing wrote to, so any external caller doing
# ``mlframe.training.trainer._CB_POOL_CACHE.clear()`` (e.g. the suite-startup clear
# at _phase_config_setup.py and 4 sites in tests/training/test_training_overhead_integration_fixes.py)
# silently cleared an empty dict and the real cache kept its stale Pool entries.
# Aliasing here so the existing clear-callers route through to the real cache without
# code changes; new code should import _CB_POOL_CACHE from mlframe.training._cb_pool directly.
from mlframe.training.cb import _CB_POOL_CACHE  # noqa: E402,F401

logger = logging.getLogger(__name__)


def _compute_oof_preds(
    *,
    model,
    train_df,
    train_target,
    is_classifier_model: bool,
    n_splits: int,
    random_seed: int,
    group_ids=None,
    has_time: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute K-fold OOF predictions for level-1 stacking. Returns (oof_preds, oof_probs) or (None, None) on skip.

    OOF preds replace the in-sample ``train_preds`` for stacking aggregation: the canonical fix for level-1 leakage
    is to ensure the predictions a meta-learner sees on each training row were produced by a sub-model that did NOT
    see that row during fit. ``cross_val_predict`` returns exactly that vector (one held-out prediction per row).

    Splitter selection mirrors RFECV: temporal suites (``has_time``) use ``TimeSeriesSplit`` so no future train row
    predicts a past one -- a shuffled fold leaks future-into-past on autocorrelated/non-stationary targets and yields
    optimistic, selection-biased OOF (the OOF surface drives ensemble winner selection). ``GroupKFold`` honours an
    explicit grouping signal; only the genuinely i.i.d. case uses shuffled ``KFold`` (seeded from the suite seed).
    Multi-output / multilabel paths are deliberately skipped here - upstream classifier-chain / multi-output wrappers
    are non-trivial to retrofit through ``cross_val_predict`` and the level-1 stacker only reads ``oof_preds`` when
    ``max_ensembling_level > 1``; for the multi-output case we fail fast at the stacker rather than risk silently
    miscomputing OOFs. Empty/None inputs return (None, None) so callers proceed with normal training.
    """
    if train_df is None or train_target is None:
        return None, None
    try:
        n_rows = len(train_df)
    except TypeError:
        return None, None
    if n_rows < max(2 * n_splits, 10):
        # Too few rows for meaningful K-fold; level-1 stacking on tiny data isn't a realistic use case anyway.
        return None, None
    try:
        from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
        from sklearn.base import clone
    except ImportError:
        return None, None

    # Use clone() so the original (already-fit) model is not retrained in place.
    try:
        estimator = clone(model)
    except (TypeError, RuntimeError):
        # Some custom wrappers (Catboost shim, TTR, ClassifierChain) don't survive sklearn.clone; skip OOF gracefully.
        return None, None

    method = "predict_proba" if is_classifier_model and hasattr(estimator, "predict_proba") else "predict"

    if has_time and not (group_ids is not None and len(group_ids) == n_rows):
        # Temporal suite, no groups: TimeSeriesSplit is NOT a partition (early rows are never a test fold), so
        # cross_val_predict refuses it. Run the fold loop manually and leave the warm-up block as NaN -- no honest
        # OOF prediction exists for rows that were never held out. This mirrors the RFECV TimeSeriesSplit path while
        # keeping the OOF surface temporally honest (no future row predicts a past one).
        return _compute_oof_preds_timeseries(
            estimator=estimator, train_df=train_df, train_target=train_target,
            method=method, n_splits=n_splits,
        )

    if group_ids is not None and len(group_ids) == n_rows:
        # Drop shuffle: GroupKFold does not accept shuffle/seed, and group-aware OOF respects the user's grouping signal.
        # Group structure takes precedence over the time axis: TimeSeriesSplit has no group-awareness, so a grouped
        # temporal suite keeps whole groups together (the splitter upstream already assigns spanning groups to the
        # later split, preserving temporal honesty at the group granularity).
        splitter = GroupKFold(n_splits=min(n_splits, len(set(np.asarray(group_ids)))))
        _groups_arg = np.asarray(group_ids)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        _groups_arg = None

    try:
        oof = cross_val_predict(
            estimator,
            train_df,
            train_target,
            cv=splitter,
            groups=_groups_arg,
            method=method,
            n_jobs=1,
        )
    except (ValueError, TypeError, RuntimeError, NotImplementedError) as exc:  # noqa: BLE001
        logger.info("OOF prediction skipped: %s", exc)
        return None, None

    if method == "predict_proba":
        return None, np.asarray(oof)
    return np.asarray(oof), None


def _compute_oof_preds_timeseries(*, estimator, train_df, train_target, method: str, n_splits: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Manual TimeSeriesSplit OOF: cross_val_predict can't be used (TimeSeriesSplit is not a partition).

    Returns (oof_preds, oof_probs) where the warm-up rows (those never in any test fold) are NaN. Downstream OOF
    consumers mask non-finite rows before scoring.
    """
    from sklearn.base import clone as _clone
    from sklearn.model_selection import TimeSeriesSplit

    try:
        n_rows = len(train_df)
    except TypeError:
        return None, None
    y_arr = np.asarray(train_target)

    def _row(df, idx):
        if hasattr(df, "iloc"):
            return df.iloc[idx]
        return df[idx]

    tss = TimeSeriesSplit(n_splits=n_splits)
    oof_proba = None
    oof_pred = np.full(n_rows, np.nan, dtype=float)
    try:
        for tr_idx, te_idx in tss.split(np.arange(n_rows)):
            est = _clone(estimator)
            est.fit(_row(train_df, tr_idx), y_arr[tr_idx])
            if method == "predict_proba":
                p = np.asarray(est.predict_proba(_row(train_df, te_idx)))
                if oof_proba is None:
                    oof_proba = np.full((n_rows, p.shape[1]), np.nan, dtype=float)
                oof_proba[te_idx] = p
            else:
                oof_pred[te_idx] = np.asarray(est.predict(_row(train_df, te_idx))).ravel()
    except (ValueError, TypeError, RuntimeError, NotImplementedError) as exc:  # noqa: BLE001
        logger.info("Time-aware OOF prediction skipped: %s", exc)
        return None, None

    if method == "predict_proba":
        return None, oof_proba
    return oof_pred, None


def _build_configs_from_params(
    # Data params
    df=None,
    train_df=None,
    val_df=None,
    test_df=None,
    target=None,
    train_target=None,
    val_target=None,
    test_target=None,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    calib_df=None,
    calib_target=None,
    calib_idx=None,
    group_ids=None,
    sample_weight=None,
    timestamps=None,
    drop_columns=None,
    default_drop_columns=None,
    target_label_encoder=None,
    skip_infinity_checks=False,
    n_features=None,
    target_type=None,  # threaded through for downstream chart dispatch gate
    # Control params
    verbose=False,
    # use_cache=True: cache loading is almost always faster than retraining; force retrain via explicit False.
    use_cache=True,
    just_evaluate=False,
    compute_trainset_metrics=False,
    compute_valset_metrics=True,
    compute_testset_metrics=True,
    pre_pipeline=None,
    skip_pre_pipeline_transform=False,
    skip_preprocessing=False,
    fit_params=None,
    callback_params=None,
    model_category=None,
    slice_stable_es=None,
    # Metrics params
    nbins=10,
    custom_ice_metric=None,
    custom_rice_metric=None,
    subgroups=None,
    train_details="",
    val_details="",
    test_details="",
    # Reporting / display params
    figsize=(15, 5),
    print_report=True,
    show_perf_chart=True,
    show_fi=True,
    feature_importance_config=None,
    plot_file="",
    data_dir="",
    models_subdir=MODELS_SUBDIR,
    display_sample_size=0,
    show_feature_names=False,
    show_prob_histogram=True,
    prob_histogram_yscale="auto",
    show_inline_population_labels=True,
    # None sentinels: when the caller omits these, ReportingConfig's own field defaults are the
    # single source, so trainer-level defaults can never drift from the config (the prior literals
    # had drifted: a stale title template missing KS/MCC/BSS and the slow kaleido-PNG plot_outputs).
    title_metrics_template=None,
    plot_outputs=None,
    plot_dpi=None,
    # None sentinels (same single-source rationale as title_metrics_template / plot_outputs above):
    # the trainer literals had drifted from the config (missing CONFUSED_PAIRS / NDCG_BY_QSIZE / COVERAGE).
    multiclass_panels=None,
    multilabel_panels=None,
    ltr_panels=None,
    quantile_panels=None,
    # Naming params
    model_name="",
    model_name_prefix="",
    # Confidence params
    include_confidence_analysis=False,
    confidence_analysis_use_shap=True,
    confidence_analysis_max_features=6,
    confidence_analysis_cmap="bwr",
    confidence_analysis_alpha=0.9,
    confidence_analysis_ylabel="Feature value",
    confidence_analysis_title="Confidence of correct Test set predictions",
    confidence_model_kwargs=None,
    # Predictions params
    train_preds=None,
    train_probs=None,
    val_preds=None,
    val_probs=None,
    test_preds=None,
    test_probs=None,
    # Catch-all for new ReportingConfig fields that flow through
    # train_eval.py:300 via ``**all_params`` (which expands the
    # caller's reporting_config). Each new ReportingConfig field
    # otherwise breaks the splat with TypeError. Recognised fields
    # below are routed into the config objects; everything else
    # is silently dropped so the splat survives future additions.
    # Observed in prod: ``honest_estimator_diagnostics`` (added
    # to ReportingConfig but never plumbed here) raised TypeError
    # at process_model.
    honest_estimator_diagnostics=None,
    # 2026-05-28 W5 wiring: ReportingConfig.mase_seasonality
    # (default 1 at _reporting_configs.py:140). Passed as int|None
    # here so callers that don't set it leave ReportingConfig at
    # its source default. Mirrors honest_estimator_diagnostics gate.
    mase_seasonality=None,
    **_unused_reporting_kwargs,
) -> tuple:
    """Build config objects from old-style parameters."""
    merged_drop_columns = list(drop_columns or []) + list(default_drop_columns or [])

    data_config = DataConfig(
        df=df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target=target,
        train_target=train_target,
        val_target=val_target,
        test_target=test_target,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        calib_df=calib_df,
        calib_target=calib_target,
        calib_idx=calib_idx,
        group_ids=group_ids,
        sample_weight=sample_weight,
        timestamps=timestamps,
        drop_columns=merged_drop_columns,
        target_label_encoder=target_label_encoder,
        skip_infinity_checks=skip_infinity_checks,
        n_features=n_features,
        target_type=str(target_type) if target_type is not None else None,
    )

    control_config = TrainingControlConfig(
        verbose=verbose,
        use_cache=use_cache,
        just_evaluate=just_evaluate,
        compute_trainset_metrics=compute_trainset_metrics,
        compute_valset_metrics=compute_valset_metrics,
        compute_testset_metrics=compute_testset_metrics,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        fit_params=fit_params,
        callback_params=callback_params,
        model_category=model_category,
        slice_stable_es=slice_stable_es,
    )

    metrics_config = MetricsConfig(
        nbins=nbins,
        custom_ice_metric=custom_ice_metric,
        custom_rice_metric=custom_rice_metric,
        subgroups=subgroups,
        train_details=train_details,
        val_details=val_details,
        test_details=test_details,
    )

    if feature_importance_config is None:
        fi_cfg = FeatureImportanceConfig()
    elif isinstance(feature_importance_config, dict):
        fi_cfg = FeatureImportanceConfig(**feature_importance_config)
    else:
        fi_cfg = feature_importance_config

    _rep_kwargs = dict(
        figsize=figsize,
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        show_fi=show_fi,
        feature_importance_config=fi_cfg,
        display_sample_size=display_sample_size,
        show_feature_names=show_feature_names,
        show_prob_histogram=show_prob_histogram,
        prob_histogram_yscale=prob_histogram_yscale,
        show_inline_population_labels=show_inline_population_labels,
        plot_dpi=plot_dpi,
    )
    # Pass only when explicitly set; otherwise ReportingConfig's field default is the single source.
    for _name, _val in (
        ("title_metrics_template", title_metrics_template),
        ("plot_outputs", plot_outputs),
        ("multiclass_panels", multiclass_panels),
        ("multilabel_panels", multilabel_panels),
        ("ltr_panels", ltr_panels),
        ("quantile_panels", quantile_panels),
    ):
        if _val is not None:
            _rep_kwargs[_name] = _val
    if honest_estimator_diagnostics is not None:
        _rep_kwargs["honest_estimator_diagnostics"] = honest_estimator_diagnostics
    if mase_seasonality is not None:
        _rep_kwargs["mase_seasonality"] = int(mase_seasonality)
    # Generic catch-all routing: any leftover kwarg that names a real
    # ReportingConfig field is forwarded so new fields auto-propagate
    # without a per-field gate above. Truly unknown keys are dropped at
    # DEBUG (the splat must survive future additions), never raised.
    _rep_fields = ReportingConfig.model_fields
    _rep_unrecognized = []
    for _k, _v in _unused_reporting_kwargs.items():
        if _k in _rep_fields and _k not in _rep_kwargs:
            _rep_kwargs[_k] = _v
        else:
            _rep_unrecognized.append(_k)
    if _rep_unrecognized:
        logger.debug("_build_configs_from_params dropped unknown kwargs: %s", sorted(_rep_unrecognized))
    reporting_config = ReportingConfig(**_rep_kwargs)

    output_config = OutputConfig(
        plot_file=plot_file or "",
        data_dir=data_dir or "",
        # Wave 14 P2 (re-opened 2026-05-20): ``models_subdir or "models"``
        # silently rewrote a legitimate ``models_subdir=""`` (intent: write
        # models flat in data_dir, no subfolder) to ``"models"`` subdir.
        # Use explicit None-check so the empty-string intent is preserved.
        # ``plot_file or ""`` and ``data_dir or ""`` ABOVE intentionally
        # collapse falsy -> "" because their None and "" both mean
        # "feature disabled"; ``models_subdir`` is asymmetric -- None
        # means "use default" but "" is a real caller choice.
        models_dir="models" if models_subdir is None else models_subdir,
    )

    naming_config = NamingConfig(
        model_name=model_name,
        model_name_prefix=model_name_prefix,
    )

    confidence_config = ConfidenceAnalysisConfig(
        include=include_confidence_analysis,
        use_shap=confidence_analysis_use_shap,
        max_features=confidence_analysis_max_features,
        cmap=confidence_analysis_cmap,
        alpha=confidence_analysis_alpha,
        ylabel=confidence_analysis_ylabel,
        title=confidence_analysis_title,
        model_kwargs=confidence_model_kwargs or {},
    )

    predictions_container = PredictionsContainer(
        train_preds=train_preds,
        train_probs=train_probs,
        val_preds=val_preds,
        val_probs=val_probs,
        test_preds=test_preds,
        test_probs=test_probs,
    )

    return data_config, control_config, metrics_config, reporting_config, naming_config, confidence_config, predictions_container, output_config




def _configure_xgboost_params(
    configs,
    cpu_configs,
    use_regression: bool,
    prefer_cpu_for_xgboost: bool,
    prefer_calibrated_classifiers: bool,
    xgboost_verbose,
    metamodel_func: Callable,
) -> dict:
    """Configure XGBoost model parameters.

    Goes through the ``_xgb_classifier_cls`` / ``_xgb_regressor_cls``
    factories so the DMatrix-reuse shim toggle (``USE_XGB_DMATRIX_REUSE_SHIM``,
    declared at module level) is the single switching point. To revert to
    vanilla XGBoost, see the docstring of ``USE_XGB_DMATRIX_REUSE_SHIM``.
    """
    xgb_configs = cpu_configs if prefer_cpu_for_xgboost else configs

    if use_regression:
        model_cls = _xgb_regressor_cls()
        model = metamodel_func(model_cls(**xgb_configs.XGB_GENERAL_PARAMS))
    else:
        model_cls = _xgb_classifier_cls()
        xgb_classif_params = xgb_configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else xgb_configs.XGB_GENERAL_CLASSIF
        model = model_cls(**xgb_classif_params)

    return dict(model=model, fit_params=dict(verbose=xgboost_verbose))


def _configure_lightgbm_params(
    configs,
    cpu_configs,
    use_regression: bool,
    prefer_cpu_for_lightgbm: bool,
    prefer_calibrated_classifiers: bool,
    metamodel_func: Callable,
) -> dict:
    """Configure LightGBM model parameters.

    Goes through the ``_lgb_classifier_cls`` / ``_lgb_regressor_cls``
    factories so the Dataset-reuse shim toggle (``USE_LGB_DATASET_REUSE_SHIM``,
    declared at module level) is the single switching point. To revert to
    vanilla LightGBM, see the docstring of ``USE_LGB_DATASET_REUSE_SHIM``.
    """
    lgb_configs = cpu_configs if prefer_cpu_for_lightgbm else configs

    if use_regression:
        model_cls = _lgb_regressor_cls()
        model = metamodel_func(model_cls(**lgb_configs.LGB_GENERAL_PARAMS))
        fit_params = {}
    else:
        model_cls = _lgb_classifier_cls()
        model = model_cls(**lgb_configs.LGB_GENERAL_PARAMS)
        fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}

    return dict(model=model, fit_params=fit_params)


def _configure_mlp_params(
    configs,
    config_params: dict,
    use_regression: bool,
    metamodel_func: Callable,
    target_type=None,
    n_train: int | None = None,
) -> dict:
    """Configure MLP (PyTorch Lightning) model parameters.

    When ``target_type`` is supplied (multiclass / multilabel), consult ``NeuralNetStrategy.get_classif_objective_kwargs`` for the correct loss_fn + labels_dtype + task_type. Falls back to the legacy ``use_regression`` boolean for back-compat.

    ``n_train`` (when known) drives a small-data depth auto-tune: the
    suite default of nlayers=2 (lowered from 4 on 2026-05-27 after the
    extreme-AR regression run showed deeper nets gain nothing over
    boosters on tabular targets) already minimises the over-fit /
    OOD-extrapolation surface on <10k-row train splits, but the
    auto-tune still drops to nlayers=1 (single hidden layer) on very
    small data unless the caller has set
    ``mlp_kwargs["network_params"]["nlayers"]`` explicitly.
    """
    mlp_kwargs = config_params.get("mlp_kwargs", {})

    _arch_cls, _reg_cls, _cls_cls = _get_neural_components()
    if _arch_cls is None:
        raise ImportError(
            "MLP model requires the optional 'neural' extras "
            "(lightning + torchmetrics). Install via "
            "``pip install mlframe[neural]`` or omit ``mlp`` from mlframe_models."
        )
    # Defaults: nlayers=2 + ratio=2.0 -> 128->64->1, shallow tabular MLP.
    # 2026-05-27 (user request): cut from nlayers=4 (128->64->32->16->1)
    # to nlayers=2. Empirically on extreme-AR + group-aware-split regime
    # (TVT_regression.log) the 4-layer network never closed the gap to
    # boosters (R^2=-0.16 on test) -- the extra depth added optimisation
    # difficulty and OOD-extrapolation surface without measurable
    # accuracy gain. Shallower nets also halve training time and reduce
    # memory pressure during forward / backward. Callers wanting the
    # historical 4-layer topology can override via
    # ``mlp_kwargs["network_params"]["nlayers"]=4``.
    # Zero dropout + no batchnorm: dropout catastrophically kills the MLP on near-linear targets (y ~= 0.95 * x_prev + tiny residual) - four hidden layers of dropout=0.15 destroy ~52% of the signal (0.85^4) on every forward pass and the network cannot find the strong linear mapping. Tabular regression with strong linear / additive signal does not benefit from dropout (none of the big tabular libs - CB / XGB / LGB - use it either). Users with truly noise-dominated data can opt in via ``mlp_kwargs["network_params"]["dropout_prob"]=0.15``.
    mlp_network_params = dict(
        nlayers=2,
        first_layer_num_neurons=128,
        min_layer_neurons=16,
        neurons_by_layer_arch=_arch_cls.Declining,
        consec_layers_neurons_ratio=2.0,
        # 2026-05-27 (user request): activation chain history is
        #   LeakyReLU -> Tanh -> GELU + spectral_norm=True.
        # GELU is unbounded above but smoother than ReLU (gradient flows
        # everywhere). To keep the unseen-group test-split safety that
        # Tanh used to provide via saturation, ``spectral_norm=True``
        # (set below) caps each Linear's largest singular value at 1.0
        # via power iteration. Combined with GELU's Lipschitz-1
        # property the whole network is globally Lipschitz with bound
        # = depth, so OOD inputs cannot amplify magnitude geometrically.
        # The output stays bounded by ``output_activation='tanh_train_range'``
        # below, which gives a SECOND hard cap in y-scale.
        # Users may revert to Tanh via
        #   mlp_kwargs["network_params"]["activation_function"]=torch.nn.Tanh
        # or fully turn SN off via
        #   mlp_kwargs["network_params"]["spectral_norm"]=False.
        activation_function=torch.nn.GELU,
        # Kaiming-normal with relu nonlinearity is the standard
        # recommendation for GELU (close enough to relu's gain). Tanh
        # used xavier_normal_; switching to GELU + kaiming_normal_
        # together keeps init / activation aligned.
        weights_init_fcn=partial(
            nn.init.kaiming_normal_, nonlinearity="relu",
        ),
        # SN on by default 2026-05-27: bounds Lipschitz constant of
        # the linear maps to 1.0 (after power-iter convergence), making
        # catastrophic OOD extrapolation (R^2=-326 / -30 historical
        # regressions) geometrically impossible.
        spectral_norm=True,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        # 2026-05-26: ``use_batchnorm=True`` is the suite default.
        # Prior default (False) + LeakyReLU + Declining 128->64->32->16
        # + Adam + kaiming-normal init produced saturated inner pre-
        # activations on a 4.1M-row, 206-feature TVT regression: with
        # ``output_activation='tanh_train_range'`` firing under the
        # extreme-AR gate, the inner net's unbounded outputs got hard-
        # capped to +-1 in standardised space and the destandardised
        # predictions clustered at the rails [y_mid - scale, y_mid +
        # scale] (R^2 = -30.84 vs Ridge R^2 = 1.00 on identical data).
        # BatchNorm normalises per-FEATURE across the batch -- exactly
        # the axis that LN destroys (LN is per-row across features and
        # was correctly disabled below). BN keeps inner pre-activations
        # in a usable range and prevents the saturation. Users may opt
        # out via ``mlp_kwargs["network_params"]["use_batchnorm"]=False``.
        use_batchnorm=True,
        # Wave 2026-05-21: ``use_layernorm=False`` for the suite default.
        # ``generate_mlp`` defaults LN_in to True for transformer-style row-
        # independent batches, but it is WRONG for tabular regression: LN
        # normalises per-row across features, destroying the inter-row
        # absolute-scale signal that strong-linear / additive targets rely
        # on. Observed in prod (4M-row regression, y near-perfect
        # auto-regressive in a lag feature): LN_in + 5-epoch time budget +
        # group split collapsed MLP to a tight pred cluster (std ~0.17 *
        # y_std), test R^2 = -4.75 vs Ridge R^2 = 1.00 on identical data.
        # Synthetic-bench confirmed: LN_in off -> test R^2 =
        # 0.99 (matches Ridge); LN_in on -> 0.95 baseline that under
        # specific stress patterns collapses entirely. Pre-pipeline
        # already applies StandardScaler per-feature; further per-row
        # LayerNorm is double-norm noise. Users who really need LN can
        # opt in via ``mlp_kwargs["network_params"]["use_layernorm"]=True``.
        use_layernorm=False,
        # 2026-05-26: output bounded by default. The tanh_train_range
        # output activation hard-caps the regression head to
        # ``[(y_min+y_max)/2 - ((y_max-y_min)/2 + 3*y_std),
        #   (y_min+y_max)/2 + ((y_max-y_min)/2 + 3*y_std)]``. scale +
        # center are auto-derived from y_train in
        # ``neural.base._fit_inner_network`` when caller leaves them
        # unset, so this is a zero-config default. Previously only the
        # extreme-AR gate enabled this; making it the default closes
        # the failure mode at the source for ALL regression runs
        # (envelope-clip remains as a downstream defence). Users may
        # opt out via ``mlp_kwargs["network_params"]["output_activation"]="linear"``.
        output_activation="tanh_train_range",
    )
    if mlp_kwargs:
        mlp_network_params.update(mlp_kwargs.get("network_params", {}))

    # Small-data depth auto-tune: 4-layer LeakyReLU MLP over-fits the
    # train split and catastrophically extrapolates on the test split
    # when n_train is small (regression-collapse-sensor caught this on
    # the resiliency-suite mixed-scale scenario; 4920-row train,
    # 80-row test, pred_std ~600x target_std). Bench 2026-05-23 confirmed
    # nlayers=2 STILL collapses on the mixed-scale resiliency scenario;
    # nlayers=1 is the only depth that produces honest predictions under
    # the small-data + short-budget regime. Only auto-applies when the
    # caller hasn't explicitly set nlayers in network_params; explicit
    # override always wins.
    _SMALL_DATA_NLAYERS_AUTO_TUNE_THRESHOLD = 10_000
    _is_small_data = (
        n_train is not None
        and n_train < _SMALL_DATA_NLAYERS_AUTO_TUNE_THRESHOLD
    )
    _user_overrode_network = bool(
        mlp_kwargs and "nlayers" in mlp_kwargs.get("network_params", {})
    )
    if _is_small_data and not _user_overrode_network:
        if mlp_network_params["nlayers"] > 1:
            _orig_nlayers = mlp_network_params["nlayers"]
            mlp_network_params["nlayers"] = 1
            logger.info(
                "_configure_mlp_params: n_train=%d < %d -> nlayers auto-reduced "
                "from %d to 1 to avoid over-fit + test-split extrapolation. "
                "Override via mlp_kwargs={'network_params':{'nlayers': N}}.",
                n_train, _SMALL_DATA_NLAYERS_AUTO_TUNE_THRESHOLD, _orig_nlayers,
            )

    mlp_general_params = configs.MLP_GENERAL_PARAMS.copy()
    # NOTE on LR auto-tune: an earlier iteration also auto-reduced
    # learning_rate from 3e-3 to 3e-4 on small data, but that broke
    # scenario01 (near-perfect AR signal that needs the full 3e-3 step
    # to converge within max_epochs=30). The mixed-scale scenario that
    # motivated the LR cut (scenario05) requires more than just LR
    # tuning -- the per-target HPT detector in the file-level TODO is
    # the right place. Keep nlayers=1 small-data tune; leave LR alone.
    if use_regression:
        mlp_general_params["model_params"] = mlp_general_params.get("model_params", {}).copy()
        mlp_general_params["model_params"]["loss_fn"] = F.mse_loss
        mlp_general_params["datamodule_params"] = mlp_general_params.get("datamodule_params", {}).copy()
        mlp_general_params["datamodule_params"]["labels_dtype"] = torch.float32
        mlp_model = _reg_cls(network_params=mlp_network_params, **mlp_general_params)
        # Auto-standardise the regression target for MLP: a kaiming-init network outputs ~0 at init, so on a target with mean=11500 the network takes many epochs just to learn the constant offset.
        # Stock sklearn ``TransformedTargetRegressor`` standardises ONLY the ``y`` arg of fit(), leaving ``eval_set=(X_val, y_val)`` unchanged; PyTorch-Lightning consumes ``eval_set`` for its val_dataloader and computes ``val_loss`` against RAW y_val while the model predicts on STANDARDISED scale (gap of train_loss=0.16 std-units vs val_loss=1.3e+8 raw-units). The subclass below intercepts ``eval_set`` in fit_kwargs and transforms its y component too, keeping train + val on the same scale so early-stop / val_MSE callbacks see meaningful numbers.
        #
        # 2026-05-18 Pack #8: ``_TTRWithEvalSetScaling`` lives at module level
        # (``mlframe.training._ttr_eval_set_scaling``) so dill can serialise
        # fitted MLP models. Pre-fix the class was defined INSIDE this
        # function and dill choked on the ABC-metaclass ``_abc._abc_data``
        # slot through the function closure (production log:
        # ``Could not save model: cannot pickle '_abc._abc_data' object``).
        from sklearn.preprocessing import StandardScaler

        from .targets import _TTRWithEvalSetScaling

        mlp_model = _TTRWithEvalSetScaling(regressor=mlp_model, transformer=StandardScaler())
    else:
        # Target-type-aware loss / dtype / task_type for multi-* classification. Strategy method returns the dispatch dict; empty for binary (defaults already correct).
        if target_type is not None:
            from .strategies import NeuralNetStrategy

            n_cls = 0  # not used by NeuralNetStrategy.get_classif_objective_kwargs
            mlp_obj = NeuralNetStrategy().get_classif_objective_kwargs(target_type, n_cls)
            if mlp_obj:
                mlp_general_params["model_params"] = mlp_general_params.get("model_params", {}).copy()
                if "loss_fn" in mlp_obj:
                    mlp_general_params["model_params"]["loss_fn"] = mlp_obj["loss_fn"]
                # task_type lands inside model_params and is consumed by
                # MLPTorchModel.__init__ to switch predict_step activation.
                if "task_type" in mlp_obj:
                    mlp_general_params["model_params"]["task_type"] = mlp_obj["task_type"]
                if "labels_dtype" in mlp_obj:
                    mlp_general_params["datamodule_params"] = mlp_general_params.get("datamodule_params", {}).copy()
                    mlp_general_params["datamodule_params"]["labels_dtype"] = mlp_obj["labels_dtype"]
        # ``output_activation="tanh_train_range"`` is a REGRESSION-head bound
        # (the default set at :709, the only activation that requires
        # output_activation_scale/center per flat.py:536). It is meaningless
        # for a classification head and -- worse -- a binary classifier builds
        # a 1-output sigmoid head (num_classes==1), which makes generate_mlp
        # take the bounded-output branch and demand scale/center that the
        # classifier never computes (the y-derived auto-fill in base.py is
        # gated on num_classes==1 regression). Drop ONLY the train-range value
        # so the classifier head uses its native default; an explicit caller
        # 'linear' (or any future classifier-valid activation) is preserved.
        # Surfaced by fuzz (7 binary/multiclass mlp combos).
        _cls_network_params = dict(mlp_network_params)
        if _cls_network_params.get("output_activation") == "tanh_train_range":
            _cls_network_params.pop("output_activation")
        mlp_model = _cls_cls(network_params=_cls_network_params, **mlp_general_params)

    return dict(model=metamodel_func(mlp_model))


def _configure_recurrent_params(
    recurrent_models: list[str],
    recurrent_config: Any | None,
    sequences_train: list[np.ndarray] | None,
    features_train: pd.DataFrame | np.ndarray | None,
    use_regression: bool,
    metamodel_func: Callable | None = None,
) -> dict[str, dict]:
    """Configure recurrent model (LSTM, GRU, RNN, Transformer) parameters.

    Parameters
    ----------
    recurrent_models : list of str
        List of recurrent model types to configure (e.g., ["lstm", "gru"]).
    recurrent_config : RecurrentConfig or None
        Configuration for recurrent models. If None, uses defaults.
    sequences_train : list of np.ndarray or None
        Training sequences (variable length).
    features_train : DataFrame or np.ndarray or None
        Tabular features for HYBRID mode.
    use_regression : bool
        Whether to use regression (MSELoss) or classification (CrossEntropyLoss).
    metamodel_func : callable, optional
        Function to wrap the model (e.g., for calibration).

    Returns
    -------
    dict
        Dictionary mapping model names to their configurations.
    """
    from mlframe.training.neural import (
        RNNType,
        InputMode,
        RecurrentConfig,
        RecurrentClassifierWrapper,
        RecurrentRegressorWrapper,
    )

    if metamodel_func is None:

        def metamodel_func(x):
            return x

    # Determine input mode based on available data
    has_sequences = sequences_train is not None and len(sequences_train) > 0
    has_features = features_train is not None
    if hasattr(features_train, "shape"):
        has_features = has_features and features_train.shape[1] > 0

    if has_sequences and has_features:
        input_mode = InputMode.HYBRID
    elif has_sequences:
        input_mode = InputMode.SEQUENCE_ONLY
    else:
        input_mode = InputMode.FEATURES_ONLY

    # Use provided config or create default
    if recurrent_config is None:
        recurrent_config = RecurrentConfig()

    # seq_input_dim / features_dim are computed at fit-time by _RecurrentWrapperBase from input shapes (see neural/recurrent.py _aux_input_size / _seq_input_size).

    result = {}

    for model_name in recurrent_models:
        model_name_lower = model_name.lower()

        # Map model name to RNNType
        rnn_type_map = {
            "lstm": RNNType.LSTM,
            "gru": RNNType.GRU,
            "rnn": RNNType.RNN,
            "transformer": RNNType.TRANSFORMER,
        }
        if model_name_lower not in rnn_type_map:
            raise ValueError(f"Unknown recurrent model type: {model_name}. " f"Supported: {list(rnn_type_map.keys())}")

        rnn_type = rnn_type_map[model_name_lower]

        # RecurrentConfig field names are exact: n_heads (not num_heads) and mlp_hidden_sizes (not mlp_hidden_dims); seq/features dims are inferred at fit-time.
        config = RecurrentConfig(
            input_mode=input_mode,
            rnn_type=rnn_type,
            hidden_size=recurrent_config.hidden_size,
            num_layers=recurrent_config.num_layers,
            dropout=recurrent_config.dropout,
            bidirectional=recurrent_config.bidirectional,
            n_heads=recurrent_config.n_heads,
            use_attention=recurrent_config.use_attention,
            mlp_hidden_sizes=recurrent_config.mlp_hidden_sizes,
            num_classes=recurrent_config.num_classes,
            learning_rate=recurrent_config.learning_rate,
            weight_decay=recurrent_config.weight_decay,
            max_epochs=recurrent_config.max_epochs,
            batch_size=recurrent_config.batch_size,
            early_stopping_patience=recurrent_config.early_stopping_patience,
        )

        # Select wrapper class based on task type
        WrapperClass = RecurrentRegressorWrapper if use_regression else RecurrentClassifierWrapper
        wrapper = WrapperClass(config=config)

        result[model_name_lower] = dict(model=metamodel_func(wrapper))

    return result




__all__ = [
    "train_and_evaluate_model",
    "configure_training_params",
    "_build_configs_from_params",
    "run_confidence_analysis",
]


# ----------------------------------------------------------------------
# Sibling-module re-exports. The two largest functions live in
# ``_trainer_train_and_evaluate.py`` (~673 LOC) and
# ``_trainer_configure.py`` (~557 LOC) so this file stays below the
# 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._trainer_train_and_evaluate import train_and_evaluate_model  # noqa: E402,F401
from ._trainer_configure import configure_training_params  # noqa: E402,F401
