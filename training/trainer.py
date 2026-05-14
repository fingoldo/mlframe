"""
Core training and evaluation functions.

This module contains:
- train_and_evaluate_model: Main training function
- train_and_evaluate_model_v2: Config-based wrapper
- configure_training_params: Training parameter configuration
- _build_configs_from_params: Config object builder
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
from typing import Optional, Tuple, Union, Callable, Sequence, List, Any, Dict

import numpy as np
import pandas as pd
import polars as pl
import joblib

from pyutilz.system import compute_total_gpus_ram
from mlframe.metrics import compute_probabilistic_multiclass_error
from .utils import maybe_clean_ram_adaptive as _maybe_clean_ram

# Heavy optional deps: defer failures to first actual use so `import mlframe.training`
# stays cheap and does not crash when a given backend is not installed. Mirrors the
# lazy-loading style in `mlframe.training.__init__.__getattr__`.
try:
    import matplotlib.pyplot as plt  # only used in a handful of plotting branches
except ImportError:  # pragma: no cover -- optional backend
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

# Optional model backends -- lazy/tolerant of missing deps, matching __init__.py style.
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

# 2026-05-13 refactor: extracted modules
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
    MODELS_SUBDIR, _get_flaml_zeroshot, _get_neural_components,
    _lgb_classifier_cls, _lgb_regressor_cls,
    _patch_dataset_constructors_with_logging,
    _patch_lgb_feature_names_in_setter,
    _xgb_classifier_cls, _xgb_regressor_cls,
)
from mlframe.metrics import fast_roc_auc
from mlframe.feature_selection.wrappers import RFECV
from pyutilz.pandaslib import get_df_memory_consumption
from .configs import (
    DataConfig, TrainingControlConfig, MetricsConfig, ReportingConfig,
    FeatureImportanceConfig, OutputConfig, NamingConfig,
    ConfidenceAnalysisConfig, PredictionsContainer, LinearModelConfig,
    MultilabelDispatchConfig, VALID_LINEAR_MODEL_TYPES as LINEAR_MODEL_TYPES,
)
from .helpers import get_training_configs

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

_CB_POOL_CACHE: "Dict[tuple, Any]" = {}

logger = logging.getLogger(__name__)


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
    group_ids=None,
    sample_weight=None,
    drop_columns=None,
    default_drop_columns=None,
    target_label_encoder=None,
    skip_infinity_checks=False,
    n_features=None,
    target_type=None,  # 2026-05-10: thread through for downstream chart dispatch gate
    # Control params
    verbose=False,
    # 2026-04-27: use_cache default flipped False -> True for consistency
    # with TrainingControlConfig.use_cache=True and the de-facto behavior
    # of train_eval.py:664's .get("use_cache", True). Cache loading is
    # almost always faster than retraining; force retrain via explicit False.
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
    # Metrics params
    nbins=10,
    custom_ice_metric=None,
    custom_rice_metric=None,
    subgroups=None,
    train_details="",
    val_details="",
    test_details="",
    # Reporting / display params (the pre-2026-04-27 display config is now
    # ReportingConfig - filesystem paths moved to OutputConfig; per-metric
    # title toggles collapsed into the ordered string template
    # `title_metrics_template`; histogram subplot toggles added; old fi_kwargs
    # dict replaced by typed FeatureImportanceConfig).
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
    title_metrics_template="ICE BR_DECOMP ECE CMAEW LL ROC_AUC PR_AUC",
    plot_outputs="plotly[html,png]",
    plot_dpi=None,
    multiclass_panels="CONFUSION PR_F1 ROC CALIB_GRID PROB_DIST TOP_K_ACC",
    multilabel_panels="PR_F1 CALIB_GRID COOCCURRENCE CARDINALITY JACCARD_DIST",
    ltr_panels="NDCG_K NDCG_DIST LIFT MRR_DIST SCORE_BY_REL",
    quantile_panels="RELIABILITY PINBALL_BY_ALPHA INTERVAL_BAND WIDTH_DIST PIT_HIST",
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
):
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
        group_ids=group_ids,
        sample_weight=sample_weight,
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

    reporting_config = ReportingConfig(
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
        title_metrics_template=title_metrics_template,
        plot_outputs=plot_outputs,
        plot_dpi=plot_dpi,
        multiclass_panels=multiclass_panels,
        multilabel_panels=multilabel_panels,
        ltr_panels=ltr_panels,
        quantile_panels=quantile_panels,
    )

    output_config = OutputConfig(
        plot_file=plot_file or "",
        data_dir=data_dir or "",
        models_dir=models_subdir or "models",
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Main Training Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def train_and_evaluate_model(
    model: object,
    data: DataConfig,
    control: TrainingControlConfig,
    metrics: MetricsConfig,
    reporting: ReportingConfig,
    naming: NamingConfig,
    output: Optional[OutputConfig] = None,
    confidence: Optional[ConfidenceAnalysisConfig] = None,
    predictions: Optional[PredictionsContainer] = None,
    train_od_idx: Optional[np.ndarray] = None,
    val_od_idx: Optional[np.ndarray] = None,
    trainset_features_stats: Optional[dict] = None,
    trusted_root: Optional[str] = None,
):
    """Train and evaluate a machine learning model with comprehensive metrics and optional caching.

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
        Reporting / display configuration (figsize, plot settings, title-metrics
        template, histogram subplot, feature-importance config). Was
        the pre-2026-04-27 display config (now renamed and slimmed).
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
    from IPython.display import display as ipython_display

    # Initialize optional configs with defaults
    if confidence is None:
        confidence = ConfidenceAnalysisConfig()
    if predictions is None:
        predictions = PredictionsContainer()

    # ---------------------------------------------------------------------------
    # Unpack data config
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # Unpack control config
    # ---------------------------------------------------------------------------
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

    # ---------------------------------------------------------------------------
    # Unpack metrics config
    # ---------------------------------------------------------------------------
    nbins = metrics.nbins
    custom_ice_metric = metrics.custom_ice_metric
    custom_rice_metric = metrics.custom_rice_metric
    subgroups = metrics.subgroups
    train_details = metrics.train_details
    val_details = metrics.val_details
    test_details = metrics.test_details

    # ---------------------------------------------------------------------------
    # Unpack reporting config (the pre-2026-04-27 display config). Filesystem paths read from
    # control.* / direct trainer state now (data_dir/models_subdir/plot_file no
    # longer live on the reporting config). FI plotting reads from a typed
    # FeatureImportanceConfig instead of a stringly-typed dict.
    # ---------------------------------------------------------------------------
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
    # ``quantile_alphas`` arrives via fit_params (per-fit context),
    # not via ReportingConfig -- it depends on which alphas the model
    # was trained on, not on display preference. Resolved at the
    # _compute_split_metrics call site.
    quantile_alphas = None
    if hasattr(model, "_mlframe_quantile_alphas"):
        quantile_alphas = getattr(model, "_mlframe_quantile_alphas", None)

    # ---------------------------------------------------------------------------
    # Unpack output config (was bundled into the display config pre-refactor). Default-construct
    # if the caller didn't pass one - keeps train_eval.py:_run_v2_path callers
    # that haven't migrated yet working with empty paths.
    # ---------------------------------------------------------------------------
    if output is None:
        output = OutputConfig()
    plot_file = output.plot_file
    data_dir = output.data_dir
    models_subdir = output.models_dir

    # ---------------------------------------------------------------------------
    # Unpack naming config
    # ---------------------------------------------------------------------------
    model_name = naming.model_name
    model_name_prefix = naming.model_name_prefix

    # ---------------------------------------------------------------------------
    # Unpack confidence config
    # ---------------------------------------------------------------------------
    include_confidence_analysis = confidence.include
    confidence_analysis_use_shap = confidence.use_shap
    confidence_analysis_max_features = confidence.max_features
    confidence_analysis_cmap = confidence.cmap
    confidence_analysis_alpha = confidence.alpha
    confidence_analysis_ylabel = confidence.ylabel
    confidence_analysis_title = confidence.title
    confidence_model_kwargs = dict(confidence.model_kwargs) if confidence.model_kwargs else {}

    # ---------------------------------------------------------------------------
    # Unpack predictions container
    # ---------------------------------------------------------------------------
    train_preds = predictions.train_preds
    train_probs = predictions.train_probs
    val_preds = predictions.val_preds
    val_probs = predictions.val_probs
    test_preds = predictions.test_preds
    test_probs = predictions.test_probs

    # ---------------------------------------------------------------------------
    # Begin original function logic
    # ---------------------------------------------------------------------------
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
        except (EOFError, OSError, ModuleNotFoundError, pickle.UnpicklingError, AttributeError) as e:
            logger.warning(f"Failed to load cached model from {model_file_name}: {e}. Will retrain instead.")
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

    # Decategorise float-typed pandas categorical columns BEFORE the
    # pre_pipeline runs (RFECV inner CB / XGB inside the pre_pipeline
    # would otherwise reject them -- fuzz c0102, see helper docstring).
    train_df, val_df, test_df = _decategorise_float_cat_columns(
        train_df,
        val_df=val_df,
        test_df=test_df,
    )

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
    )

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

        _setup_eval_set(model_type_name, fit_params, val_df, val_target, callback_params, model_obj, model_category)
        _maybe_clean_ram()
    else:
        _disable_xgboost_early_stopping_if_needed(model_type_name, model_obj)

    if model is not None and fit_params:
        _filter_categorical_features(fit_params, train_df, val_df=val_df, test_df=test_df)

    if model is not None:
        if (not use_cache) or (not exists(model_file_name)):
            _setup_sample_weight(sample_weight, train_idx, model_obj, fit_params)
            if verbose:
                logger.info("training dataset shape: %s", train_df.shape)

            if display_sample_size:
                ipython_display(train_df.head(display_sample_size).style.set_caption(f"{model_name} features head"))
                ipython_display(train_df.tail(display_sample_size).style.set_caption(f"{model_name} features tail"))

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

            # XGB cat-category alignment (no-op for non-XGB models): align
            # the ``categories`` list across train / val / test so
            # val/test rows whose category wasn't seen in train don't
            # trip XGBoost's ``Found a category not in the training set``
            # rejection at predict time (fuzz seed=2024 c0060). Done AFTER
            # pre_pipeline so the alignment uses the actual cat layout
            # the model.fit + model.predict will see (pre_pipeline can
            # rename / re-cast cat columns; aligning before that would
            # be undone).
            train_df, val_df, test_df = _align_xgb_cat_categories(
                model_type_name,
                train_df,
                val_df=val_df,
                test_df=test_df,
            )

            if not just_evaluate:
                # 2026-05-13 (user request): nest Lightning checkpoints +
                # CSV logger output under the per-model directory
                # (``{dirname(model_file_name)}/{basename_no_ext}/``) so
                # different (target, model, schema_hash) combos don't
                # collide in a shared project-root ``logs/`` folder.
                # Only applies to TTR-wrapped Lightning regressors --
                # tree models ignore this attribute. Set on the inner
                # regressor (under TTR's ``.regressor``) when present,
                # falling back to the model itself for direct Lightning
                # regressors.
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

    metrics = {"train": {}, "val": {}, "test": {}, "best_iter": best_iter}

    if compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics:
        t0_metrics = timer()
        if verbose:
            logger.info("Computing model's performance...")

        common_metrics_params = dict(
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

        # Note: concurrent ThreadPoolExecutor was tried but matplotlib figure creation
        # from concurrent threads races on pyplot's shared state even with Agg backend,
        # producing "Argument must be an image or collection" errors in calibration plots.
        # Sequential path is correct; the earlier _prepare_test_split refactor still stands.
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Configure Training Params Helper Functions
# -----------------------------------------------------------------------------------------------------------------------------------------------------


def _configure_xgboost_params(
    configs,
    cpu_configs,
    use_regression: bool,
    prefer_cpu_for_xgboost: bool,
    prefer_calibrated_classifiers: bool,
    use_flaml_zeroshot: bool,
    xgboost_verbose,
    metamodel_func,
):
    """Configure XGBoost model parameters.

    Goes through the ``_xgb_classifier_cls`` / ``_xgb_regressor_cls``
    factories so the DMatrix-reuse shim toggle (``USE_XGB_DMATRIX_REUSE_SHIM``,
    declared at module level) is the single switching point. To revert to
    vanilla XGBoost, see the docstring of ``USE_XGB_DMATRIX_REUSE_SHIM``.
    """
    xgb_configs = cpu_configs if prefer_cpu_for_xgboost else configs

    if use_regression:
        model_cls = _xgb_regressor_cls(use_flaml_zeroshot)
        model = metamodel_func(model_cls(**xgb_configs.XGB_GENERAL_PARAMS))
    else:
        model_cls = _xgb_classifier_cls(use_flaml_zeroshot)
        xgb_classif_params = xgb_configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else xgb_configs.XGB_GENERAL_CLASSIF
        model = model_cls(**xgb_classif_params)

    return dict(model=model, fit_params=dict(verbose=xgboost_verbose))


def _configure_lightgbm_params(
    configs,
    cpu_configs,
    use_regression: bool,
    prefer_cpu_for_lightgbm: bool,
    prefer_calibrated_classifiers: bool,
    use_flaml_zeroshot: bool,
    metamodel_func,
):
    """Configure LightGBM model parameters.

    Goes through the ``_lgb_classifier_cls`` / ``_lgb_regressor_cls``
    factories so the Dataset-reuse shim toggle (``USE_LGB_DATASET_REUSE_SHIM``,
    declared at module level) is the single switching point. To revert to
    vanilla LightGBM, see the docstring of ``USE_LGB_DATASET_REUSE_SHIM``.
    """
    lgb_configs = cpu_configs if prefer_cpu_for_lightgbm else configs

    if use_regression:
        model_cls = _lgb_regressor_cls(use_flaml_zeroshot)
        model = metamodel_func(model_cls(**lgb_configs.LGB_GENERAL_PARAMS))
        fit_params = {}
    else:
        model_cls = _lgb_classifier_cls(use_flaml_zeroshot)
        model = model_cls(**lgb_configs.LGB_GENERAL_PARAMS)
        fit_params = dict(eval_metric=cpu_configs.lgbm_integral_calibration_error) if prefer_calibrated_classifiers else {}

    return dict(model=model, fit_params=fit_params)


def _configure_mlp_params(
    configs,
    config_params: dict,
    use_regression: bool,
    metamodel_func: callable,
    target_type=None,
) -> dict:
    """Configure MLP (PyTorch Lightning) model parameters.

    2026-05-07: when ``target_type`` is supplied (multiclass / multilabel),
    consult ``NeuralNetStrategy.get_classif_objective_kwargs`` for the
    correct loss_fn + labels_dtype + task_type. Falls back to legacy
    ``use_regression`` boolean for back-compat.
    """
    mlp_kwargs = config_params.get("mlp_kwargs", {})

    _arch_cls, _reg_cls, _cls_cls = _get_neural_components()
    if _arch_cls is None:
        raise ImportError(
            "MLP model requires the optional 'neural' extras "
            "(lightning + torchmetrics). Install via "
            "``pip install mlframe[neural]`` or omit ``mlp`` from mlframe_models."
        )
    # 2026-05-11 (user feedback): default architecture trimmed. Previous (nlayers=20, ratio=1.5) generated a 14-layer monster like 100->66->44->29->19->13->8->5->3->2->1->1->1 -- absurd funnel that collapses representational capacity to 1 neuron by mid-network. New defaults: nlayers=4 + ratio=2.0 -> 128->64->32->16->1, a classic shallow tabular MLP. Caller can still override via mlp_kwargs["network_params"] when a different topology is needed.
    #
    # 2026-05-13 (TVT-failure root cause): defaults switched to ZERO dropout +
    # NO batchnorm. The previous defaults (``dropout_prob=0.15`` +
    # ``inputs_dropout_prob=0.002`` + ``use_batchnorm=True``) catastrophically
    # killed the MLP on near-linear targets like TVT (y ~= 0.95 * TVT_prev +
    # tiny residual), where linear regression is the MLE estimator and the
    # MLP's job is to match it. Four hidden layers of dropout=0.15 means
    # ~52% of the signal (0.85^4) is destroyed on every forward pass; the
    # network simply cannot find the strong linear mapping. Production
    # symptom: 2-hour MLP run on 4M-row TVT collapsed predictions to a
    # narrow band [11k, 11.7k] around the mean, R2=0.33 vs linear R2=0.85.
    #
    # New defaults: dropout=0 + batchnorm=False. Tabular regression with
    # strong linear / additive signal does NOT benefit from dropout (none
    # of the big tabular libs -- CB / XGB / LGB -- use it either). Users
    # whose dataset is truly noise-dominated can opt in via
    # ``mlp_kwargs["network_params"]["dropout_prob"]=0.15``.
    mlp_network_params = dict(
        nlayers=4,
        first_layer_num_neurons=128,
        min_layer_neurons=16,
        neurons_by_layer_arch=_arch_cls.Declining,
        consec_layers_neurons_ratio=2.0,
        activation_function=torch.nn.LeakyReLU,
        weights_init_fcn=partial(nn.init.kaiming_normal_, nonlinearity="leaky_relu", a=0.01),
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        use_batchnorm=False,
    )
    if mlp_kwargs:
        mlp_network_params.update(mlp_kwargs.get("network_params", {}))

    mlp_general_params = configs.MLP_GENERAL_PARAMS.copy()
    if use_regression:
        mlp_general_params["model_params"] = mlp_general_params.get("model_params", {}).copy()
        mlp_general_params["model_params"]["loss_fn"] = F.mse_loss
        mlp_general_params["datamodule_params"] = mlp_general_params.get("datamodule_params", {}).copy()
        mlp_general_params["datamodule_params"]["labels_dtype"] = torch.float32
        mlp_model = _reg_cls(network_params=mlp_network_params, **mlp_general_params)
        # F1 fix (2026-05-11): auto-standardise the regression target for MLP. A kaiming-init network outputs ~0 at init; on a target with mean=11500 the network takes many epochs to learn just the constant offset.
        # F7 fix (2026-05-11): the initial F1 fix used sklearn's stock ``TransformedTargetRegressor`` which standardises ONLY the ``y`` arg of fit(), leaving ``eval_set=(X_val, y_val)`` unchanged. PyTorch-Lightning consumes ``eval_set`` for its val_dataloader and computes ``val_loss`` against RAW y_val while the model predicts on STANDARDISED scale -- train_loss=0.16 (std units) vs val_loss=1.3e+8 (raw units) gap observed in the 2026-05-12 run. New subclass intercepts ``eval_set`` in fit_kwargs and transforms its y component too, keeping train + val on the SAME scale so early-stop / val_MSE callbacks see meaningful numbers.
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.preprocessing import StandardScaler

        class _TTRWithEvalSetScaling(TransformedTargetRegressor):
            """``TransformedTargetRegressor`` extension that ALSO standardises any ``eval_set`` / ``X_val`` + ``y_val`` arrays in fit_params. Required for inner estimators (PyTorch-Lightning MLP, LightGBM, etc.) that consume eval_set for their own val-loss / early-stopping. Without this, train sees standardised y and val sees raw y, making the early-stop metric nonsensical."""

            def fit(self, X, y, **fit_params):
                # Fit the transformer FIRST on y so we have the same scale to apply to eval_set's y_val. Mirrors what ``TransformedTargetRegressor.fit`` does internally (line 167 in sklearn 1.5+).
                import numpy as _np
                from sklearn.base import clone as _clone

                y_arr = _np.asarray(y, dtype=_np.float64)
                if y_arr.ndim == 1:
                    y_arr_2d = y_arr.reshape(-1, 1)
                else:
                    y_arr_2d = y_arr
                self.transformer_ = _clone(self.transformer) if self.transformer is not None else None
                if self.transformer_ is not None:
                    self.transformer_.fit(y_arr_2d)
                    # Intercept + transform eval_set's y_val before delegating.
                    if "eval_set" in fit_params and fit_params["eval_set"] is not None:
                        es = fit_params["eval_set"]
                        # eval_set comes as ``(X_val, y_val)`` for MLP / LGB. For XGB / CB it's ``[(X_val, y_val), ...]``.
                        if isinstance(es, tuple) and len(es) == 2:
                            X_val, y_val = es
                            y_val_arr = _np.asarray(y_val, dtype=_np.float64)
                            y_val_2d = y_val_arr.reshape(-1, 1) if y_val_arr.ndim == 1 else y_val_arr
                            y_val_scaled = self.transformer_.transform(y_val_2d).reshape(y_val_arr.shape)
                            fit_params["eval_set"] = (X_val, y_val_scaled)
                        elif isinstance(es, list):
                            new_es = []
                            for entry in es:
                                if isinstance(entry, tuple) and len(entry) == 2:
                                    X_v, y_v = entry
                                    y_v_arr = _np.asarray(y_v, dtype=_np.float64)
                                    y_v_2d = y_v_arr.reshape(-1, 1) if y_v_arr.ndim == 1 else y_v_arr
                                    y_v_scaled = self.transformer_.transform(y_v_2d).reshape(y_v_arr.shape)
                                    new_es.append((X_v, y_v_scaled))
                                else:
                                    new_es.append(entry)
                            fit_params["eval_set"] = new_es
                # Defer the actual fit to the parent (which will refit transformer + call regressor.fit).
                return super().fit(X, y, **fit_params)

        mlp_model = _TTRWithEvalSetScaling(regressor=mlp_model, transformer=StandardScaler())
    else:
        # 2026-05-07: target-type-aware loss / dtype / task_type for multi-*
        # classification. Strategy method returns the dispatch dict;
        # empty for binary (defaults already correct).
        if target_type is not None:
            from .strategies import NeuralNetStrategy
            from .configs import TargetTypes as _TT

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
        mlp_model = _cls_cls(network_params=mlp_network_params, **mlp_general_params)

    return dict(model=metamodel_func(mlp_model))


def _configure_recurrent_params(
    recurrent_models: List[str],
    recurrent_config: Optional[Any],
    sequences_train: Optional[List[np.ndarray]],
    features_train: Optional[Union[pd.DataFrame, np.ndarray]],
    use_regression: bool,
    metamodel_func: callable = None,
) -> Dict[str, dict]:
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

    # Infer dimensions from data
    if has_sequences:
        seq_input_dim = sequences_train[0].shape[-1] if sequences_train[0].ndim > 1 else 1
    else:
        seq_input_dim = 0

    if has_features:
        if hasattr(features_train, "shape"):
            features_dim = features_train.shape[1]
        else:
            features_dim = len(features_train.columns)
    else:
        features_dim = 0

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

        # Create model-specific config.
        # 2026-04-24 (test_recurrent_lstm_smoke surfaced 4 bugs):
        #   * ``seq_input_dim`` / ``features_dim`` were passed as
        #     RecurrentConfig kwargs but the dataclass has no such
        #     fields. The wrapper computes both internally during
        #     ``fit`` from input shapes (see ``_RecurrentWrapperBase``
        #     in neural/recurrent.py:1041-1042: ``_aux_input_size``
        #     and ``_seq_input_size`` populated at fit-time). The
        #     dimensions captured at lines 3274-3284 above are now
        #     unused; left in place because future config-time
        #     validation may want them.
        #   * ``num_heads`` was a typo for ``n_heads`` (RecurrentConfig
        #     declares ``n_heads`` at neural/recurrent.py:170).
        #   * ``mlp_hidden_dims`` was a typo for ``mlp_hidden_sizes``
        #     (declared at neural/recurrent.py:174).
        # Without this fix every recurrent model (LSTM/GRU/RNN/
        # Transformer) crashes immediately with TypeError /
        # AttributeError on construction.
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


# -----------------------------------------------------------------------------------------------------------------------------------------------------
# Configure Training Parameters
# -----------------------------------------------------------------------------------------------------------------------------------------------------


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
    train_idx: np.ndarray = None,
    val_idx: np.ndarray = None,
    test_idx: np.ndarray = None,
    cat_features: list = None,
    text_features: list = None,
    embedding_features: list = None,
    fairness_features: Sequence = None,
    cont_nbins: int = 6,
    fairness_min_pop_cat_thresh: Union[float, int] = 1000,
    use_robust_eval_metric: bool = False,
    sample_weight: np.ndarray = None,
    prefer_gpu_configs: bool = True,
    nbins: int = 10,
    use_regression: bool = False,
    verbose: bool = True,
    rfecv_model_verbose: bool = True,
    prefer_cpu_for_lightgbm: bool = True,
    prefer_cpu_for_xgboost: bool = False,
    xgboost_verbose: Union[int, bool] = False,
    cb_fit_params: dict = None,
    prefer_calibrated_classifiers: bool = True,
    default_regression_scoring: dict = None,
    default_classification_scoring: dict = None,
    train_details: str = "",
    val_details: str = "",
    test_details: str = "",
    group_ids: np.ndarray = None,
    model_name: str = "",
    common_params: dict = None,
    config_params: dict = None,
    metamodel_func: callable = None,
    use_flaml_zeroshot: bool = False,
    _precomputed_fairness_subgroups: dict = None,
    mlframe_models: list = None,
    linear_model_config: "LinearModelConfig" = None,
    callback_params: dict = None,
    train_df_size_bytes: Optional[float] = None,
    val_df_size_bytes: Optional[float] = None,
    target_type: Optional["TargetTypes"] = None,
    n_classes: Optional[int] = None,
    multilabel_dispatch_config: Optional["MultilabelDispatchConfig"] = None,
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

    def _identity(x):
        return x

    # Helper for lazy model creation
    models_set = set(mlframe_models) if mlframe_models else None

    def _should_create_model(model_name: str) -> bool:
        """Check if a model should be created based on mlframe_models filter."""
        return models_set is None or model_name in models_set

    if metamodel_func is None:
        metamodel_func = _identity

    if default_regression_scoring is None:
        default_regression_scoring = dict(score_func=mean_absolute_error, response_method="predict", greater_is_better=False)

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

    # ---- multilabel + post-hoc calibration safety gate ----
    # ``CalibratedClassifierCV`` is single-output only; combining it with a
    # MULTILABEL target silently fails inside the wrapper (label-list shape
    # mismatch deep in sklearn). Honour ``MultilabelDispatchConfig.
    # allow_uncalibrated_multi``: when False (default -- strict), refuse the
    # combo loudly so the misconfiguration is visible at config time; when
    # True, drop the calibration request with a warning and continue. No-op
    # when target is not multilabel or no MultilabelDispatchConfig was
    # supplied (legacy call path stays unchanged).
    if target_type is not None and prefer_calibrated_classifiers and multilabel_dispatch_config is not None:
        from .configs import TargetTypes as _TT

        if target_type == _TT.MULTILABEL_CLASSIFICATION:
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

    # 2026-04-24 Session 6: route target_type/n_classes into get_training_configs
    # so the per-strategy classification dispatch (CB MultiLogloss, XGB
    # multi:softprob+num_class, LGB multiclass+num_class) gets injected.
    # Without this, multilabel targets reach CB without loss_function set,
    # and CB's _get_loss_function_for_train tries len(set(label)) on the 2-D
    # ndarray and crashes with TypeError: unhashable type: 'numpy.ndarray'.
    if target_type is not None and "target_type" not in config_params:
        config_params["target_type"] = target_type
    if n_classes is not None and "n_classes" not in config_params:
        config_params["n_classes"] = n_classes
    # 2026-05-08 perf: thread mlframe_models -> get_training_configs so
    # the MLP config block (and its ~14s pytorch / lightning import on
    # first call) is skipped when no neural model is requested.
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
            # ambiguous`` on the per-cell-array comparison surfaced fuzz
            # 3-way c0000 (cb / pandas / multilabel target).
            _is_object_of_arrays = False
            if target_arr is not None and target_arr.dtype == object and target_arr.ndim == 1 and target_arr.shape[0] > 0:
                _first = target_arr[0]
                _is_object_of_arrays = hasattr(_first, "shape") or (hasattr(_first, "__len__") and not isinstance(_first, (str, bytes)))
            if target_arr is not None and target_arr.ndim == 2:
                nlabels = target_arr.shape[1] + 1  # treat as ">2" -> multiclass-style metrics
            elif _is_object_of_arrays:
                try:
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

    if use_robust_eval_metric and subgroups is not None:
        indexed_subgroups = create_fairness_subgroups_indices(
            subgroups=subgroups, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, group_weights={}, cont_nbins=cont_nbins
        )
    else:
        indexed_subgroups = None

    # 2026-04-27 Session 7 batch 6: per-section timers in
    # configure_training_params. Surfaced when a 9M-row prod run showed
    # ``select_target done in 18.2s`` vs ~0.5s on smaller earlier
    # runs. Three candidate hot-spots: get_training_configs (called
    # twice — CPU + GPU), get_df_memory_consumption(deep=False), and
    # the GPU probe (cached nvidia-smi subprocess). The timers below
    # localise the spend so the operator can see the breakdown
    # without instrumenting by hand.
    _t0_cfg = timer()
    cpu_configs = get_training_configs(has_gpu=False, subgroups=indexed_subgroups, **config_params)
    _t_cpu_cfg = timer() - _t0_cfg
    _t0_cfg = timer()
    gpu_configs = get_training_configs(has_gpu=None, subgroups=indexed_subgroups, **config_params)
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

    # Skip expensive GPU probe (nvidia-smi subprocess ~0.5s) when GPU configs
    # are disabled or CatBoost is explicitly on CPU -- the result is unused.
    _t0_gpu = timer()
    cb_task_type = config_params.get("cb_kwargs", {}).get("task_type")
    cb_devices = config_params.get("cb_kwargs", {}).get("devices")
    if not prefer_gpu_configs or cb_task_type == "CPU":
        all_gpus = {}
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
        # 2026-05-10: thread target_type through so the ensemble path
        # (score_ensemble -> _process_single_ensemble_method ->
        # _build_configs_from_params) can gate render_multi_target_panels
        # via DataConfig.target_type. Without this the ensemble report
        # block goes through report_model_perf with target_type=None and
        # auto_dispatch falls back to "fire LTR/multilabel/multiclass
        # panels for any target with group_ids set" — wrong on regression.
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
        # Defensively pre-set the polars-fastpath sticky flag (2026-04-24).
        # Background: ``_predict_with_fallback`` lazily flips this attribute
        # to True after the FIRST polars-fastpath dispatch miss, so the
        # short-circuit fires only on the SECOND predict call onward.
        # That's fine for re-using a single fitted model (VAL -> TEST), but
        # in a suite each weight-schema iteration calls ``sklearn.clone()``
        # on this base ``_cb_model`` -- and clone strips non-param attrs,
        # giving every fresh CB instance a blank flag. The 2026-04-24 prod
        # log captured the symptom: CB uniform AND CB recency BOTH paid
        # the polars-miss + 2-3 s pandas-conversion roundtrip on their
        # first TEST predict, with WARN noise on each.
        # We know empirically (every prod run since 2026-04-19) that CB
        # 1.2.x's ``_set_features_order_data_polars_categorical_column``
        # has dispatch gaps on our nullable-Categorical / Enum schema, so
        # opting CB into pandas at predict time is bestthing -- bypasses
        # the doomed retry on success, costs nothing on failure. Set on
        # the base instance so ``clone()`` carries the param-equivalent
        # state forward (sklearn.clone preserves ``get_params()`` keys;
        # for the attr to survive clone we re-assert it inside
        # ``train_eval.py:process_model``'s clone call too -- but writing
        # it here is the ergonomic source of truth).
        try:
            _cb_model._mlframe_polars_fastpath_broken = True
        except Exception:
            # CB Python class is permissive about attributes; slot-only
            # forks could refuse -- degrade to "pay first-call retry".
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

    # 2026-04-24 Session 6: per-strategy multilabel-wrap helper. Strategies
    # without native (N, K) target support (HGB, XGB-via-MultiOutputClassifier,
    # LGB, Linear) need MultiOutputClassifier when target is multilabel.
    # Inner-estimator early_stopping that depends on eval_set must be disabled
    # because the outer wrapper doesn't slice eval_set per label -- without
    # an eval_set the inner fit would crash ("at least one dataset and eval
    # metric is required for evaluation").
    def _wrap_for_multilabel_if_needed(estimator, strategy_cls):
        if use_regression or target_type is None or not hasattr(target_type, "name") or target_type.name != "MULTILABEL_CLASSIFICATION":
            return estimator
        # Disable eval_set-dependent early stopping on the inner estimator.
        try:
            params = estimator.get_params()
        except Exception:
            params = {}
        _patch = {}
        if "early_stopping_rounds" in params and params.get("early_stopping_rounds") is not None:
            _patch["early_stopping_rounds"] = None
        # XGB sklearn >=2 uses callbacks for early stopping too; strip them.
        if "callbacks" in params and params.get("callbacks"):
            _patch["callbacks"] = None
        if _patch:
            try:
                estimator.set_params(**_patch)
            except Exception:
                pass
        return strategy_cls().wrap_multilabel(
            estimator,
            target_type,
            multilabel_config=multilabel_dispatch_config,
            n_labels=n_classes,
        )

    hgb_params = None
    if _should_create_model("hgb"):
        from .strategies import HGBStrategy as _HGBS

        _hgb_est = (
            HistGradientBoostingRegressor(**configs.HGB_GENERAL_PARAMS)
            if use_regression
            else _wrap_for_multilabel_if_needed(
                HistGradientBoostingClassifier(**configs.HGB_GENERAL_PARAMS),
                _HGBS,
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
            use_flaml_zeroshot=use_flaml_zeroshot,
            xgboost_verbose=xgboost_verbose,
            metamodel_func=metamodel_func,
        )
        # XGB sklearn wrapper rejects 2-D y unless we use multi_strategy='multi_output_tree'
        # (WIP in 3.x). Default to MultiOutputClassifier per Session-6 design.
        from .strategies import XGBoostStrategy as _XGBS

        xgb_params["model"] = _wrap_for_multilabel_if_needed(xgb_params["model"], _XGBS)

    lgb_params = None
    if _should_create_model("lgb"):
        lgb_params = _configure_lightgbm_params(
            configs=configs,
            cpu_configs=cpu_configs,
            use_regression=use_regression,
            prefer_cpu_for_lightgbm=prefer_cpu_for_lightgbm,
            prefer_calibrated_classifiers=prefer_calibrated_classifiers,
            use_flaml_zeroshot=use_flaml_zeroshot,
            metamodel_func=metamodel_func,
        )
        # LGB has no native multilabel -- wrap with MultiOutputClassifier.
        from .strategies import TreeModelStrategy as _LGBS

        lgb_params["model"] = _wrap_for_multilabel_if_needed(lgb_params["model"], _LGBS)

    mlp_params = None
    if _should_create_model("mlp"):
        mlp_params = _configure_mlp_params(
            configs=configs,
            config_params=config_params,
            use_regression=use_regression,
            metamodel_func=metamodel_func,
            target_type=target_type,
        )

    ngb_params = None
    if _should_create_model("ngb"):
        # 2026-05-07: target-type-aware Dist for NGBClassifier.
        # Default ``Dist=Bernoulli`` (binary only) crashes on K>2 with
        # ``IndexError: index out of bounds``. For multiclass we need
        # ``Dist=k_categorical(K)``. NGBoost has no native multilabel /
        # ranker, so those target types fall through to the default
        # (likely with a downstream error if reached -- they should be
        # filtered earlier when the suite checks per-strategy multilabel
        # / ranking flags).
        ngb_init_kwargs = dict(configs.NGB_GENERAL_PARAMS)
        from .configs import TargetTypes as _TT

        if not use_regression and target_type == _TT.MULTICLASS_CLASSIFICATION:
            try:
                from ngboost.distns import k_categorical

                # n_classes pulled from the actual y -- NGB needs the
                # exact K to size the categorical Dist's internal
                # parameter array. Fall back to inspecting train_target
                # via config_params (where train_target lives at this
                # call layer).
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
        linear_config_kwargs = {"model_type": model_type}
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
        from .strategies import LinearModelStrategy as _LMS

        _linear_est = _wrap_for_multilabel_if_needed(_linear_est, _LMS)
        linear_model_params[model_type] = dict(model=metamodel_func(_linear_est))

    # Get individual params (may be None if not in mlframe_models)
    linear_params = linear_model_params.get("linear")
    ridge_params = linear_model_params.get("ridge")
    lasso_params = linear_model_params.get("lasso")
    elasticnet_params = linear_model_params.get("elasticnet")
    huber_params = linear_model_params.get("huber")
    ransac_params = linear_model_params.get("ransac")
    sgd_params = linear_model_params.get("sgd")

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
        estimator=(
            metamodel_func((flaml_zeroshot.LGBMRegressor if use_flaml_zeroshot else LGBMRegressor)(**configs.LGB_GENERAL_PARAMS))
            if use_regression
            else (flaml_zeroshot.LGBMClassifier if use_flaml_zeroshot else LGBMClassifier)(**configs.LGB_GENERAL_PARAMS)
        ),
        fit_params=lgb_fit_params,
        cat_features=cat_features,
        scoring=rfecv_scoring,
        **rfecv_params,
    )

    xgb_rfecv = RFECV(
        estimator=(
            metamodel_func((flaml_zeroshot.XGBRegressor if use_flaml_zeroshot else XGBRegressor)(**configs.XGB_GENERAL_PARAMS))
            if use_regression
            else (flaml_zeroshot.XGBClassifier if use_flaml_zeroshot else XGBClassifier)(
                **(configs.XGB_CALIB_CLASSIF if prefer_calibrated_classifiers else configs.XGB_GENERAL_CLASSIF)
            )
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

    return (
        common_params,
        models_params,
        cb_rfecv,
        lgb_rfecv,
        xgb_rfecv,
        cpu_configs,
        gpu_configs,
    )


__all__ = [
    "train_and_evaluate_model",
    "configure_training_params",
    "_build_configs_from_params",
    "run_confidence_analysis",
]
