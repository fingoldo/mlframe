"""``train_and_evaluate_model`` carved out of ``mlframe.training.trainer``.

Bound back into the parent's namespace via re-export at the parent's
module bottom so historical
``from mlframe.training.trainer import train_and_evaluate_model``
resolves transparently.
"""
from __future__ import annotations

import copy
from timeit import default_timer as timer
from functools import partial
from os.path import exists
from types import SimpleNamespace
from typing import Any, Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ._reporting_configs import ConfidenceAnalysisConfig, NamingConfig, PredictionsContainer, ReportingConfig
    from ._training_runtime_configs import DataConfig, MetricsConfig, OutputConfig, TrainingControlConfig

import numpy as np
import polars as pl

from mlframe.metrics.core import compute_probabilistic_multiclass_error
from mlframe.reporting.async_render_hooks import log_render_queued, render_queued_mark
from .phases import phase
from .utils import maybe_clean_ram_adaptive as _maybe_clean_ram

# Heavy optional deps: defer failures to first actual use so `import mlframe.training` stays cheap and does not crash when a given backend is not installed.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


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
from ._feature_name_sanitize import sanitize_frame_columns as _sanitize_frame_columns
from ._training_loop import (  # noqa: F401
    _SigmoidAdapter, _PostHocCalibratedModel,
    _PostHocMultiCalibratedModel, _PerClassIsotonicCalibrator,
    _maybe_apply_posthoc_calibration, _train_model_with_fallback,
)
from ._data_helpers import (  # noqa: F401
    _setup_eval_set, _setup_early_stopping_callback,
)
from ._calib_oof_outputs import compute_calib_and_oof_outputs, maybe_run_confidence_analysis
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

# Split-metrics emitters carved into a sibling to keep this orchestration module under the LOC budget.
from ._trainer_train_and_evaluate_helpers import _run_val_split_metrics, _run_test_split_metrics


from ._trainer_train_and_evaluate_parts import (  # noqa: F401  -- carved helpers
    logger,
    _oof_train_timestamps,
    _train_envelope_stats,
    _train_and_evaluate_into_non_booster_fits,
    _train_and_evaluate_cap_iter_boosters,
    _train_and_evaluate_use_cache_exists_model,
    _train_and_evaluate_df_none_train_df,
    _train_and_evaluate_range_no_test_fix,
    _train_and_evaluate_forwarding_happens_through_apply,
    _train_and_evaluate_model_none_pre_pipeline,
    _train_and_evaluate_val_df_none,
    _train_and_evaluate_train_df_none,
    _train_and_evaluate_nest_lightning_checkpoints_csv,
    _train_and_evaluate_score_ensemble_can_pick,
    _train_and_evaluate_train_runs_sequentially_may,
    _train_and_evaluate_run_test_df_none,
    _train_and_evaluate_was_trained_instead_silently,
)


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
    oof_has_time: Optional[bool] = None,
    oof_random_seed: int = 42,
    cur_target_name: str | None = None,
):
    """Train and evaluate a machine learning model with comprehensive metrics and optional caching.

    ``oof_has_time`` selects the OOF splitter: when True the K-fold OOF pass forward-chains over the train rows' ``timestamps`` (temporal honesty,
    no future-into-past leak) instead of a shuffled ``KFold``; None (the suite passes the shared split policy's decision) means False here. ``oof_random_seed`` is the suite master seed plumbed
    into the i.i.d. shuffled-KFold OOF path (replaces the historical hardcoded 42 so the OOF surface varies with the
    run seed for variance/stability analysis).

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
    cur_target_name : str, optional
        The actual target this call is training (original or a composite-discovered derived
        target). Threaded into the process-wide pre-pipeline cache key so two different targets
        sharing the same model type never replay each other's fitted pre_pipeline -- surfaced live
        as sklearn's "Feature names should match those that were passed during fit" inside a
        composite-target-discovery suite call, where two targets with byte-identical structurally
        equivalent pipelines but different row-wise-extension column sets could collide.

    Returns
    -------
    tuple
        (result_namespace, train_df, val_df, test_df) where result_namespace contains
        model, predictions, metrics, and other training artifacts.
    """
    # Lazy import of parent-resident helpers: ``.trainer`` re-imports this sibling at its bottom, so a top-level ``from .trainer
    # import ...`` would create a hard cycle the meta-test flags.
    _mono_patience_cfg: Any = None
    t0_metrics: Any = None
    from .trainer import ConfidenceAnalysisConfig, FeatureImportanceConfig, OutputConfig, PredictionsContainer, _extract_targets_from_indices, _prepare_train_df_for_fitting, _setup_model_info_and_paths, _setup_sample_weight, _update_model_name_after_training, _validate_infinity_and_columns, _validate_target_values
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
    calib_df = data.calib_df
    calib_target = data.calib_target
    group_ids = data.group_ids
    sample_weight = data.sample_weight
    timestamps = data.timestamps
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

    # Thread ``TrainingBehaviorConfig.monotonic_decline_patience`` (default 20; None disables) to the boosters:
    # for cb it travels via ``callback_params`` (consumed by ``_setup_early_stopping_callback``), for the lgb /
    # xgb shims it is a ``.fit()`` kwarg read from ``fit_params``. A value already present in callback_params / fit_params (explicit per-call override) wins.
    _mono_beh = getattr(getattr(control, "behavior", None), "__dict__", None)
    if _mono_beh is not None and "monotonic_decline_patience" in _mono_beh:
        _mono_patience_cfg = _mono_beh["monotonic_decline_patience"]
    else:
        _mono_patience_cfg = getattr(control, "monotonic_decline_patience", 20)
    if model_category == "cb" or callback_params:
        # Only materialise callback_params for cb (which consumes the key) or when the caller already passed one -- avoid injecting an empty callbacks kwarg
        # into non-booster fits that previously got None.
        callback_params = dict(callback_params or {})
        callback_params.setdefault("monotonic_decline_patience", _mono_patience_cfg)
    fit_params = _train_and_evaluate_into_non_booster_fits(model_category, fit_params, _mono_patience_cfg)

    # Thread the live training-performance surfaces to the boosters' shared UniversalCallback. Same behavior-config plumbing as monotonic_decline_patience
    # above. These only choose how the per-iteration trajectory is SURFACED
    # while the fit runs -- the trajectory itself is recorded on the callback either way and harvested into the run
    # metadata, so turning the log line off costs no information.
    #   live_trainperf_plot   -> progress_widget          (default True; a hard no-op outside a notebook)
    #   live_trainperf_report -> report_progress_to_log   (default False; the periodic "iter=..., best=..." line)
    if model_category == "cb" or callback_params:
        _live_plot = _mono_beh.get("live_trainperf_plot", True) if _mono_beh is not None else getattr(control, "live_trainperf_plot", True)
        _live_report = _mono_beh.get("live_trainperf_report", False) if _mono_beh is not None else getattr(control, "live_trainperf_report", False)
        callback_params = dict(callback_params or {})
        callback_params.setdefault("progress_widget", bool(_live_plot))
        callback_params.setdefault("report_progress_to_log", bool(_live_report))

    # Thread per-iteration metric-capture knobs to the boosters (meta-learning / HPO-from-early-observation). Same behavior-config plumbing as
    # monotonic_decline_patience: cb via callback_params, lgb / xgb via fit_params. ``capture_iteration_metrics`` defaults to None in the config -> resolve to
    # the family default (OFF for boosters, since re-predicting val every round is non-trivial; the user opts in explicitly).
    _cap_iter_cfg = _mono_beh.get("capture_iteration_metrics") if _mono_beh is not None else getattr(control, "capture_iteration_metrics", None)
    _iter_stride_cfg = _mono_beh.get("iteration_metrics_stride", 1) if _mono_beh is not None else getattr(control, "iteration_metrics_stride", 1)
    _cap_iter_boosters = bool(_cap_iter_cfg) if _cap_iter_cfg is not None else False
    callback_params, fit_params = _train_and_evaluate_cap_iter_boosters(_cap_iter_boosters, model_category, callback_params, _iter_stride_cfg, fit_params)

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
    binary_panels = reporting.binary_panels
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

    train_preds = predictions.train_preds
    train_probs = predictions.train_probs
    val_preds = predictions.val_preds
    val_probs = predictions.val_probs
    test_preds = predictions.test_preds
    test_probs = predictions.test_probs

    _maybe_clean_ram()

    columns: list[Any] = []
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

    model, pre_pipeline = _train_and_evaluate_use_cache_exists_model(use_cache, model_file_name, trusted_root, model, pre_pipeline)
    # Continue to training - model remains as originally passed

    if fit_params is None:
        fit_params = {}
    else:
        fit_params = copy.copy(fit_params)

    train_target, val_target, test_target = _extract_targets_from_indices(target, train_idx, val_idx, test_idx, train_target, val_target, test_target)

    train_df, val_df = _train_and_evaluate_df_none_train_df(df, train_df, train_idx, real_drop_columns, val_df, val_idx)

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
    _pre_pipeline_groups = _train_and_evaluate_range_no_test_fix(group_ids, train_idx, train_df, _pre_pipeline_groups)

    # Extract train-subset sample_weight BEFORE FS runs so weight-aware MRMR / RFECV (when stamped with the
    # _mlframe_use_sample_weights_in_fs_ marker by _build_pre_pipelines) can receive it via fit_params.
    # _setup_sample_weight runs AFTER FS at L730 and writes to the model's fit_params dict; the FS-side
    # forwarding happens through _apply_pre_pipeline_transforms -> _passthrough_cols_fit_transform.
    _pre_pipeline_sample_weight = None
    _pre_pipeline_sample_weight = _train_and_evaluate_forwarding_happens_through_apply(sample_weight, train_idx, _pre_pipeline_sample_weight)

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
        # This was the ONLY caller of ``_apply_pre_pipeline_transforms`` and never passed
        # ``target_name``, so it silently defaulted to None on every call. The process-wide
        # ``_PRE_PIPELINE_CACHE``'s content fingerprint alone then wrongly matched two DIFFERENT
        # targets/runs whose train/val frames happened to be byte-identical (e.g. two suite calls
        # built from the same RNG seed, or two composite-discovered targets sharing the same base
        # X), and replayed one target's fitted pre_pipeline (feature_names_in_ from ITS row-wise-
        # extension columns) onto the other's test_df -- surfaced live as sklearn's "Feature names
        # should match those that were passed during fit" inside a composite-target-discovery
        # suite call. ``naming.model_name`` is just the model TYPE ("linear"), identical across
        # every target using that model, so it cannot discriminate here -- ``cur_target_name`` (the
        # actual target being trained) is the real discriminator the cache key's own docstring
        # describes.
        target_name=cur_target_name,
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
            "Feature selection removed all features for %s - skipping training. " "This typically means no features had predictive power for the target.",
            model_name,
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

    _orig_train_df, _orig_val_df = _train_and_evaluate_model_none_pre_pipeline(model, pre_pipeline, skip_pre_pipeline_transform, train_df, val_df, _orig_train_df, _orig_val_df)

    model, model_obj, val_target = _train_and_evaluate_val_df_none(val_df, val_target, control, model_category, sample_weight, val_idx, group_ids, callback_params, model_obj, model_type_name, verbose, fit_params, model, oof_random_seed)

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
                from mlframe.training.reporting.shared import style_with_caption as _style_with_caption
                ipython_display(_style_with_caption(train_df.head(display_sample_size), f"{model_name} features head"))
                ipython_display(_style_with_caption(train_df.tail(display_sample_size), f"{model_name} features tail"))

            _train_and_evaluate_train_df_none(train_df, model_name, show_feature_names)

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
                _train_and_evaluate_nest_lightning_checkpoints_csv(model_file_name, model)
                model, best_iter = _train_model_with_fallback(
                    model=model,
                    model_obj=model_obj,
                    model_type_name=model_type_name,
                    train_df=train_df,
                    train_target=train_target,
                    fit_params=fit_params,
                    verbose=bool(verbose),
                )

                # Handle failed model training (e.g., dtype incompatibility)
                if model is None:
                    logger.warning("Model %s training failed - skipping evaluation", model_type_name)
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
            _train_and_evaluate_score_ensemble_can_pick(oof_n_splits, just_evaluate, model_type_name, train_target, model, train_df, oof_random_seed, group_ids, train_idx, oof_has_time, _pre_pipeline_sample_weight, timestamps)

    metrics_out: dict[str, Any] = {"train": {}, "val": {}, "test": {}, "best_iter": best_iter}

    _render_mark = render_queued_mark()
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
        _y_train_envelope_stats = _train_envelope_stats(train_target, reporting.mase_seasonality)

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
            binary_panels=binary_panels,
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
            # Full-length row timestamps; _compute_split_metrics slices per
            # split idx to gate the residual-vs-time / metric-over-time panels.
            split_timestamps=timestamps,
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
        columns, train_preds, train_probs = _train_and_evaluate_train_runs_sequentially_may(splits_config, metrics_out, has_val, has_test, common_metrics_params, columns, train_preds, train_probs)

        _val_cfg = next((c for c in splits_config if c[0] == "val" and c[-1]), None)
        _run_test = compute_testset_metrics and ((test_idx is not None and len(test_idx) > 0) or test_df is not None)

        _train_and_evaluate_run_test_df_none(_run_test, df, test_df, train_df)

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
        # Concurrent ThreadPoolExecutor was tried but matplotlib figure creation from concurrent threads races on pyplot's shared state even with Agg backend, producing "Argument must be an image or collection" errors in calibration plots. Sequential path is correct.
        with phase("compute_split_metrics", split="val"):
            val_res = _run_val_split_metrics(_val_cfg, metrics_out, has_test, common_metrics_params)
        with phase("compute_split_metrics", split="test"):
            test_res = _run_test_split_metrics(
                _run_test, metrics_out, test_df, test_target, test_idx,
                test_preds, test_probs, test_details, common_metrics_params,
            )

        if val_res is not None:
            val_preds, val_probs, columns = val_res
        if test_res is not None:
            test_preds, test_probs, columns = test_res

        # Same test_idx-slicing convention as `_pre_pipeline_sample_weight` above (train_idx-sliced),
        # so the confidence-analysis diagnostic reflects the same weighted objective the real model
        # was trained on instead of silently reverting to unweighted.
        _test_sample_weight = None
        _test_sample_weight = _train_and_evaluate_was_trained_instead_silently(sample_weight, test_idx, _test_sample_weight)

        maybe_run_confidence_analysis(
            run_test=_run_test,
            confidence=confidence,
            test_df=test_df,
            test_target=test_target,
            test_probs=test_probs,
            fit_params=fit_params,
            model_type_name=model_type_name,
            figsize=figsize,
            verbose=verbose,
            sample_weight=_test_sample_weight,
        )

    if (compute_trainset_metrics or compute_valset_metrics or compute_testset_metrics) and verbose:
        logger.info("  Metrics computation done -- %.1fs", timer() - t0_metrics)
        log_render_queued(_render_mark)

    _maybe_clean_ram()

    _calib_probs_out, _calib_target_out, _calib_preds_out, _oof_preds_out, _oof_probs_out, _oof_target_out = compute_calib_and_oof_outputs(
        model=model,
        calib_df=calib_df,
        calib_target=calib_target,
        real_drop_columns=real_drop_columns,
        pre_pipeline=pre_pipeline,
        skip_pre_pipeline_transform=skip_pre_pipeline_transform,
        skip_preprocessing=skip_preprocessing,
        fit_params=fit_params,
        model_type_name=model_type_name,
        model_name=model_name,
        row_wise_extensions_config=control.row_wise_extensions_config,
    )

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
            oof_target=_oof_target_out,
            calib_probs=_calib_probs_out,
            calib_target=_calib_target_out,
            calib_preds=_calib_preds_out,
            metrics=metrics_out,
            columns=columns,
            pre_pipeline=pre_pipeline,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            trainset_features_stats=trainset_features_stats,
            plot_file=plot_file or "",  # this model's chart prefix; charts rendered for it after the fit (composite y-scale) reuse it
            # The chart title and split date details of THIS model, so charts rendered for it later (composite y-scale)
            # carry the same header as its own charts instead of a hand-built shorter one.
            chart_model_name=model_name,
            chart_split_details={"val": val_details or "", "test": test_details or ""},
        ),
        _orig_train_df,
        _orig_val_df,
        _orig_test_df,
    )
