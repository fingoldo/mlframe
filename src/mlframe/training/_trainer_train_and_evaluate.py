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
from typing import Optional, TYPE_CHECKING
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
from types import SimpleNamespace as _SimpleNamespace


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
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    st._mono_patience_cfg = None
    st.t0_metrics = None
    from .trainer import ConfidenceAnalysisConfig, FeatureImportanceConfig, OutputConfig, PredictionsContainer, _extract_targets_from_indices, _prepare_train_df_for_fitting, _setup_model_info_and_paths, _setup_sample_weight, _update_model_name_after_training, _validate_infinity_and_columns, _validate_target_values
    from IPython.display import display as ipython_display

    # Initialize optional configs with defaults
    if confidence is None:
        confidence = ConfidenceAnalysisConfig()
    if predictions is None:
        predictions = PredictionsContainer()

    st.df = data.df
    train_df = data.train_df
    st.val_df = data.val_df
    st.test_df = data.test_df
    st.target = data.target
    st.train_target = data.train_target
    st.val_target = data.val_target
    st.test_target = data.test_target
    st.train_idx = data.train_idx
    st.val_idx = data.val_idx
    st.test_idx = data.test_idx
    st.calib_df = data.calib_df
    st.calib_target = data.calib_target
    st.group_ids = data.group_ids
    st.sample_weight = data.sample_weight
    st.timestamps = data.timestamps
    st.drop_columns = list(data.drop_columns) if data.drop_columns else []
    st.target_label_encoder = data.target_label_encoder
    st.skip_infinity_checks = data.skip_infinity_checks
    st.n_features = data.n_features

    st.verbose = control.verbose
    st.use_cache = control.use_cache
    st.just_evaluate = control.just_evaluate
    st.compute_trainset_metrics = control.compute_trainset_metrics
    st.compute_valset_metrics = control.compute_valset_metrics
    st.compute_testset_metrics = control.compute_testset_metrics
    st.pre_pipeline = control.pre_pipeline
    st.skip_pre_pipeline_transform = control.skip_pre_pipeline_transform
    st.skip_preprocessing = control.skip_preprocessing
    st.fit_params = control.fit_params
    st.callback_params = control.callback_params
    st.model_category = control.model_category

    # Thread ``TrainingBehaviorConfig.monotonic_decline_patience`` (default 20; None disables) to the boosters:
    # for cb it travels via ``callback_params`` (consumed by ``_setup_early_stopping_callback``), for the lgb /
    # xgb shims it is a ``.fit()`` kwarg read from ``fit_params``. A value already present in callback_params / fit_params (explicit per-call override) wins.
    st._mono_beh = getattr(getattr(control, "behavior", None), "__dict__", None)
    if st._mono_beh is not None and "monotonic_decline_patience" in st._mono_beh:
        st._mono_patience_cfg = st._mono_beh["monotonic_decline_patience"]
    else:
        st._mono_patience_cfg = getattr(control, "monotonic_decline_patience", 20)
    if st.model_category == "cb" or st.callback_params:
        # Only materialise callback_params for cb (which consumes the key) or when the caller already passed one -- avoid injecting an empty callbacks kwarg
        # into non-booster fits that previously got None.
        st.callback_params = dict(st.callback_params or {})
        st.callback_params.setdefault("monotonic_decline_patience", st._mono_patience_cfg)
    st.fit_params = _train_and_evaluate_into_non_booster_fits(st.model_category, st.fit_params, st._mono_patience_cfg)

    # Thread the live training-performance surfaces to the boosters' shared UniversalCallback. Same behavior-config plumbing as monotonic_decline_patience
    # above. These only choose how the per-iteration trajectory is SURFACED
    # while the fit runs -- the trajectory itself is recorded on the callback either way and harvested into the run
    # metadata, so turning the log line off costs no information.
    #   live_trainperf_plot   -> progress_widget          (default True; a hard no-op outside a notebook)
    #   live_trainperf_report -> report_progress_to_log   (default False; the periodic "iter=..., best=..." line)
    _train_and_evaluate_m_step1_live_trainperf_report(st, control)

    # Thread per-iteration metric-capture knobs to the boosters (meta-learning / HPO-from-early-observation). Same behavior-config plumbing as
    # monotonic_decline_patience: cb via callback_params, lgb / xgb via fit_params. ``capture_iteration_metrics`` defaults to None in the config -> resolve to
    # the family default (OFF for boosters, since re-predicting val every round is non-trivial; the user opts in explicitly).
    st._cap_iter_cfg = st._mono_beh.get("capture_iteration_metrics") if st._mono_beh is not None else getattr(control, "capture_iteration_metrics", None)
    st._iter_stride_cfg = st._mono_beh.get("iteration_metrics_stride", 1) if st._mono_beh is not None else getattr(control, "iteration_metrics_stride", 1)
    st._cap_iter_boosters = bool(st._cap_iter_cfg) if st._cap_iter_cfg is not None else False
    st.callback_params, st.fit_params = _train_and_evaluate_cap_iter_boosters(st._cap_iter_boosters, st.model_category, st.callback_params, st._iter_stride_cfg, st.fit_params)

    st.nbins = metrics.nbins
    st.custom_ice_metric = metrics.custom_ice_metric
    st.custom_rice_metric = metrics.custom_rice_metric
    st.subgroups = metrics.subgroups
    st.train_details = metrics.train_details
    st.val_details = metrics.val_details
    st.test_details = metrics.test_details

    st.figsize = reporting.figsize
    st.print_report = reporting.print_report
    st.show_perf_chart = reporting.show_perf_chart
    st.show_fi = reporting.show_fi
    st.fi_config = reporting.feature_importance_config or FeatureImportanceConfig()
    st.fi_kwargs = dict(
        figsize=st.fi_config.figsize,
        num_factors=st.fi_config.num_factors,
        positive_fi_only=st.fi_config.positive_fi_only,
        show_plots=st.fi_config.show_plots,
        max_zero_fi_to_plot=getattr(st.fi_config, "max_zero_fi_to_plot", 4),
    )
    display_sample_size = reporting.display_sample_size
    st.show_feature_names = reporting.show_feature_names
    st.show_prob_histogram = reporting.show_prob_histogram
    st.prob_histogram_yscale = reporting.prob_histogram_yscale
    st.show_inline_population_labels = reporting.show_inline_population_labels
    st.title_metrics_tokens = reporting.title_metrics_tokens
    st.plot_outputs = reporting.plot_outputs
    st.plot_dpi = reporting.plot_dpi
    st.binary_panels = reporting.binary_panels
    st.multiclass_panels = reporting.multiclass_panels
    st.multilabel_panels = reporting.multilabel_panels
    st.ltr_panels = reporting.ltr_panels
    st.quantile_panels = reporting.quantile_panels
    # ``quantile_alphas`` arrives via fit_params (per-fit context), not via ReportingConfig - it depends on which alphas the model was trained on, not on display preference. Resolved at the _compute_split_metrics call site.
    st.quantile_alphas = None
    if hasattr(model, "_mlframe_quantile_alphas"):
        st.quantile_alphas = getattr(model, "_mlframe_quantile_alphas", None)

    if output is None:
        output = OutputConfig()
    st.plot_file = output.plot_file
    st.data_dir = output.data_dir
    st.models_subdir = output.models_dir

    model_name = naming.model_name
    st.model_name_prefix = naming.model_name_prefix

    st.train_preds = predictions.train_preds
    st.train_probs = predictions.train_probs
    st.val_preds = predictions.val_preds
    st.val_probs = predictions.val_probs
    st.test_preds = predictions.test_preds
    st.test_probs = predictions.test_probs

    _maybe_clean_ram()

    st.columns = []
    st.best_iter = None

    st._orig_train_df = train_df
    st._orig_val_df = st.val_df
    st._orig_test_df = st.test_df

    st.real_drop_columns = _validate_infinity_and_columns(
        df=st.df,
        train_df=train_df,
        skip_infinity_checks=st.skip_infinity_checks,
        drop_columns=st.drop_columns,
    )

    if not st.custom_ice_metric:
        st.custom_ice_metric = partial(compute_probabilistic_multiclass_error, nbins=st.nbins)

    st.model_obj, st.model_type_name, model_name, st.plot_file, st.model_file_name = _setup_model_info_and_paths(
        model=model,
        model_name=model_name,
        model_name_prefix=st.model_name_prefix,
        plot_file=st.plot_file,
        data_dir=st.data_dir,
        models_subdir=st.models_subdir,
    )

    model, st.pre_pipeline = _train_and_evaluate_use_cache_exists_model(st.use_cache, st.model_file_name, trusted_root, model, st.pre_pipeline)
    # Continue to training - model remains as originally passed

    if st.fit_params is None:
        st.fit_params = {}
    else:
        st.fit_params = copy.copy(st.fit_params)

    st.train_target, st.val_target, st.test_target = _extract_targets_from_indices(st.target, st.train_idx, st.val_idx, st.test_idx, st.train_target, st.val_target, st.test_target)

    train_df, st.val_df = _train_and_evaluate_df_none_train_df(st.df, train_df, st.train_idx, st.real_drop_columns, st.val_df, st.val_idx)

    # Decategorise float-typed pandas categorical columns BEFORE the pre_pipeline runs (RFECV inner CB / XGB inside the pre_pipeline would otherwise reject them; see helper docstring).
    train_df, st.val_df, st.test_df = _decategorise_float_cat_columns(
        train_df,
        val_df=st.val_df,
        test_df=st.test_df,
    )

    # Thread group_ids into the pre_pipeline fit so RFECV(cv=GroupKFold())
    # and grouped MRMR receive the same sample-grouping signal the suite-level
    # callers already pass into trainer.fit. Only forwarded on train+val sample
    # range (no test). fix audit row FS-P1-1.
    st._pre_pipeline_groups = None
    st._pre_pipeline_groups = _train_and_evaluate_range_no_test_fix(st.group_ids, st.train_idx, train_df, st._pre_pipeline_groups)

    # Extract train-subset sample_weight BEFORE FS runs so weight-aware MRMR / RFECV (when stamped with the
    # _mlframe_use_sample_weights_in_fs_ marker by _build_pre_pipelines) can receive it via fit_params.
    # _setup_sample_weight runs AFTER FS at L730 and writes to the model's fit_params dict; the FS-side
    # forwarding happens through _apply_pre_pipeline_transforms -> _passthrough_cols_fit_transform.
    st._pre_pipeline_sample_weight = None
    st._pre_pipeline_sample_weight = _train_and_evaluate_forwarding_happens_through_apply(st.sample_weight, st.train_idx, st._pre_pipeline_sample_weight)

    train_df, st.val_df = _apply_pre_pipeline_transforms(
        model=model,
        pre_pipeline=st.pre_pipeline,
        train_df=train_df,
        val_df=st.val_df,
        train_target=st.train_target,
        skip_pre_pipeline_transform=st.skip_pre_pipeline_transform,
        skip_preprocessing=st.skip_preprocessing,
        use_cache=st.use_cache,
        model_file_name=st.model_file_name,
        verbose=st.verbose,
        selector_passthrough_cols=(list(st.fit_params.get("text_features") or []) + list(st.fit_params.get("embedding_features") or [])) or None,
        groups=st._pre_pipeline_groups,
        sample_weight=st._pre_pipeline_sample_weight,
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
    st.val_df = _sanitize_frame_columns(st.val_df)

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
                pre_pipeline=st.pre_pipeline,
                train_od_idx=train_od_idx,
                val_od_idx=val_od_idx,
                trainset_features_stats=trainset_features_stats,
            ),
            None,
            None,
            None,
        )

    st._orig_train_df, st._orig_val_df = _train_and_evaluate_model_none_pre_pipeline(model, st.pre_pipeline, st.skip_pre_pipeline_transform, train_df, st.val_df, st._orig_train_df, st._orig_val_df)

    model, st.model_obj, st.val_target = _train_and_evaluate_val_df_none(st.val_df, st.val_target, control, st.model_category, st.sample_weight, st.val_idx, st.group_ids, st.callback_params, st.model_obj, st.model_type_name, st.verbose, st.fit_params, model, oof_random_seed)

    if model is not None and st.fit_params:
        # Two-phase coupling with FS (FS runs at the _apply_pre_pipeline_transforms call upstream):
        # FS may drop columns from ``train_df``, so the ``cat_features`` declared in ``fit_params``
        # can now reference columns that no longer exist. ``_filter_categorical_features`` reconciles
        # the cat list against the post-FS frame; reordering this block relative to the FS call
        # would silently feed CatBoost/LightGBM stale cat_features and trigger
        # "feature_name not found" at fit time.
        _filter_categorical_features(st.fit_params, train_df, val_df=st.val_df, test_df=st.test_df)

    if model is not None:
        if (not st.use_cache) or (not exists(st.model_file_name)):
            _setup_sample_weight(st.sample_weight, st.train_idx, st.model_obj, st.fit_params)
            if st.verbose:
                logger.info("training dataset shape: %s", train_df.shape)

            if display_sample_size:
                from mlframe.training.reporting.shared import style_with_caption as _style_with_caption
                ipython_display(_style_with_caption(train_df.head(display_sample_size), f"{model_name} features head"))
                ipython_display(_style_with_caption(train_df.tail(display_sample_size), f"{model_name} features tail"))

            _train_and_evaluate_train_df_none(train_df, model_name, st.show_feature_names)

            train_df, st.fit_params = _prepare_train_df_for_fitting(train_df, model, st.model_type_name, st.fit_params)

            _maybe_clean_ram()
            if st.verbose:
                logger.info("Training the model...")

            if isinstance(st.train_target, pl.Series):
                st.train_target = st.train_target.to_numpy()

            # Detect classification vs regression from the model type
            # name suffix (covers all four GBM backends + sklearn linear
            # + MultiOutputClassifier + ClassifierChain). Used by
            # ``_validate_target_values`` to flag single-class collapse
            # before the per-backend C++ crash.
            _is_clf = "Classifier" in st.model_type_name or st.model_type_name in ("ClassifierChain", "_ChainEnsemble")
            _validate_target_values(st.train_target, "train", is_classification=_is_clf)
            if st.val_target is not None:
                _validate_target_values(st.val_target, "val", is_classification=_is_clf)

            # XGB cat-category alignment (no-op for non-XGB models): align the ``categories`` list across train / val / test so val/test rows whose category wasn't seen in train don't trip XGBoost's ``Found a category not in the training set`` rejection at predict time. Done AFTER pre_pipeline so the alignment uses the actual cat layout the model.fit + model.predict will see (pre_pipeline can rename / re-cast cat columns; aligning before that would be undone).
            train_df, st.val_df, st.test_df = _align_xgb_cat_categories(
                st.model_type_name,
                train_df,
                val_df=st.val_df,
                test_df=st.test_df,
            )

            if not st.just_evaluate:
                # Nest Lightning checkpoints + CSV logger output under the per-model directory (``{dirname(model_file_name)}/{basename_no_ext}/``) so different (target, model, schema_hash) combos don't collide in a shared project-root ``logs/`` folder. Only applies to TTR-wrapped Lightning regressors; tree models ignore this attribute. Set on the inner regressor (under TTR's ``.regressor``) when present, falling back to the model itself for direct Lightning regressors.
                _train_and_evaluate_nest_lightning_checkpoints_csv(st.model_file_name, model)
                model, st.best_iter = _train_model_with_fallback(
                    model=model,
                    model_obj=st.model_obj,
                    model_type_name=st.model_type_name,
                    train_df=train_df,
                    train_target=st.train_target,
                    fit_params=st.fit_params,
                    verbose=bool(st.verbose),
                )

                # Handle failed model training (e.g., dtype incompatibility)
                if model is None:
                    logger.warning("Model %s training failed - skipping evaluation", st.model_type_name)
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
                            pre_pipeline=st.pre_pipeline,
                            train_od_idx=train_od_idx,
                            val_od_idx=val_od_idx,
                            trainset_features_stats=trainset_features_stats,
                        ),
                        None,
                        None,
                        None,
                    )

            model_name = _update_model_name_after_training(model_name, len(train_df), st.train_details, st.best_iter)

            # K-fold OOF predictions for level-1 stacking. The in-sample ``train_preds`` (computed by ``_compute_split_metrics``
            # below for the "train" split) leak: every row was seen by the model during fit, so a meta-learner trained on those
            # predictions learns the residual structure of the in-sample fit, not the generalisation behaviour. OOF preds
            # produced by holding each row out via K-fold CV are the canonical replacement. Attached to the model object so
            # ``score_ensemble`` can pick them up at level-1 aggregation time without changing the public return signature.
            _train_and_evaluate_score_ensemble_can_pick(oof_n_splits, st.just_evaluate, st.model_type_name, st.train_target, model, train_df, oof_random_seed, st.group_ids, st.train_idx, oof_has_time, st._pre_pipeline_sample_weight, st.timestamps, st.fit_params)

    st.metrics_out = {"train": {}, "val": {}, "test": {}, "best_iter": st.best_iter}

    st._render_mark = render_queued_mark()
    _train_and_evaluate_m_step2_st_compute_trainset(st, reporting, model, model_name, data, train_df, confidence)

    if (st.compute_trainset_metrics or st.compute_valset_metrics or st.compute_testset_metrics) and st.verbose:
        logger.info("  Metrics computation done -- %.1fs", timer() - st.t0_metrics)
        log_render_queued(st._render_mark)

    _maybe_clean_ram()

    st._calib_probs_out, st._calib_target_out, st._calib_preds_out, st._oof_preds_out, st._oof_probs_out, st._oof_target_out = compute_calib_and_oof_outputs(
        model=model,
        calib_df=st.calib_df,
        calib_target=st.calib_target,
        real_drop_columns=st.real_drop_columns,
        pre_pipeline=st.pre_pipeline,
        skip_pre_pipeline_transform=st.skip_pre_pipeline_transform,
        skip_preprocessing=st.skip_preprocessing,
        fit_params=st.fit_params,
        model_type_name=st.model_type_name,
        model_name=model_name,
        row_wise_extensions_config=control.row_wise_extensions_config,
        calib_df_pre_pipeline=data.calib_df_pre_pipeline,
        test_df=data.test_df,
    )

    return (
        SimpleNamespace(
            model=model,
            test_preds=st.test_preds,
            test_probs=st.test_probs,
            test_target=st.test_target,
            val_preds=st.val_preds,
            val_probs=st.val_probs,
            val_target=st.val_target,
            train_preds=st.train_preds,
            train_probs=st.train_probs,
            train_target=st.train_target,
            oof_preds=st._oof_preds_out,
            oof_probs=st._oof_probs_out,
            oof_target=st._oof_target_out,
            calib_probs=st._calib_probs_out,
            calib_target=st._calib_target_out,
            calib_preds=st._calib_preds_out,
            metrics=st.metrics_out,
            columns=st.columns,
            pre_pipeline=st.pre_pipeline,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            trainset_features_stats=trainset_features_stats,
            plot_file=st.plot_file or "",  # this model's chart prefix; charts rendered for it after the fit (composite y-scale) reuse it
            # The chart title and split date details of THIS model, so charts rendered for it later (composite y-scale)
            # carry the same header as its own charts instead of a hand-built shorter one.
            chart_model_name=model_name,
            chart_split_details={"val": st.val_details or "", "test": st.test_details or ""},
        ),
        st._orig_train_df,
        st._orig_val_df,
        st._orig_test_df,
    )


def _train_and_evaluate_m_step1_live_trainperf_report(st, control):
    """Step 1 of train_and_evaluate_model: lines starting at ``if st.model_category == "cb" or st.callback_params:``."""
    if st.model_category == "cb" or st.callback_params:
        _live_plot = st._mono_beh.get("live_trainperf_plot", True) if st._mono_beh is not None else getattr(control, "live_trainperf_plot", True)
        _live_report = st._mono_beh.get("live_trainperf_report", False) if st._mono_beh is not None else getattr(control, "live_trainperf_report", False)
        st.callback_params = dict(st.callback_params or {})
        st.callback_params.setdefault("progress_widget", bool(_live_plot))
        st.callback_params.setdefault("report_progress_to_log", bool(_live_report))


def _train_and_evaluate_m_step2_st_compute_trainset(st, reporting, model, model_name, data, train_df, confidence):
    """Step 2 of train_and_evaluate_model: lines starting at ``if st.compute_trainset_metrics or st.compute_valset_metrics or st.comp``."""
    if st.compute_trainset_metrics or st.compute_valset_metrics or st.compute_testset_metrics:
        st.t0_metrics = timer()
        if st.verbose:
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
        _y_train_envelope_stats = _train_envelope_stats(st.train_target, reporting.mase_seasonality)

        common_metrics_params = dict(
            # ReportingConfig is forwarded so report_regression_model_perf
            # can read overrides like regression_title_metrics_tokens; before
            # this wiring the function referenced ``reporting_config`` as a
            # free variable and the custom config was silently ignored
            # (the try/except just fell back to defaults via NameError).
            reporting_config=reporting,
            model=model,
            model_type_name=st.model_type_name,
            model_name=model_name,
            group_ids=st.group_ids,
            target_label_encoder=st.target_label_encoder,
            figsize=st.figsize,
            nbins=st.nbins,
            print_report=st.print_report,
            plot_file=st.plot_file,
            show_perf_chart=st.show_perf_chart,
            show_fi=st.show_fi,
            fi_kwargs=st.fi_kwargs,
            subgroups=st.subgroups,
            custom_ice_metric=st.custom_ice_metric,
            custom_rice_metric=st.custom_rice_metric,
            n_features=st.n_features,
            show_prob_histogram=st.show_prob_histogram,
            prob_histogram_yscale=st.prob_histogram_yscale,
            show_inline_population_labels=st.show_inline_population_labels,
            title_metrics_tokens=st.title_metrics_tokens,
            plot_outputs=st.plot_outputs,
            plot_dpi=st.plot_dpi,
            binary_panels=st.binary_panels,
            multiclass_panels=st.multiclass_panels,
            multilabel_panels=st.multilabel_panels,
            ltr_panels=st.ltr_panels,
            quantile_panels=st.quantile_panels,
            quantile_alphas=st.quantile_alphas,
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
            split_timestamps=st.timestamps,
        )

        has_val = (st.val_idx is not None and len(st.val_idx) > 0) or st.val_df is not None
        has_test = (st.test_idx is not None and len(st.test_idx) > 0) or st.test_df is not None

        splits_config = [
            (
                "train",
                train_df,
                st.train_target,
                st.train_idx,
                st.train_preds,
                st.train_probs,
                st.train_details,
                st.compute_trainset_metrics and (st.train_idx is not None or train_df is not None),
            ),
            (
                "val",
                st.val_df,
                st.val_target,
                st.val_idx,
                st.val_preds,
                st.val_probs,
                st.val_details,
                st.compute_valset_metrics and ((st.val_idx is not None and len(st.val_idx) > 0) or st.val_df is not None),
            ),
        ]

        # Train runs sequentially (may feed into val/test setup); val+test parallelize later.
        st.columns, st.train_preds, st.train_probs = _train_and_evaluate_train_runs_sequentially_may(splits_config, st.metrics_out, has_val, has_test, common_metrics_params, st.columns, st.train_preds, st.train_probs)

        _val_cfg = next((c for c in splits_config if c[0] == "val" and c[-1]), None)
        _run_test = st.compute_testset_metrics and ((st.test_idx is not None and len(st.test_idx) > 0) or st.test_df is not None)

        _train_and_evaluate_run_test_df_none(_run_test, st.df, st.test_df, train_df)

        if _run_test:
            st.test_df, st.test_target, st.columns = _prepare_test_split(
                df=st.df,
                test_df=st.test_df,
                test_idx=st.test_idx,
                test_target=st.test_target,
                target=st.target,
                real_drop_columns=st.real_drop_columns,
                model=model,
                pre_pipeline=st.pre_pipeline,
                skip_pre_pipeline_transform=st.skip_pre_pipeline_transform,
                skip_preprocessing=st.skip_preprocessing,
                selector_passthrough_cols=(list(st.fit_params.get("text_features") or []) + list(st.fit_params.get("embedding_features") or [])) or None,
            )
            # Same engineered-name sanitization as the train/val frames above:
            # the test frame is transformed here by the same fitted pipeline, so
            # the pure map reproduces the identical label rename and predict
            # matches the model's fitted feature names.
            st.test_df = _sanitize_frame_columns(st.test_df)
            if st.test_df is not None:
                st._orig_test_df = st.test_df

        # Parallelize val and test metric computation -- numba kernels release GIL,
        # Agg matplotlib is thread-safe. Pure-Python parts still block, but the
        # heavy cumtime (binning, AUC, calibration plot save) runs concurrently.
        # Concurrent ThreadPoolExecutor was tried but matplotlib figure creation from concurrent threads races on pyplot's shared state even with Agg backend, producing "Argument must be an image or collection" errors in calibration plots. Sequential path is correct.
        with phase("compute_split_metrics", split="val"):
            val_res = _run_val_split_metrics(_val_cfg, st.metrics_out, has_test, common_metrics_params)
        with phase("compute_split_metrics", split="test"):
            test_res = _run_test_split_metrics(
                _run_test, st.metrics_out, st.test_df, st.test_target, st.test_idx,
                st.test_preds, st.test_probs, st.test_details, common_metrics_params,
            )

        if val_res is not None:
            st.val_preds, st.val_probs, st.columns = val_res
        if test_res is not None:
            st.test_preds, st.test_probs, st.columns = test_res

        # Same test_idx-slicing convention as `_pre_pipeline_sample_weight` above (train_idx-sliced),
        # so the confidence-analysis diagnostic reflects the same weighted objective the real model
        # was trained on instead of silently reverting to unweighted.
        _test_sample_weight = None
        _test_sample_weight = _train_and_evaluate_was_trained_instead_silently(st.sample_weight, st.test_idx, _test_sample_weight)

        maybe_run_confidence_analysis(
            run_test=_run_test,
            confidence=confidence,
            test_df=st.test_df,
            test_target=st.test_target,
            test_probs=st.test_probs,
            fit_params=st.fit_params,
            model_type_name=st.model_type_name,
            figsize=st.figsize,
            verbose=st.verbose,
            sample_weight=_test_sample_weight,
        )
