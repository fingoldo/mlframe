"""Core training functions for mlframe."""

from __future__ import annotations


import logging

logger = logging.getLogger(__name__)

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl
from pyutilz.strings import slugify
from pyutilz.system import tqdmu_lazy_start

from ..configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    ConfidenceAnalysisConfig,
    DummyBaselinesConfig,
    FeatureSelectionConfig,
    FeatureTypesConfig,
    LearningToRankConfig,
    LinearModelConfig,
    ModelHyperparamsConfig,
    MultilabelDispatchConfig,
    OutlierDetectionConfig,
    OutputConfig,
    PreprocessingBackendConfig,
    PreprocessingConfig,
    PreprocessingExtensionsConfig,
    QuantileRegressionConfig,
    ReportingConfig,
    TargetTypes,
    TrainingBehaviorConfig,
    TrainingSplitConfig,
)
from ..extractors import FeaturesAndTargetsExtractor
from ..helpers import (
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
)
from ..phases import phase, reset_phase_registry
from ..utils import (
    log_phase,
    log_ram_usage,
)

from .utils import (
    _ensure_logging_visible,
    _finalize_and_save_metadata,
    _get_pipeline_components,
    _initialize_training_defaults,
    _log_cardinality_and_drift_snapshot,
    _maybe_dispatch_to_ltr_ranker_suite,
    _phase_auto_detect_feature_types,
    _phase_fit_pipeline,
    _phase_global_outlier_detection,
    _phase_load_and_preprocess,
    _phase_pandas_conversion_and_cat_prep,
    _phase_train_val_test_split,
)
from ._training_context import TrainingContext
# CODE-P1-8: single consolidated import for all per-phase entry points (was 8 separate ``from
# ._phase_X import Y`` lines). Call e.g. ``pr.apply_polars_categorical_fixes(...)``.
from . import _phase_runners as pr


from ._misc_helpers import _split_preds_probs, _prep_polars_df  # noqa: F401


def train_mlframe_models_suite(
    df: Union[pl.DataFrame, pd.DataFrame, str],
    target_name: str,
    model_name: str,
    features_and_targets_extractor: FeaturesAndTargetsExtractor,
    # Model selection (top-level kwargs - these answer "what does this suite do")
    mlframe_models: Optional[List[str]] = None,
    recurrent_models: Optional[List[str]] = None,
    recurrent_config: Optional[Any] = None,
    sequences: Optional[List[np.ndarray]] = None,
    use_ordinary_models: bool = True,
    use_mlframe_ensembles: bool = True,
    # None = auto-detected from FTE.build_targets; LEARNING_TO_RANK routes to the ranker suite.
    target_type: Optional["TargetTypes"] = None,
    ranking_config: Optional["LearningToRankConfig"] = None,
    preprocessing_config: Optional[Union[PreprocessingConfig, Dict]] = None,
    split_config: Optional[Union[TrainingSplitConfig, Dict]] = None,
    pipeline_config: Optional[Union[PreprocessingBackendConfig, Dict]] = None,
    preprocessing_extensions: Optional[Union["PreprocessingExtensionsConfig", Dict]] = None,
    feature_types_config: Optional[Union[FeatureTypesConfig, Dict]] = None,
    linear_model_config: Optional[LinearModelConfig] = None,
    hyperparams_config: Optional[Union[ModelHyperparamsConfig, Dict]] = None,
    behavior_config: Optional[Union[TrainingBehaviorConfig, Dict]] = None,
    multilabel_dispatch_config: Optional["MultilabelDispatchConfig"] = None,
    reporting_config: Optional[Union["ReportingConfig", Dict]] = None,
    output_config: Optional[Union["OutputConfig", Dict]] = None,
    outlier_detection_config: Optional[Union["OutlierDetectionConfig", Dict]] = None,
    feature_selection_config: Optional[Union[FeatureSelectionConfig, Dict]] = None,
    confidence_analysis_config: Optional[Union["ConfidenceAnalysisConfig", Dict]] = None,
    baseline_diagnostics_config: Optional[Union["BaselineDiagnosticsConfig", Dict]] = None,
    dummy_baselines_config: Optional[Union["DummyBaselinesConfig", Dict]] = None,
    quantile_regression_config: Optional[Union["QuantileRegressionConfig", Dict]] = None,
    # MLFRAME_DISABLE_COMPOSITE=1 env var forces OFF regardless of config (kill switch).
    composite_target_discovery_config: Optional[Union["CompositeTargetDiscoveryConfig", Dict]] = None,
    feature_handling_config: Optional[Any] = None,
    verbose: int = 1,
) -> Tuple[Dict, Dict]:
    """
    Train a suite of ML models on a dataset.

    Args:
        df: DataFrame or path to parquet file
        target_name: Name of the target to predict
        model_name: Base name for the models
        features_and_targets_extractor: FeaturesAndTargetsExtractor instance for computing targets

        mlframe_models: List of model types to train (cb, lgb, xgb, mlp, hgb, linear, ridge, etc.)
        recurrent_models: List of recurrent model types to train (lstm, gru, rnn, transformer).
            These models handle sequential data and support variable-length sequences.
        recurrent_config: RecurrentConfig object for recurrent model hyperparameters.
            If None, uses default configuration.
        sequences: Pre-extracted sequences as list of (seq_len, n_features) arrays.
            If None and extractor has sequence_columns configured, sequences will be
            extracted automatically using extractor.get_sequences().
        use_ordinary_models: Whether to train regular models
        use_mlframe_ensembles: Whether to create ensembles

        preprocessing_config: Preprocessing configuration. Holds custom transformer overrides
            (``scaler``, ``imputer``, ``category_encoder``).
        split_config: Train/val/test split configuration
        pipeline_config: Pipeline configuration

        feature_selection_config: Holds ``use_mrmr_fs``, ``mrmr_kwargs``, ``rfecv_models``,
            ``rfecv_kwargs``, ``custom_pre_pipelines``.

        hyperparams_config: Model hyperparameters (iterations, learning rate, per-model kwargs).
        behavior_config: Training behavior flags (GPU preference, calibration, fairness).

        reporting_config: Calibration / training-report look. Holds figure size, chart toggles,
            title-metrics template, histogram subplot toggles, inline population labels, FI plot config.
        output_config: Filesystem destinations - ``data_dir``, ``models_dir``, ``plot_file``, ``save_charts``.
        outlier_detection_config: Outlier-detector + ``apply_to_val``.

        verbose: Verbosity level (0=silent, 1=info, 2=debug)

    Returns:
        Tuple of (models_dict, metadata_dict)

    Example:
        ```python
        models, metadata = train_mlframe_models_suite(
            df="data.parquet",
            target_name="target",
            model_name="experiment_1",
            features_and_targets_extractor=my_ft_extractor,
            mlframe_models=["linear", "ridge", "cb", "lgb"],
            preprocessing_config=PreprocessingConfig(fillna_value=0.0),
            split_config=TrainingSplitConfig(test_size=0.1, val_size=0.1),
            reporting_config=ReportingConfig(
                title_metrics_template="ICE BR_DECOMP ECE CMAEW",
                show_prob_histogram=True,
            ),
            output_config=OutputConfig(data_dir="./artifacts", save_charts=True),
        )
        ```
    """

    if verbose:
        _ensure_logging_visible()

    # Module-global registry; not safe to invoke concurrent training suites from the same process.
    reset_phase_registry()

    if not isinstance(df, (pd.DataFrame, pl.DataFrame, str)):
        raise TypeError(f"df must be pandas DataFrame, polars DataFrame, or path string, " f"got {type(df).__name__}")
    if isinstance(df, str) and not df.lower().endswith(".parquet"):
        raise ValueError(f"File path must be a .parquet file, got: {df}")

    if not target_name:
        raise ValueError("target_name cannot be empty")
    if not model_name:
        raise ValueError("model_name cannot be empty")
    if features_and_targets_extractor is None:
        raise ValueError("features_and_targets_extractor is required")

    ctx = pr.setup_configuration(
        preprocessing_config=preprocessing_config,
        pipeline_config=pipeline_config,
        feature_types_config=feature_types_config,
        split_config=split_config,
        hyperparams_config=hyperparams_config,
        behavior_config=behavior_config,
        reporting_config=reporting_config,
        output_config=output_config,
        outlier_detection_config=outlier_detection_config,
        feature_selection_config=feature_selection_config,
        confidence_analysis_config=confidence_analysis_config,
        baseline_diagnostics_config=baseline_diagnostics_config,
        dummy_baselines_config=dummy_baselines_config,
        quantile_regression_config=quantile_regression_config,
        composite_target_discovery_config=composite_target_discovery_config,
        feature_handling_config=feature_handling_config,
        ranking_config=ranking_config,
        model_name=model_name,
        target_name=target_name,
        mlframe_models=mlframe_models,
        use_mlframe_ensembles=use_mlframe_ensembles,
        use_ordinary_models=use_ordinary_models,
        verbose=verbose,
    )

    # LTR opt-in: helper returns None for non-LTR call sites. Moved AFTER setup_configuration because the helper now reads
    # Pydantic-shape configs from ``ctx`` (output / split / hyperparams / reporting); pre-setup the configs may still be
    # dict-or-None form and would trip ``_cfg_get`` callers downstream.
    _ltr_result = _maybe_dispatch_to_ltr_ranker_suite(
        ctx,
        target_type=target_type,
        df=df,
        features_and_targets_extractor=features_and_targets_extractor,
    )
    if _ltr_result is not None:
        return _ltr_result

    preprocessing_config = ctx.preprocessing_config
    pipeline_config = ctx.pipeline_config
    feature_types_config = ctx.feature_types_config
    split_config = ctx.split_config
    hyperparams_config = ctx.hyperparams_config
    behavior_config = ctx.behavior_config
    reporting_config = ctx.reporting_config
    output_config = ctx.output_config
    outlier_detection_config = ctx.outlier_detection_config
    feature_selection_config = ctx.feature_selection_config
    baseline_diagnostics_config = ctx.baseline_diagnostics_config
    dummy_baselines_config = ctx.dummy_baselines_config
    quantile_regression_config = ctx.quantile_regression_config
    composite_target_discovery_config = ctx.composite_target_discovery_config
    data_dir = ctx.data_dir
    models_dir = ctx.models_dir
    save_charts = ctx.save_charts
    outlier_detector = ctx.outlier_detector
    od_val_set = ctx.od_val_set
    use_mrmr_fs = ctx.use_mrmr_fs
    mrmr_kwargs = ctx.mrmr_kwargs
    rfecv_models = ctx.rfecv_models
    custom_pre_pipelines = ctx.custom_pre_pipelines
    common_params_dict = ctx.common_params_dict
    mlframe_models = ctx.mlframe_models
    metadata = ctx.metadata

    # ctx-form: parallel-session migrated _phase_load_and_preprocess to read from / write to ctx in place.
    ctx.df = df
    ctx.recurrent_models = recurrent_models
    ctx.sequences = sequences
    _phase_load_and_preprocess(ctx, features_and_targets_extractor=features_and_targets_extractor)
    df = ctx.df
    target_by_type = ctx.target_by_type
    # Empty target_by_type means the extractor returned no targets - usually a
    # caller-side mis-configuration (e.g. SimpleFeaturesAndTargetsExtractor with
    # classification_exact_values set but classification_targets omitted, since
    # build_targets gates the exact-values branch on classification_targets
    # being truthy). Pre-fix this short-circuited silently to (empty_models, metadata)
    # making such misconfigurations look like a fast successful run; loud WARN
    # surfaces them at suite entry instead.
    if not target_by_type:
        logger.warning(
            "train_mlframe_models_suite: features_and_targets_extractor produced an "
            "empty target_by_type. No models will be trained. Check the extractor's "
            "configuration - common cause is passing classification_exact_values / "
            "classification_thresholds without classification_targets=[...] (the "
            "default SimpleFeaturesAndTargetsExtractor.build_targets gates those "
            "branches on classification_targets being truthy)."
        )
    group_ids_raw = ctx.group_ids_raw
    group_ids = ctx.group_ids
    timestamps = ctx.timestamps
    artifacts = ctx.artifacts
    additional_columns_to_drop = ctx.additional_columns_to_drop
    sample_weights = ctx.sample_weights
    baseline_rss_mb = ctx.baseline_rss_mb
    df_size_mb = ctx.df_size_mb
    sequences = ctx.sequences

    (
        train_idx, val_idx, test_idx,
        train_details, val_details, test_details,
        train_df, val_df, test_df,
        fairness_subgroups, fairness_features,
        train_sequences, val_sequences, test_sequences,
        baseline_rss_mb,
    ) = _phase_train_val_test_split(
        df=df,
        target_by_type=target_by_type,
        timestamps=timestamps,
        group_ids=group_ids,
        group_ids_raw=group_ids_raw,
        artifacts=artifacts,
        sequences=sequences,
        split_config=split_config,
        behavior_config=behavior_config,
        metadata=metadata,
        data_dir=data_dir,
        models_dir=models_dir,
        target_name=target_name,
        model_name=model_name,
        df_size_mb=df_size_mb,
        verbose=verbose,
    )
    # ``del df`` drops the local rebound name so the only remaining strong reference
    # is ``ctx.df``; nulling that lets the GC reclaim the now-unreferenced source frame.
    # Without both, the post-split full dataframe lingers in memory until the suite ends.
    del df
    ctx.df = None
    # Mirror locals into ctx in a single bulk loop. This is the in-progress migration from
    # the legacy "phase returns big tuple, caller fans out into locals" form to a pure
    # ctx-form where phases write straight to ctx. Until every phase is converted the
    # bulk-copy keeps the existing return-tuple shape working while ctx-form readers
    # downstream see the same values.
    for _k in ("train_idx", "val_idx", "test_idx", "train_details", "val_details",
               "test_details", "train_df", "val_df", "test_df", "fairness_subgroups",
               "fairness_features", "train_sequences", "val_sequences", "test_sequences",
               "baseline_rss_mb"):
        setattr(ctx, _k, locals()[_k])

    (
        train_df, val_df, test_df,
        pipeline, extensions_pipeline,
        cat_features, cat_features_polars,
        was_polars_input, all_models_polars_native, polars_pipeline_applied,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        pipeline_config, preprocessing_extensions,
        train_df_pandas_pre,
    ) = _phase_fit_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        mlframe_models=mlframe_models,
        pipeline_config=pipeline_config,
        preprocessing_config=preprocessing_config,
        feature_types_config=feature_types_config,
        preprocessing_extensions=preprocessing_extensions,
        metadata=metadata,
        verbose=verbose,
        # Threaded through so apply_preprocessing_extensions can grab a 1-D
        # regression target for PySR symbolic feature discovery (used to be
        # silently None -> PySR-skip wiring bug). train_idx slices the
        # PRE-split target down to train-set rows so PySR sees train-only y.
        target_by_type=target_by_type,
        train_idx=ctx.train_idx,
    )
    for _k in ("train_df", "val_df", "test_df", "pipeline", "extensions_pipeline",
               "cat_features", "cat_features_polars", "was_polars_input",
               "all_models_polars_native", "polars_pipeline_applied",
               "train_df_polars_pre", "val_df_polars_pre", "test_df_polars_pre",
               "pipeline_config", "preprocessing_extensions",
               "train_df_pandas_pre"):
        setattr(ctx, _k, locals()[_k])

    (
        train_df, val_df, test_df,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        text_features, embedding_features, cat_features,
        text_emb_set, _dropped_high_card_data,
    ) = _phase_auto_detect_feature_types(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_df_polars_pre=train_df_polars_pre,
        val_df_polars_pre=val_df_polars_pre,
        test_df_polars_pre=test_df_polars_pre,
        cat_features=cat_features,
        cat_features_polars=cat_features_polars,
        was_polars_input=was_polars_input,
        all_models_polars_native=all_models_polars_native,
        pipeline_config=pipeline_config,
        feature_types_config=feature_types_config,
        metadata=metadata,
        verbose=verbose,
        train_df_pandas_pre=train_df_pandas_pre,
    )

    if verbose:
        _log_cardinality_and_drift_snapshot(
            train_df=train_df, val_df=val_df, test_df=test_df,
            cat_features=cat_features,
            text_features=text_features,
            embedding_features=embedding_features,
        )

    metadata["text_features"] = text_features
    metadata["embedding_features"] = embedding_features
    for _k in ("train_df", "val_df", "test_df", "train_df_polars_pre", "val_df_polars_pre",
               "test_df_polars_pre", "text_features", "embedding_features", "cat_features",
               "text_emb_set", "_dropped_high_card_data"):
        setattr(ctx, _k, locals()[_k])

    if verbose:
        log_phase("PHASE 4: Model Training")

    with phase("initialize_training_defaults"):
        (
            common_params_dict,
            rfecv_models,
            mrmr_kwargs,
        ) = _initialize_training_defaults(
            common_params_dict=common_params_dict,
            rfecv_models=rfecv_models,
            mrmr_kwargs=mrmr_kwargs,
        )

    # Propagate split-config random_seed so the default CatBoostEncoder is
    # deterministic across runs. fix audit row FE-P2-5.
    _seed_for_components = getattr(split_config, "random_seed", None) if split_config is not None else None
    category_encoder, imputer, scaler = _get_pipeline_components(
        preprocessing_config, cat_features, random_seed=_seed_for_components,
    )
    # Propagate to ctx so _phase_train_one_target reads the resolved components, not the None defaults from TrainingContext (LinearModelStrategy.build_pipeline silently skips imputation when imputer=None, sending raw NaN into LinearRegression.fit).
    ctx.category_encoder = category_encoder
    ctx.imputer = imputer
    ctx.scaler = scaler

    # CACHE-P2-1: MUST run BEFORE _phase_pandas_conversion_and_cat_prep. The
    # polars vs pandas branch below picks its backend from ``train_df``'s
    # current type; if the pandas conversion fires first the polars fastpath
    # silently degrades to pandas without surfacing the regression. Keep
    # this block ABOVE the ``_phase_pandas_conversion_and_cat_prep`` call.
    #
    # train_df is still polars at this point IFF the upstream split kept the polars fastpath alive
    # (no pandas-only preprocessor forced a conversion). The polars stats path lazily expresses the
    # numeric/categorical summaries without materialising a pandas copy, so we must branch here
    # rather than always falling through to the pandas-typed get_trainset_features_stats below.
    if isinstance(train_df, pl.DataFrame):
        if verbose:
            logger.info("Computing trainset_features_stats on Polars...")
        with phase("trainset_features_stats", backend="polars"):
            trainset_features_stats = get_trainset_features_stats_polars(train_df)
    else:
        if verbose:
            logger.info("Computing trainset_features_stats on pandas...")
        with phase("trainset_features_stats", backend="pandas"):
            trainset_features_stats = get_trainset_features_stats(train_df)

    (
        train_df_pd, val_df_pd, test_df_pd,
        train_df_polars, val_df_polars, test_df_polars,
        train_df, val_df, test_df,
        train_df_size_bytes_cached, val_df_size_bytes_cached,
        defer_pandas_conv, baseline_rss_mb,
    ) = _phase_pandas_conversion_and_cat_prep(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_df_polars_pre=train_df_polars_pre,
        val_df_polars_pre=val_df_polars_pre,
        test_df_polars_pre=test_df_polars_pre,
        cat_features=cat_features,
        was_polars_input=was_polars_input,
        all_models_polars_native=all_models_polars_native,
        needs_polars_pre_clone=(
            was_polars_input
            and not pipeline_config.skip_categorical_encoding
            and pipeline_config.categorical_encoding is not None
        ),
        mlframe_models=mlframe_models,
        recurrent_models=recurrent_models,
        rfecv_models=rfecv_models,
        baseline_rss_mb=baseline_rss_mb,
        df_size_mb=df_size_mb,
        verbose=verbose,
    )
    # Store cached sizes on ctx BEFORE the per-target loop so _train_one_target can read them.
    ctx.train_df_size_bytes_cached = train_df_size_bytes_cached
    ctx.val_df_size_bytes_cached = val_df_size_bytes_cached

    # ctx-form: parallel-session migrated _phase_global_outlier_detection to read from / write to ctx in place.
    ctx.train_df_pd = train_df_pd
    ctx.val_df_pd = val_df_pd
    ctx.train_df_polars = train_df_polars
    ctx.val_df_polars = val_df_polars
    ctx.train_idx = train_idx
    ctx.val_idx = val_idx
    ctx.target_by_type = target_by_type
    ctx.outlier_detector = outlier_detector
    ctx.od_val_set = od_val_set
    ctx.baseline_rss_mb = baseline_rss_mb
    ctx.df_size_mb = df_size_mb
    ctx.metadata = metadata
    _phase_global_outlier_detection(ctx)
    filtered_train_df = ctx.filtered_train_df
    filtered_val_df = ctx.filtered_val_df
    filtered_train_idx = ctx.filtered_train_idx
    filtered_val_idx = ctx.filtered_val_idx
    train_od_idx = ctx.train_od_idx
    val_od_idx = ctx.val_od_idx
    outlier_detection_result = ctx.outlier_detection_result
    train_df_polars = ctx.train_df_polars
    val_df_polars = ctx.val_df_polars

    ctx.filtered_train_df = filtered_train_df
    ctx.filtered_val_df = filtered_val_df
    ctx.filtered_train_idx = filtered_train_idx
    ctx.filtered_val_idx = filtered_val_idx
    ctx.train_od_idx = train_od_idx
    ctx.val_od_idx = val_od_idx

    # Discovery cache lives under ``data_dir/.discovery_cache`` when data_dir is set; a value of None disables caching (back-compat for callers without an output_config).
    _discovery_cache_dir = None
    try:
        if data_dir:
            from pathlib import Path as _P
            _discovery_cache_dir = str(_P(data_dir) / ".discovery_cache")
    except Exception:
        _discovery_cache_dir = None
    target_by_type, metadata = pr.run_composite_target_discovery(
        composite_target_discovery_config=composite_target_discovery_config,
        target_by_type=target_by_type,
        mlframe_models=mlframe_models,
        metadata=metadata,
        filtered_train_df=filtered_train_df,
        filtered_train_idx=filtered_train_idx,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        test_df_pd=test_df_pd,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        baseline_diagnostics_config=baseline_diagnostics_config,
        cat_features=cat_features,
        verbose=verbose,
        discovery_cache_dir=_discovery_cache_dir,
    )

    (
        train_df_polars, val_df_polars, test_df_polars,
        train_df_pd, val_df_pd, test_df_pd,
        filtered_train_df, filtered_val_df,
    ) = pr.apply_polars_categorical_fixes(
        train_df_polars=train_df_polars,
        val_df_polars=val_df_polars,
        test_df_polars=test_df_polars,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        test_df_pd=test_df_pd,
        filtered_train_df=filtered_train_df,
        filtered_val_df=filtered_val_df,
        cat_features=cat_features,
        align_polars_categorical_dicts=behavior_config.align_polars_categorical_dicts,
        defer_pandas_conv=defer_pandas_conv,
        was_polars_input=was_polars_input,
        verbose=bool(verbose),
    )
    # 2026-05-16 wiring fix: write the filled frames BACK to ctx.
    # _train_one_target later does ``train_df_polars = ctx.train_df_polars``
    # (main.py:563+); without this back-write it would read the pre-fix
    # frames with nulls still in cat columns, causing CB Arrow Pool to
    # crash with 'Data with nulls is not supported for categorical
    # columns'. Surfaced 2026-05-16 by
    # test_sensor_polars_utf8_nullable_cat_fills_before_cb +
    # test_sensor_enum_null_fill_reaches_lazy_pandas_conversion.
    for _k in ("train_df_polars", "val_df_polars", "test_df_polars",
               "train_df_pd", "val_df_pd", "test_df_pd",
               "filtered_train_df", "filtered_val_df"):
        setattr(ctx, _k, locals()[_k])

    # Save metadata early so partial training runs leave already-trained models usable.
    _finalize_and_save_metadata(ctx)

    # Maps slugified names back to originals for load_mlframe_suite.
    slug_to_original_target_type = {}
    slug_to_original_target_name = {}

    _all_target_audits = pr.run_temporal_audit_batch(
        behavior_config=behavior_config,
        features_and_targets_extractor=features_and_targets_extractor,
        timestamps=timestamps,
        target_by_type=target_by_type,
        verbose=bool(verbose),
    )

    for target_type, targets in tqdmu_lazy_start(target_by_type.items(), desc="target type"):
        slug_to_original_target_type[slugify(str(target_type).lower())] = target_type

        # !TODO ! optimize for creation of inner feature matrices of cb,lgb,xgb here. They should be created once per featureset, not once per target.
        for cur_target_name, cur_target_values in tqdmu_lazy_start(targets.items(), desc="target"):
            pr._train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values)

    models = ctx.models
    metadata = ctx.metadata
    _non_neural_train_times = ctx._non_neural_train_times
    train_df_polars = ctx.train_df_polars
    val_df_polars = ctx.val_df_polars
    test_df_polars = ctx.test_df_polars
    train_df_pd = ctx.train_df_pd
    val_df_pd = ctx.val_df_pd
    test_df_pd = ctx.test_df_pd
    filtered_train_df = ctx.filtered_train_df
    filtered_val_df = ctx.filtered_val_df
    pipeline = ctx.pipeline
    defer_pandas_conv = ctx.defer_pandas_conv
    baseline_rss_mb = ctx.baseline_rss_mb
    train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    trainset_features_stats = ctx.trainset_features_stats
    slug_to_original_target_type = ctx.slug_to_original_target_type
    slug_to_original_target_name = ctx.slug_to_original_target_name

    # CODE-P1-12: read recurrent_models from ctx (not the closed-over function param) so any
    # mid-flow mutation of ctx.recurrent_models propagates correctly to train_recurrent_models.
    models = pr.train_recurrent_models(
        models=models,
        recurrent_models=ctx.recurrent_models,
        recurrent_config=recurrent_config,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        test_sequences=test_sequences,
        train_df=train_df,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        target_by_type=target_by_type,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        _non_neural_train_times=_non_neural_train_times,
        model_name=model_name,
        verbose=bool(verbose),
    )
    ctx.models = models

    if verbose:
        log_phase(f"Training suite completed for {model_name}, {sum(len(v) for targets in models.values() for v in targets.values())} models.")
        log_ram_usage()

    metadata = pr.finalize_suite(ctx)

    models, metadata = pr.run_composite_post_processing(
        models=models,
        metadata=metadata,
        target_by_type=target_by_type,
        composite_target_discovery_config=composite_target_discovery_config,
        target_name=target_name,
        model_name=model_name,
        filtered_train_df=filtered_train_df,
        filtered_val_df=filtered_val_df,
        test_df_pd=test_df_pd,
        filtered_train_idx=filtered_train_idx,
        filtered_val_idx=filtered_val_idx,
        test_idx=test_idx,
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        train_idx=train_idx,
        val_idx=val_idx,
        dummy_baselines_config=dummy_baselines_config,
        reporting_config=reporting_config,
        plot_file=None,
        verbose=bool(verbose),
    )
    # ``_dropped_high_card_data`` is bound unconditionally above (post-auto-detect tuple
    # unpacking); the previous try/except guarded against a NameError that can no longer
    # happen along any control-flow path. Clearing frees per-column nan-imputation arrays.
    _dropped_high_card_data.clear()

    return dict(models), metadata


# Re-export predict / load entry points for back-compat.
from .predict import (  # noqa: E402,F401
    predict_mlframe_models_suite,
    predict_from_models,
    load_mlframe_suite,
)
