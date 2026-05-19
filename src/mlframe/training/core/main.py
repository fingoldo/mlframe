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
from pathlib import Path as _P  # PATHLIB-IMPORT-PER-CALL: hoist to module scope (was paid per suite call)
from ..extractors import FeaturesAndTargetsExtractor
from ..feature_handling.fingerprint import reset_session as reset_fh_session
from ..helpers import (
    TrainMlframeSuitePrecomputed,
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


from ._misc_helpers import _bulk_setattr_to_ctx, _split_preds_probs, _prep_polars_df  # noqa: F401


# Module-level handles for the prelude patch functions. They're populated
# lazily on the first ``train_mlframe_models_suite`` call (delays the
# joblib / lightgbm / catboost / xgboost touches that the patches do).
# Module-level binding lets tests monkeypatch them.
apply_loky_cpu_count_override = None
apply_third_party_patches_once = None


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
    precomputed: Optional["TrainMlframeSuitePrecomputed"] = None,
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

        precomputed: Opt-in ``TrainMlframeSuitePrecomputed`` bundle for repeated-suite-on-same-train
            benchmarking. When supplied, each non-None field skips the matching in-suite compute step:
            ``trainset_features_stats`` (skips the stats pass), ``dummy_baselines`` (skips the per-target
            dummy-baseline pass by short-circuiting via ``dummy_baselines_config.enabled=False`` and
            pre-seeding ``metadata["dummy_baselines"]``), ``composite_target_specs`` (skips the
            composite discovery phase and pre-seeds ``metadata["composite_target_specs"]``). Build the
            bundle via ``mlframe.training.helpers.precompute_all`` or by hand from a prior run's
            metadata. None (default) preserves legacy behaviour: every step computes inline.

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

    # Apply the third-party patches the suite relies on
    # (LGBMModel.feature_names_in_ setter; dataset-build logging) lazily
    # at suite entry. Previously these ran as import-time side effects
    # of ``mlframe.training``; now bare imports leave joblib / lightgbm
    # / catboost / xgboost untouched.
    # The two functions are looked up on this module so test code can
    # monkeypatch them in-place to observe call ordering.
    global apply_loky_cpu_count_override, apply_third_party_patches_once
    if apply_loky_cpu_count_override is None:
        from .. import apply_loky_cpu_count_override as _apply_loky
        apply_loky_cpu_count_override = _apply_loky
    if apply_third_party_patches_once is None:
        from .._model_factories import apply_third_party_patches_once as _apply_patches
        apply_third_party_patches_once = _apply_patches
    apply_loky_cpu_count_override()
    apply_third_party_patches_once()

    # Module-global registry; not safe to invoke concurrent training suites from the same process.
    reset_phase_registry()
    # Rotate the FH InMemoryKey session token alongside the phase registry. Without this, two
    # consecutive suite calls within the same process keep the prior SessionToken and any
    # ``id(train_df)`` reuse (Python may recycle ids after the first frame is GC'd) collides on a
    # cached entry whose underlying state belongs to the prior suite. The session reset guarantees
    # each suite starts from a fresh FH cache namespace.
    reset_fh_session()

    if not isinstance(df, (pd.DataFrame, pl.DataFrame, str)):
        raise TypeError(f"df must be pandas DataFrame, polars DataFrame, or path string, " f"got {type(df).__name__}")
    if isinstance(df, str) and not df.lower().endswith(".parquet"):
        raise ValueError(f"File path must be a .parquet file, got: {df}")

    if target_name is None or not isinstance(target_name, str):
        raise TypeError(f"target_name must be a non-empty string, got {type(target_name).__name__}")
    if not target_name.strip():
        raise ValueError("target_name cannot be empty or whitespace-only")
    if model_name is None or not isinstance(model_name, str):
        raise TypeError(f"model_name must be a non-empty string, got {type(model_name).__name__}")
    if not model_name.strip():
        raise ValueError("model_name cannot be empty or whitespace-only")
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
    _bulk_setattr_to_ctx(ctx, (
        "train_idx", "val_idx", "test_idx", "train_details", "val_details",
        "test_details", "train_df", "val_df", "test_df", "fairness_subgroups",
        "fairness_features", "train_sequences", "val_sequences", "test_sequences",
        "baseline_rss_mb",
    ), locals())

    (
        train_df, val_df, test_df,
        pipeline, extensions_pipeline,
        cat_features, cat_features_polars,
        was_polars_input, all_models_polars_native, polars_pipeline_applied,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        pipeline_config, preprocessing_extensions,
        train_df_pandas_pre_meta,
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
    _bulk_setattr_to_ctx(ctx, (
        "train_df", "val_df", "test_df", "pipeline", "extensions_pipeline",
        "cat_features", "cat_features_polars", "was_polars_input",
        "all_models_polars_native", "polars_pipeline_applied",
        "train_df_polars_pre", "val_df_polars_pre", "test_df_polars_pre",
        "pipeline_config", "preprocessing_extensions",
        "train_df_pandas_pre_meta",
    ), locals())

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
        train_df_pandas_pre_meta=train_df_pandas_pre_meta,
    )

    if verbose:
        _log_cardinality_and_drift_snapshot(
            train_df=train_df, val_df=val_df, test_df=test_df,
            cat_features=cat_features,
            text_features=text_features,
            embedding_features=embedding_features,
            ctx=ctx,
        )

    metadata["text_features"] = text_features
    metadata["embedding_features"] = embedding_features
    _bulk_setattr_to_ctx(ctx, (
        "train_df", "val_df", "test_df", "train_df_polars_pre", "val_df_polars_pre",
        "test_df_polars_pre", "text_features", "embedding_features", "cat_features",
        "text_emb_set", "_dropped_high_card_data",
    ), locals())

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
            suite_verbose=getattr(ctx, "verbose", None),
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
    # Opt-in fast path: when caller supplies a pre-computed stats dict via ``precomputed``, skip the
    # inline pass entirely. Useful for repeated suite runs on the same train frame (benchmarking).
    # PRECOMP-NO-FP-CHECK: when the caller stamped ``train_df_fingerprint`` on the bundle, verify it
    # matches the live train frame. A mismatch (caller passed a bundle from a different run) is a
    # silent label-leak vector -- we WARN-and-recompute rather than trust the precomputed stats.
    _precomp_fp_ok = True
    if (
        precomputed is not None
        and precomputed.train_df_fingerprint
        and train_df is not None
    ):
        try:
            from ..feature_handling.fingerprint import fingerprint_df as _fp
            _live_fp = _fp(train_df).short()
            _bundle_fp = str(precomputed.train_df_fingerprint)
            if _bundle_fp not in (_live_fp, _live_fp[:len(_bundle_fp)]):
                logger.warning(
                    "precomputed.train_df_fingerprint (%s) does not match live train_df fingerprint (%s); "
                    "ignoring the precomputed bundle and recomputing inline.",
                    _bundle_fp, _live_fp,
                )
                _precomp_fp_ok = False
        except Exception as _fp_err:
            logger.debug("precompute fingerprint cross-check skipped (%s)", _fp_err)
    # Truthy gate (not "is not None"): empty dicts/Series must NOT silently disable the inline compute -- the
    # ``precompute_*`` stubs historically returned ``{}`` and at-call callers occasionally pass through partial
    # bundles from disk.
    if _precomp_fp_ok and precomputed is not None and precomputed.trainset_features_stats:
        if verbose:
            logger.info("Using caller-supplied trainset_features_stats (precomputed bundle); skipping inline compute.")
        trainset_features_stats = precomputed.trainset_features_stats
    elif isinstance(train_df, pl.DataFrame):
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
        # _phase_fit_pipeline reports whether the pre-fit polars-stage actually ran (skipped when caller passed
        # pandas or when no model is polars-native). The pandas-conversion phase needs the truthful value, not the
        # default=True placeholder, to decide whether the polars-side cat fixes (Utf8 -> Categorical fills) need to be
        # mirrored back into the pandas-side frames before CatBoost Pool construction.
        polars_pipeline_applied=polars_pipeline_applied,
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
            _discovery_cache_dir = str(_P(data_dir) / ".discovery_cache")
    except Exception:
        _discovery_cache_dir = None
    # Opt-in fast path: caller-supplied composite_target_specs bypasses the discovery phase entirely.
    # Pre-seed metadata so downstream readers (per-target dummy-baseline inversion, predict-time
    # composite inverse transforms) find the specs in their usual location. target_by_type is left
    # unchanged because the caller-supplied specs imply the augmented targets already live in it.
    # Truthy gate (not "is not None"): an empty composite spec dict carries zero discovered targets, which is
    # indistinguishable from "discovery skipped"; if we let it through, the suite would silently lose every
    # composite target. The stub helpers used to return {} -- truthy gate is the defensive fix.
    if _precomp_fp_ok and precomputed is not None and precomputed.composite_target_specs:
        if verbose:
            logger.info("Using caller-supplied composite_target_specs (precomputed bundle); skipping discovery.")
        metadata["composite_target_specs"] = precomputed.composite_target_specs
    else:
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
    # Write the filled frames BACK to ctx. ``_train_one_target`` later
    # does ``train_df_polars = ctx.train_df_polars``; without this
    # back-write it would read the pre-fix frames with nulls still in
    # cat columns, causing CB Arrow Pool to crash with 'Data with nulls
    # is not supported for categorical columns'. Covered by
    # test_sensor_polars_utf8_nullable_cat_fills_before_cb +
    # test_sensor_enum_null_fill_reaches_lazy_pandas_conversion.
    _bulk_setattr_to_ctx(ctx, (
        "train_df_polars", "val_df_polars", "test_df_polars",
        "train_df_pd", "val_df_pd", "test_df_pd",
        "filtered_train_df", "filtered_val_df",
    ), locals())

    # Opt-in fast path: caller-supplied dummy_baselines bypasses the per-target dummy compute.
    # Pre-seed metadata so downstream summary / verdict consumers find the payload in its usual
    # location, then shallow-copy dummy_baselines_config with enabled=False so the per-target
    # ``run_dummy_baselines`` short-circuits (its first guard checks ``config.enabled``).
    # Truthy gate (not "is not None"): empty dummy_baselines would silently disable every per-target compute --
    # callers must supply real values; "I passed it but it's empty" must fall through to inline compute.
    if _precomp_fp_ok and precomputed is not None and precomputed.dummy_baselines:
        if verbose:
            logger.info("Using caller-supplied dummy_baselines (precomputed bundle); skipping per-target compute.")
        metadata["dummy_baselines"] = precomputed.dummy_baselines
        try:
            dummy_baselines_config = dummy_baselines_config.model_copy(update={"enabled": False})
        except AttributeError:
            # Defensive: when the config slot is plain dict / SimpleNamespace fall back to attribute set.
            try:
                dummy_baselines_config.enabled = False
            except Exception:
                pass
        ctx.dummy_baselines_config = dummy_baselines_config

    # Save metadata early so partial training runs leave already-trained models usable.
    _finalize_and_save_metadata(ctx)

    # Maps slugified names back to originals for load_mlframe_suite.
    slug_to_original_target_type = {}
    slug_to_original_target_name = {}

    ctx._all_target_audits = pr.run_temporal_audit_batch(
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

    # VOTENRANK-DISCONNECT (post-loop): collect any per-target ``_leaderboard`` payloads that
    # ``score_ensemble`` parked on the ensemble result dicts and surface them under a single
    # ``metadata["votenrank_leaderboard"]`` key. CSV export lands under
    # ``output_config.data_dir/.leaderboard.csv`` when data_dir is set. The wire is read-only on
    # ``ctx.ensembles`` so it can't reach back into the per-target loop and is safe to skip when
    # F2 has not emitted any leaderboard yet (forward-compat with older score_ensemble builds).
    try:
        _leaderboards = {}
        for _tt, _by_name in (ctx.ensembles or {}).items():
            for _tname, _ens_dict in (_by_name or {}).items():
                if not isinstance(_ens_dict, dict):
                    continue
                _lb = _ens_dict.get("_leaderboard")
                if _lb is None:
                    continue
                _leaderboards.setdefault(str(_tt), {})[_tname] = _lb
        if _leaderboards:
            ctx.metadata["votenrank_leaderboard"] = _leaderboards
            if data_dir:
                _csv_path = _P(data_dir) / ".leaderboard.csv"
                try:
                    # Concatenate per-(type, target) frames with two index columns so a reader can
                    # filter back to one slice. Honour pl.DataFrame vs pd.DataFrame via .write_csv /
                    # .to_csv duck typing; mixed cases concat after a unified to_pandas hop.
                    import pandas as _pd
                    _frames = []
                    for _tt_s, _by_name in _leaderboards.items():
                        for _tname, _lb in _by_name.items():
                            if hasattr(_lb, "write_csv") and not hasattr(_lb, "to_csv"):
                                # polars DF: dump to CSV bytes then re-read for the concat below.
                                import io as _io
                                _buf = _io.BytesIO()
                                _lb.write_csv(_buf)
                                _buf.seek(0)
                                _frame = _pd.read_csv(_buf)
                            elif hasattr(_lb, "to_csv"):
                                _frame = _pd.DataFrame(_lb) if not isinstance(_lb, _pd.DataFrame) else _lb
                            else:
                                _frame = _pd.DataFrame(_lb)
                            _frame.insert(0, "target_type", _tt_s)
                            _frame.insert(1, "target_name", _tname)
                            _frames.append(_frame)
                    if _frames:
                        _all_lb = _pd.concat(_frames, ignore_index=True)
                        _all_lb.to_csv(_csv_path, index=False)
                        if verbose:
                            logger.info("votenrank leaderboard exported: %s (%d rows)", _csv_path, len(_all_lb))
                except Exception as _csv_err:
                    logger.warning("votenrank leaderboard CSV export failed: %s", _csv_err)
    except Exception as _vn_err:
        logger.warning("votenrank leaderboard wiring failed: %s", _vn_err)

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
    # ctx is threaded through so train_recurrent_models can rerun score_ensemble with the recurrent member
    # entries appended (otherwise the recurrent models silently bypass the ensemble that already ran for the
    # same target during _train_one_target). test_df_pd added for the same reason - the helper needs all
    # three splits to compute per-member preds for the rerun.
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
        test_df_pd=test_df_pd,
        target_by_type=target_by_type,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        _non_neural_train_times=_non_neural_train_times,
        model_name=model_name,
        verbose=bool(verbose),
        ctx=ctx,
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
