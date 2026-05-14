"""
Core training functions for mlframe.

Contains the refactored train_mlframe_models_suite function.
"""

# *****************************************************************************************************************************************************
# IMPORTS
# *****************************************************************************************************************************************************

# -----------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------------------------------------------------------------------------------

import logging
import sys
from timeit import default_timer as timer

logger = logging.getLogger(__name__)


# *****************************************************************************************************************************************************
# IMPORTS (restored 2026-05-12 after Phase 5b extracted the leaf-utility
# helpers + their inline imports into core_utils.py). The big orchestrator
# train_mlframe_models_suite below still needs the same imports the helpers
# previously hauled in.
# *****************************************************************************************************************************************************

import glob
import os
from collections import defaultdict
from copy import deepcopy
from os.path import exists, join
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import joblib
import numpy as np
import pandas as pd
import polars as pl
import psutil
import scipy.stats as stats
from pyutilz.strings import slugify
from pyutilz.system import (
    clean_ram, ensure_dir_exists, tqdmu, tqdmu_lazy_start,
)
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import category_encoders as ce

from ..configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    ConfidenceAnalysisConfig,
    DummyBaselinesConfig,
    FeatureSelectionConfig,
    FeatureTypesConfig,
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
from ..baseline_diagnostics import BaselineDiagnostics, format_baseline_diagnostics_report
from ..drift_report import compute_label_distribution_drift, format_drift_report
from ..extractors import FeaturesAndTargetsExtractor
from ..helpers import (
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
)
from ..io import load_mlframe_model
from ..models import LINEAR_MODEL_TYPES, is_linear_model, is_neural_model
from ..phases import format_phase_summary, phase, reset_phase_registry
from ..pipeline import (
    apply_preprocessing_extensions,
    fit_and_transform_pipeline,
    prepare_df_for_catboost,
)
from ..preprocessing import (
    create_split_dataframes,
    load_and_prepare_dataframe,
    preprocess_dataframe,
    save_split_artifacts,
)
from ..splitting import make_train_test_split
from ..strategies import (
    PipelineCache,
    get_polars_cat_columns,
    get_strategy,
)
from ..train_eval import process_model, select_target
from ..utils import (
    compute_model_input_fingerprint,
    drop_columns_from_dataframe,
    estimate_df_size_mb,
    filter_existing,
    get_pandas_view_of_polars_df,
    get_process_rss_mb,
    log_phase,
    log_ram_usage,
    maybe_clean_ram_and_gpu,
)
from mlframe.feature_selection.filters import MRMR
from mlframe.metrics import create_fairness_subgroups
from ...ensembling import score_ensemble

# All 27 leaf utility helpers + DEFAULT_PROBABILITY_THRESHOLD now live in
# core_utils.py. The re-export shim at the bottom of this file makes them
# available under the historical ``mlframe.training.core`` namespace, but
# the train_mlframe_models_suite orchestrator below uses them directly via
# this same core_utils import to avoid the forward-reference pitfall.
from .utils import (
    DEFAULT_PROBABILITY_THRESHOLD,
    _apply_outlier_detection_global,
    _auto_detect_feature_types,
    _augment_with_dropped_high_card_cols,
    _build_common_params_for_target,
    _build_full_column_from_splits,
    _build_pre_pipelines,
    _build_process_model_kwargs,
    _build_tier_dfs,
    _compute_fairness_subgroups,
    _convert_dfs_to_pandas,
    _df_shape_str,
    _drop_cols_df,
    _elapsed_str,
    _ensure_logging_visible,
    _entry_metric,
    _filter_polars_cat_features_by_dtype,
    _finalize_and_save_metadata,
    _get_pipeline_components,
    _initialize_training_defaults,
    _maybe_dispatch_to_ltr_ranker_suite,
    _log_cardinality_and_drift_snapshot,
    _phase_auto_detect_feature_types,
    _phase_fit_pipeline,
    _phase_global_outlier_detection,
    _phase_load_and_preprocess,
    _phase_pandas_conversion_and_cat_prep,
    _phase_train_val_test_split,
    _setup_model_directories,
    _should_skip_catboost_metamodel,
    _validate_feature_type_exclusivity,
    _validate_input_columns_against_metadata,
    _validate_trusted_path,
)
from ._phase_composite_discovery import run_composite_target_discovery
from ._phase_composite_post import run_composite_post_processing
from ._phase_temporal_audit import run_temporal_audit_batch
from ._phase_polars_fixes import apply_polars_categorical_fixes
from ._phase_recurrent import train_recurrent_models
from ._phase_finalize import finalize_suite
from ._phase_config_setup import setup_configuration
from ._phase_dummy_baselines import run_dummy_baselines
from ._phase_diagnostics import run_per_target_diagnostics
from ._phase_train_one_target import _train_one_target
from ._training_context import TrainingContext


from ._misc_helpers import _split_preds_probs  # noqa: F401 — re-exported for callers


def _maybe_clear_shim_cache(est):
    """Clear XGB/LGB shim caches on estimator if present. Duck-typed via callable check."""
    fn = getattr(est, "clear_cache", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def _prep_polars_df(_df, strategy, cat_features, category_map):
    """Prepare a single Polars DF for native-model consumption."""
    if _df is None:
        return None
    if category_map is not None:
        return strategy.prepare_polars_dataframe(_df, cat_features, category_map=category_map)
    return strategy.prepare_polars_dataframe(_df, cat_features)


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
    # 2026-05-04: explicit target-type opt-in. None = auto-detected from
    # FTE.build_targets (preserves the historical classification/regression
    # routing). Set to TargetTypes.LEARNING_TO_RANK to route to the ranker
    # suite (CB/XGB/LGB native rankers + RRF/Borda ensembling). Other
    # target types stay on the standard pipeline.
    target_type: Optional["TargetTypes"] = None,
    ranking_config: Optional["LearningToRankConfig"] = None,
    # Existing typed configs (can be dicts or Pydantic objects)
    preprocessing_config: Optional[Union[PreprocessingConfig, Dict]] = None,
    split_config: Optional[Union[TrainingSplitConfig, Dict]] = None,
    pipeline_config: Optional[Union[PreprocessingBackendConfig, Dict]] = None,
    preprocessing_extensions: Optional[Union["PreprocessingExtensionsConfig", Dict]] = None,
    feature_types_config: Optional[Union[FeatureTypesConfig, Dict]] = None,
    linear_model_config: Optional[LinearModelConfig] = None,
    hyperparams_config: Optional[Union[ModelHyperparamsConfig, Dict]] = None,
    behavior_config: Optional[Union[TrainingBehaviorConfig, Dict]] = None,
    multilabel_dispatch_config: Optional["MultilabelDispatchConfig"] = None,
    # 2026-04-27 typed configs (replace prior dict pass-through + 9 orphan kwargs)
    reporting_config: Optional[Union["ReportingConfig", Dict]] = None,
    output_config: Optional[Union["OutputConfig", Dict]] = None,
    outlier_detection_config: Optional[Union["OutlierDetectionConfig", Dict]] = None,
    feature_selection_config: Optional[Union[FeatureSelectionConfig, Dict]] = None,
    confidence_analysis_config: Optional[Union["ConfidenceAnalysisConfig", Dict]] = None,
    # 2026-05-10: opt-out diagnostic that runs once per (target_type, target_name)
    # before per-target training. Reports raw headline metric, top-K feature
    # ablation deltas, init_score baseline, and a composite_recommendation flag
    # consumed by future composite-target discovery. Default ON; set
    # ``BaselineDiagnosticsConfig(enabled=False)`` to skip.
    baseline_diagnostics_config: Optional[Union["BaselineDiagnosticsConfig", Dict]] = None,
    # 2026-05-10: opt-out trivial-baseline floor diagnostic. Sit-alongside
    # BaselineDiagnostics — answers "is the task even hard?" via a per-
    # target table of dummy/naive baselines (mean / median / prior /
    # most_frequent / per_group / TS-naive when timestamps monotonic / LTR
    # random_within_query / multilabel per-label-prior). Verdict line + plot
    # for the strongest baseline only. Default ON; opt-out individual
    # target_types via ``DummyBaselinesConfig.apply_to_target_types``.
    dummy_baselines_config: Optional[Union["DummyBaselinesConfig", Dict]] = None,
    # 2026-05-10: quantile-regression knobs (alphas / crossing-fix / etc).
    # Consumed by:
    # - ``compute_dummy_baselines`` per-α empirical-quantile dispatcher
    #   when ``target_type == quantile_regression`` (auto-picks
    #   ``alphas`` from this config so the operator doesn't have to
    #   restate them per call).
    # - Future per-strategy wiring for native multi-quantile fits
    #   (CB MultiQuantile, XGB ``quantile_alpha``, LGB scalar wrapper).
    quantile_regression_config: Optional[Union["QuantileRegressionConfig", Dict]] = None,
    # 2026-05-10: opt-IN auto-discovery of composite-target transforms
    # (``T = f(y, base)``) for regression targets. When enabled, runs
    # MI-gain ranking after baseline_diagnostics and adds the discovered
    # composite targets to ``target_by_type``; the existing per-target
    # training loop then trains models on each. Discovered specs are
    # stored on ``metadata["composite_target_specs"]`` for downstream
    # inversion at predict time. Default OFF; set
    # ``CompositeTargetDiscoveryConfig(enabled=True)`` to opt in.
    # ``MLFRAME_DISABLE_COMPOSITE=1`` env var overrides to OFF
    # regardless of the config (kill switch for production rollback).
    composite_target_discovery_config: Optional[Union["CompositeTargetDiscoveryConfig", Dict]] = None,
    # 2026-05-09 phase Q: opt-in feature-handling overhaul. When set,
    # the suite logs the resolved per-model handler chain via
    # ``fhc.describe()`` at start. Consumer wiring (replacing
    # pipeline_config / feature_types_config with FHC-driven handler
    # outputs) lands in phase F-J follow-up; phase Q just exposes the
    # surface so existing pipelines aren't disturbed and users can
    # build + introspect FHC alongside the legacy path.
    feature_handling_config: Optional[Any] = None,
    # Misc
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

        preprocessing_config: Preprocessing configuration. Custom transformer overrides
            (``scaler``, ``imputer``, ``category_encoder``) live here too; previously
            these were dict-typed orphans without a typed home before the refactor.
        split_config: Train/val/test split configuration
        pipeline_config: Pipeline configuration

        feature_selection_config: Feature selection configuration. Holds
            ``use_mrmr_fs``, ``mrmr_kwargs``, ``rfecv_models``, ``rfecv_kwargs``,
            ``custom_pre_pipelines``. Previously these were five separate top-level
            kwargs of this function.

        hyperparams_config: Model hyperparameters (iterations, learning rate, per-model kwargs).
            Accepts ModelHyperparamsConfig or dict. Defaults are built in.
        behavior_config: Training behavior flags (GPU preference, calibration, fairness).
            Accepts TrainingBehaviorConfig or dict. Defaults are built in.

        reporting_config: Calibration / training-report look. Holds figure size,
            chart toggles, the title-metrics template (``ICE BR_DECOMP ECE CMAEW
            LL ROC_AUC PR_AUC`` by default), histogram subplot toggles, inline
            population labels, FI plot config. Previously reachable only via the
            dict-typed pass-through which has been deleted.
        output_config: Filesystem destinations - ``data_dir``, ``models_dir``,
            ``plot_file``, ``save_charts``. Previously these were top-level kwargs.
        outlier_detection_config: Outlier-detector + ``apply_to_val`` (was
            ``od_val_set``). Previously these were top-level kwargs.

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

    # ==================================================================================
    # 0. INPUT VALIDATION
    # ==================================================================================

    if verbose:
        _ensure_logging_visible()

    reset_phase_registry()

    # Validate df parameter
    if not isinstance(df, (pd.DataFrame, pl.DataFrame, str)):
        raise TypeError(f"df must be pandas DataFrame, polars DataFrame, or path string, " f"got {type(df).__name__}")
    if isinstance(df, str) and not df.lower().endswith(".parquet"):
        raise ValueError(f"File path must be a .parquet file, got: {df}")

    # 2026-05-04: LTR opt-in early dispatch.
    # When ``target_type=TargetTypes.LEARNING_TO_RANK`` is explicit, route
    # to the focused ranker suite (CB/XGB/LGB native rankers + RRF/Borda
    # ensembling). Helper returns ``None`` for non-LTR call sites; a
    # non-None return means the LTR suite was invoked and we forward its
    # result straight to the caller.
    _ltr_result = _maybe_dispatch_to_ltr_ranker_suite(
        target_type=target_type,
        df=df,
        target_name=target_name,
        model_name=model_name,
        features_and_targets_extractor=features_and_targets_extractor,
        mlframe_models=mlframe_models,
        use_mlframe_ensembles=use_mlframe_ensembles,
        ranking_config=ranking_config,
        split_config=split_config,
        hyperparams_config=hyperparams_config,
        reporting_config=reporting_config,
        output_config=output_config,
        verbose=verbose,
    )
    if _ltr_result is not None:
        return _ltr_result

    # Validate required parameters
    if not target_name:
        raise ValueError("target_name cannot be empty")
    if not model_name:
        raise ValueError("model_name cannot be empty")
    if features_and_targets_extractor is None:
        raise ValueError("features_and_targets_extractor is required")
    # ==================================================================================
    # 1. CONFIGURATION SETUP -- extracted helper
    # ==================================================================================
    ctx = setup_configuration(
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
        model_name=model_name,
        target_name=target_name,
        mlframe_models=mlframe_models,
        verbose=verbose,
    )
    # Local aliases for heavily-used fields (keeps diff minimal in model loop)
    preprocessing_config = ctx.preprocessing_config
    pipeline_config = ctx.pipeline_config
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
    # ==================================================================================
    # 2. DATA LOADING & PREPROCESSING (extracted 2026-05-12 into a helper)
    # ==================================================================================
    (
        df,
        target_by_type,
        group_ids_raw,
        group_ids,
        timestamps,
        artifacts,
        additional_columns_to_drop,
        sample_weights,
        baseline_rss_mb,
        df_size_mb,
        sequences,
    ) = _phase_load_and_preprocess(
        df=df,
        preprocessing_config=preprocessing_config,
        features_and_targets_extractor=features_and_targets_extractor,
        recurrent_models=recurrent_models,
        sequences=sequences,
        verbose=verbose,
    )
    for _k in ("df", "target_by_type", "group_ids_raw", "group_ids", "timestamps",
               "artifacts", "additional_columns_to_drop", "sample_weights",
               "baseline_rss_mb", "df_size_mb", "sequences"):
        setattr(ctx, _k, locals()[_k])

    # ==================================================================================
    # 3. TRAIN/VAL/TEST SPLITTING (extracted 2026-05-12 into a helper)
    # ==================================================================================
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
    del df
    ctx.df = None  # deleted after split
    for _k in ("train_idx", "val_idx", "test_idx", "train_details", "val_details",
               "test_details", "train_df", "val_df", "test_df", "fairness_subgroups",
               "fairness_features", "train_sequences", "val_sequences", "test_sequences",
               "baseline_rss_mb"):
        setattr(ctx, _k, locals()[_k])

    # ==================================================================================
    # 4. PIPELINE FITTING & TRANSFORMATION (extracted 2026-05-12 into a helper)
    # ==================================================================================
    (
        train_df, val_df, test_df,
        pipeline, extensions_pipeline,
        cat_features, cat_features_polars,
        was_polars_input, all_models_polars_native, polars_pipeline_applied,
        train_df_polars_pre, val_df_polars_pre, test_df_polars_pre,
        pipeline_config, preprocessing_extensions,
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
    )
    for _k in ("train_df", "val_df", "test_df", "pipeline", "extensions_pipeline",
               "cat_features", "cat_features_polars", "was_polars_input",
               "all_models_polars_native", "polars_pipeline_applied",
               "train_df_polars_pre", "val_df_polars_pre", "test_df_polars_pre",
               "pipeline_config", "preprocessing_extensions"):
        setattr(ctx, _k, locals()[_k])

    # ==================================================================================
    # 4.5. AUTO-DETECT TEXT & EMBEDDING FEATURES (extracted 2026-05-12 into a helper)
    # ==================================================================================
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
    )

    # Pre-train cardinality + val/test drift snapshot (extracted to helper).
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

    # ==================================================================================
    # 5. MODEL TRAINING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 4: Model Training")

    # Initialize default values for training parameters
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

    # Get pipeline components (category_encoder, imputer, scaler) from typed config or defaults
    category_encoder, imputer, scaler = _get_pipeline_components(preprocessing_config, cat_features)

    # Compute trainset stats (Polars is more efficient, but pandas works too)
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

    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    # Actual training (extracted: pandas-conversion gating + cat-feature prep + Polars release)
    # -----------------------------------------------------------------------------------------------------------------------------------------------------
    (
        train_df_pd, val_df_pd, test_df_pd,
        train_df_polars, val_df_polars, test_df_polars,
        train_df, val_df, test_df,
        train_df_size_bytes_cached, val_df_size_bytes_cached,
        can_skip_pandas_conv, baseline_rss_mb,
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
    # 2026-05-14 fix: store cached sizes on ctx BEFORE per-target loop
    # so _train_one_target can read them via ctx.train_df_size_bytes_cached.
    ctx.train_df_size_bytes_cached = train_df_size_bytes_cached
    ctx.val_df_size_bytes_cached = val_df_size_bytes_cached

    # ==================================================================================
    # 4.5 OUTLIER DETECTION (once, before model training loops) -- extracted helper
    # ==================================================================================
    (
        filtered_train_df, filtered_val_df,
        filtered_train_idx, filtered_val_idx,
        train_od_idx, val_od_idx,
        outlier_detection_result,
        train_df_polars, val_df_polars,
    ) = _phase_global_outlier_detection(
        train_df_pd=train_df_pd,
        val_df_pd=val_df_pd,
        train_df_polars=train_df_polars,
        val_df_polars=val_df_polars,
        train_idx=train_idx,
        val_idx=val_idx,
        target_by_type=target_by_type,
        outlier_detector=outlier_detector,
        od_val_set=od_val_set,
        baseline_rss_mb=baseline_rss_mb,
        df_size_mb=df_size_mb,
        metadata=metadata,
        verbose=verbose,
    )

    # Store outlier-detection results on ctx for _train_one_target
    ctx.filtered_train_df = filtered_train_df
    ctx.filtered_val_df = filtered_val_df
    ctx.filtered_train_idx = filtered_train_idx
    ctx.filtered_val_idx = filtered_val_idx
    ctx.train_od_idx = train_od_idx
    ctx.val_od_idx = val_od_idx

    # ==================================================================================
    # 4.6 COMPOSITE-TARGET DISCOVERY (opt-in; default OFF) -- extracted helper
    # ==================================================================================
    target_by_type, metadata = run_composite_target_discovery(
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
    )

    # Polars categorical fixes (null-fill + dict alignment + utf8 cast) -- extracted helper
    (
        train_df_polars, val_df_polars, test_df_polars,
        train_df_pd, val_df_pd, test_df_pd,
        filtered_train_df, filtered_val_df,
    ) = apply_polars_categorical_fixes(
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
        can_skip_pandas_conv=can_skip_pandas_conv,
        was_polars_input=was_polars_input,
        verbose=bool(verbose),
    )

    # Save metadata EARLY (before training loops) so that if training is interrupted,
    # already-trained models are still usable with the saved pipeline/preprocessing
    _finalize_and_save_metadata(
        metadata=metadata,
        outlier_detector=outlier_detector,
        outlier_detection_result=outlier_detection_result,
        trainset_features_stats=trainset_features_stats,
        data_dir=data_dir,
        models_dir=models_dir,
        target_name=target_name,
        model_name=model_name,
        verbose=verbose,
    )

    models = defaultdict(lambda: defaultdict(list))

    # Track mapping from slugified names to original names for load_mlframe_suite
    slug_to_original_target_type = {}
    slug_to_original_target_name = {}

    # 2026-04-26: precompute the temporal target audit ONCE for ALL target pairs
    # in a single polars multi-aggregation pass -- extracted helper.
    _all_target_audits = run_temporal_audit_batch(
        behavior_config=behavior_config,
        features_and_targets_extractor=features_and_targets_extractor,
        df=None,  # df already deleted after split; timestamps is the active source
        timestamps=timestamps,
        target_by_type=target_by_type,
        verbose=bool(verbose),
    )

    for target_type, targets in tqdmu_lazy_start(target_by_type.items(), desc="target type"):
        # Store original target_type mapping
        slug_to_original_target_type[slugify(str(target_type).lower())] = target_type

        # !TODO ! optimize for creation of inner feature matrices of cb,lgb,xgb here. They should be created once per featureset, not once per target.
        for cur_target_name, cur_target_values in tqdmu_lazy_start(targets.items(), desc="target"):
            _train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values)

    # Sync local refs with ctx after per-target loop (models/metadata mutated in place)
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
    can_skip_pandas_conv = ctx.can_skip_pandas_conv
    baseline_rss_mb = ctx.baseline_rss_mb
    train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    trainset_features_stats = ctx.trainset_features_stats
    slug_to_original_target_type = ctx.slug_to_original_target_type
    slug_to_original_target_name = ctx.slug_to_original_target_name

    # ==================================================================================
    # 6. RECURRENT MODEL TRAINING -- extracted helper
    # ==================================================================================
    models = train_recurrent_models(
        models=models,
        recurrent_models=recurrent_models,
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

    # Suite-end finalization -- extracted helper
    metadata = finalize_suite(
        models=models,
        metadata=metadata,
        outlier_detector=outlier_detector,
        outlier_detection_result=outlier_detection_result,
        trainset_features_stats=trainset_features_stats,
        data_dir=data_dir,
        models_dir=models_dir,
        target_name=target_name,
        model_name=model_name,
        slug_to_original_target_type=slug_to_original_target_type,
        slug_to_original_target_name=slug_to_original_target_name,
        _finalize_and_save_metadata=_finalize_and_save_metadata,
        verbose=bool(verbose),
    )
    ctx.metadata = metadata

    # ==================================================================================
    # 6-7. COMPOSITE POST-PROCESSING -- extracted helper
    # ==================================================================================
    models, metadata = run_composite_post_processing(
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
        plot_file=None,  # plot_file was per-target; composite-post uses suite-level
        verbose=bool(verbose),
    )

    # Release captured high-card column data
    try:
        _dropped_high_card_data.clear()
    except (NameError, AttributeError):
        pass

    return dict(models), metadata



# ----------------------------------------------------------------------
# Phase 5b split: re-export 27 leaf helpers + DEFAULT_PROBABILITY_THRESHOLD
# from core_utils for full back-compat.
# ----------------------------------------------------------------------
from .utils import (  # noqa: E402,F401
    DEFAULT_PROBABILITY_THRESHOLD,
    _ensure_logging_visible,
    _entry_metric,
    _augment_with_dropped_high_card_cols,
    _build_full_column_from_splits,
    _drop_cols_df,
    _validate_trusted_path,
    _df_shape_str,
    _elapsed_str,
    _detect_dataset_reuse_capabilities,
    _validate_input_columns_against_metadata,
    _filter_polars_cat_features_by_dtype,
    _auto_detect_feature_types,
    _validate_feature_type_exclusivity,
    _build_tier_dfs,
    _ensure_config,
    _apply_outlier_detection_global,
    _setup_model_directories,
    _build_common_params_for_target,
    _build_pre_pipelines,
    _build_process_model_kwargs,
    _convert_dfs_to_pandas,
    _get_pipeline_components,
    _compute_fairness_subgroups,
    _should_skip_catboost_metamodel,
    _create_initial_metadata,
    _initialize_training_defaults,
    _finalize_and_save_metadata,
)


# ----------------------------------------------------------------------
# Phase 5a split: re-export predict_*/load_* from core_predict for full
# back-compat. Existing callers ``from mlframe.training.core import
# predict_mlframe_models_suite, load_mlframe_suite`` keep working.
# ----------------------------------------------------------------------
from .predict import (  # noqa: E402,F401
    predict_mlframe_models_suite,
    predict_from_models,
    load_mlframe_suite,
)
