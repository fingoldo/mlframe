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
    TrainingConfig,
    TrainingSplitConfig,
)
from ..baseline_diagnostics import BaselineDiagnostics, format_baseline_diagnostics_report
from ..composite import CompositeTargetDiscovery
from ..drift_report import compute_label_distribution_drift, format_drift_report
from ..extractors import FeaturesAndTargetsExtractor
from ..helpers import (
    get_trainset_features_stats,
    get_trainset_features_stats_polars,
)
from ..io import load_mlframe_model
from ..models import LINEAR_MODEL_TYPES, is_linear_model
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
    _create_initial_metadata,
    _detect_dataset_reuse_capabilities,
    _df_shape_str,
    _drop_cols_df,
    _elapsed_str,
    _ensure_config,
    _ensure_logging_visible,
    _entry_metric,
    _filter_polars_cat_features_by_dtype,
    _finalize_and_save_metadata,
    _get_pipeline_components,
    _initialize_training_defaults,
    _maybe_dispatch_to_ltr_ranker_suite,
    _setup_model_directories,
    _should_skip_catboost_metamodel,
    _validate_feature_type_exclusivity,
    _validate_input_columns_against_metadata,
    _validate_trusted_path,
)


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
    # 1. CONFIGURATION SETUP
    # ==================================================================================

    if verbose:
        log_phase(f"Starting mlframe training suite: {model_name}")

    # Phase Q wiring: when the user passed a FeatureHandlingConfig,
    # log the resolved plan and validate against the active model
    # list so config-mismatch errors surface BEFORE any model fit.
    # The actual consumer-side wiring (replacing the legacy
    # pipeline_config + feature_types_config path with FHC-driven
    # handler outputs) lands in phases F-J.
    if feature_handling_config is not None:
        try:
            from mlframe.training.feature_handling import FeatureHandlingConfig as _FHC
            if isinstance(feature_handling_config, _FHC):
                if mlframe_models:
                    feature_handling_config.validate_against_models(list(mlframe_models))
                if verbose:
                    logger.info(
                        "[fhc] FeatureHandlingConfig active; resolved plan: %s",
                        feature_handling_config.describe(short=True),
                    )
        except ImportError:  # pragma: no cover
            pass

    # Convert dict configs to Pydantic if needed
    preprocessing_config = _ensure_config(preprocessing_config, PreprocessingConfig, {})
    pipeline_config = _ensure_config(pipeline_config, PreprocessingBackendConfig, {})
    feature_types_config = _ensure_config(feature_types_config, FeatureTypesConfig, {})
    split_config = _ensure_config(split_config, TrainingSplitConfig, {})
    hyperparams_config = _ensure_config(hyperparams_config, ModelHyperparamsConfig, {})
    behavior_config = _ensure_config(behavior_config, TrainingBehaviorConfig, {})
    reporting_config = _ensure_config(reporting_config, ReportingConfig, {})
    # 2026-05-11 (user request): propagate the residual-audit toggle from behavior_config into the evaluation module so ``report_model_perf`` callers (which don't carry a behavior_config reference) honor the suite-level setting. Module-level override; subsequent stand-alone calls keep the historical default since the override only flips while a suite run is in progress.
    from ..evaluation import _set_residual_audit_enabled as _set_resid_audit
    _set_resid_audit(getattr(behavior_config, "report_residual_audit", True))
    # 2026-05-10: honor ReportingConfig.plot_inline_display by setting
    # the process-wide MLFRAME_PLOT_INLINE_DISPLAY env var that
    # render_and_save consults. ``None`` = clear override (auto-detect
    # via __IPYTHON__ / sys.ps1 — existing default behavior); ``True`` /
    # ``False`` = explicit force. Useful for batch jupyter runs
    # (papermill / nbconvert / scheduled notebooks) that want save-only
    # despite running inside a kernel — saves ~50-200 ms / figure of
    # plotly inline render cost on 4-model x VAL+TEST x 6-ensemble
    # suites (accumulates to seconds).
    try:
        from mlframe.reporting.renderers.save import set_inline_display_mode as _set_idm
        _set_idm(getattr(reporting_config, "plot_inline_display", None))
    except Exception:
        pass
    output_config = _ensure_config(output_config, OutputConfig, {})
    outlier_detection_config = _ensure_config(outlier_detection_config, OutlierDetectionConfig, {})
    feature_selection_config = _ensure_config(feature_selection_config, FeatureSelectionConfig, {})
    confidence_analysis_config = _ensure_config(confidence_analysis_config, ConfidenceAnalysisConfig, {})
    baseline_diagnostics_config = _ensure_config(
        baseline_diagnostics_config, BaselineDiagnosticsConfig, {}
    )
    dummy_baselines_config = _ensure_config(
        dummy_baselines_config, DummyBaselinesConfig, {}
    )
    quantile_regression_config = _ensure_config(
        quantile_regression_config, QuantileRegressionConfig, {}
    )
    # 2026-05-10: pre-warm dummy_baselines numba kernels so the first
    # multi_output_regression call doesn't pay the 6-10s JIT cold-start.
    # Cost: one-time ~2-5s on first suite invocation per process; cached
    # afterwards. No-op when numba unavailable or dummy_baselines disabled.
    if dummy_baselines_config.enabled:
        try:
            from ..dummy_baselines import _warmup_numba_kernels
            _warmup_numba_kernels()
        except Exception:
            pass
    composite_target_discovery_config = _ensure_config(
        composite_target_discovery_config, CompositeTargetDiscoveryConfig, {}
    )
    # Production kill-switch: if the env var is set, force composite
    # discovery off regardless of the config the caller passed. Lets
    # ops disable on deployed services without code changes.
    import os as _os
    if _os.environ.get("MLFRAME_DISABLE_COMPOSITE", "").lower() in {"1", "true", "yes"}:
        composite_target_discovery_config = _ensure_config(
            {"enabled": False}, CompositeTargetDiscoveryConfig, {}
        )
        logger.info(
            "[CompositeTargetDiscovery] disabled by MLFRAME_DISABLE_COMPOSITE env var."
        )

    # 2026-04-27: pull scalar fields out of the typed configs for the existing
    # downstream code that takes plain locals. The scalar names match the
    # pre-refactor top-level kwargs so the ~100 sites of downstream code don't
    # need any further churn this revision. Deeper refactors (push the typed
    # configs down through select_target / process_model / train_and_evaluate_model)
    # are separate PRs.
    data_dir = output_config.data_dir
    models_dir = output_config.models_dir
    save_charts = output_config.save_charts

    # 2026-05-10 (rec e): surface reporting-knob resolution at suite
    # entry so the reader knows up-front whether plots will be saved /
    # rendered / short-circuited. Pre-2026-05-10 the cal-plot short-
    # circuit (no consumer in non-interactive sessions) was invisible
    # in logs; this line makes the active path explicit.
    if verbose:
        try:
            _is_interactive_logp = bool(__IPYTHON__)  # type: ignore[name-defined]  # noqa: F821
        except NameError:
            import sys as _sys_logp
            _is_interactive_logp = hasattr(_sys_logp, "ps1")
        _plot_dir = (
            f"{data_dir}/{models_dir}/{model_name}" if data_dir and save_charts else "(no save)"
        )
        _short_circuit_active = (not _is_interactive_logp) and not save_charts
        logger.info(
            "[reporting] save_charts=%s, plot_dir=%s, interactive=%s -- "
            "cal-plot short-circuit %s",
            save_charts, _plot_dir, _is_interactive_logp,
            "ACTIVE (renders skipped, no consumer)" if _short_circuit_active else "INACTIVE",
        )
        # 2026-05-10 perf advisory: warn on the kaleido bottleneck combo
        # (large dataset + plotly[png] default + save_charts on). Profiled
        # impact: 76s extra wall-time on a single 1M-row regression x lgb
        # run (4 chart-emitting points x ~12-15s/kaleido reload).
        # Multi-model x val+test x ensembles can balloon this to minutes.
        try:
            _po = getattr(reporting_config, "plot_outputs", "") or ""
        except NameError:
            _po = ""
        if (
            save_charts and "plotly" in _po and "png" in _po
        ):
            logger.warning(
                "[reporting] plot_outputs=%r emits PNG via kaleido, which "
                "spawns ~12-15s per chart on Chromium reload (Win/Linux). "
                "On large datasets (n>=1M, multi-model x val+test x "
                "ensembles) this can dominate wall-time by minutes. For "
                "fast runs use plot_outputs='matplotlib[png]' (10-20x "
                "faster, no Chromium overhead) or 'plotly[html]' (HTML-"
                "only, no kaleido at all -- HTML is interactive in jupyter "
                "and shareable as a file).",
                _po,
            )
    outlier_detector = outlier_detection_config.detector
    od_val_set = outlier_detection_config.apply_to_val
    use_mrmr_fs = feature_selection_config.use_mrmr_fs
    mrmr_kwargs = feature_selection_config.mrmr_kwargs
    rfecv_models = feature_selection_config.rfecv_models
    custom_pre_pipelines = feature_selection_config.custom_pre_pipelines if feature_selection_config.custom_pre_pipelines else None

    # Internal dict carrying ReportingConfig + PreprocessingConfig.{scaler,imputer,
    # category_encoder} down to the deep dict-key consumers in trainer.py
    # (_build_train_eval_configs assembles ReportingConfig / DataConfig from these
    # scalar keys). Built internally from the typed configs; no external dict
    # pass-through on the suite signature.
    common_params_dict: Dict[str, Any] = {}
    # ``title_metrics_tokens`` is a DERIVED field on ReportingConfig
    # (auto-populated by the model_validator from
    # ``title_metrics_template``). The deep consumer
    # ``_build_configs_from_params`` only accepts the source field, so
    # exclude the derived one to avoid an unexpected-kwarg TypeError;
    # ``train_and_evaluate_model`` re-derives it from the rebuilt
    # ReportingConfig object directly.
    # ``title_metrics_tokens`` is auto-derived by the model_validator;
    # the consumer accepts ``title_metrics_template`` and re-derives.
    # ``plot_outputs`` / ``multiclass_panels`` / ``multilabel_panels`` /
    # ``ltr_panels`` ARE in ``_build_configs_from_params``'s signature
    # (added 2026-05-08) so they thread through cleanly to
    # ``train_and_evaluate_model`` and on to ``report_model_perf``'s
    # auto-dispatcher.
    common_params_dict.update(
        reporting_config.model_dump(exclude={
            "title_metrics_tokens",
            # Suite-level only: consumed at line ~2269 above via
            # ``set_inline_display_mode``; the deep consumer
            # ``_build_configs_from_params`` does not accept it.
            "plot_inline_display",
        })
    )
    if preprocessing_config.scaler is not None:
        common_params_dict["scaler"] = preprocessing_config.scaler
    if preprocessing_config.imputer is not None:
        common_params_dict["imputer"] = preprocessing_config.imputer
    if preprocessing_config.category_encoder is not None:
        common_params_dict["category_encoder"] = preprocessing_config.category_encoder
    # ConfidenceAnalysisConfig fields - dumped into common_params_dict because
    # the deep consumer (_build_configs_from_params in trainer.py) reads scalar
    # kwargs and assembles ConfidenceAnalysisConfig from them. Field names map
    # 1:1 to the kwargs of that builder (`include_confidence_analysis`,
    # `confidence_analysis_use_shap`, etc.).
    common_params_dict["include_confidence_analysis"] = confidence_analysis_config.include
    common_params_dict["confidence_analysis_use_shap"] = confidence_analysis_config.use_shap
    common_params_dict["confidence_analysis_max_features"] = confidence_analysis_config.max_features
    common_params_dict["confidence_analysis_cmap"] = confidence_analysis_config.cmap
    common_params_dict["confidence_analysis_alpha"] = confidence_analysis_config.alpha
    common_params_dict["confidence_analysis_ylabel"] = confidence_analysis_config.ylabel
    common_params_dict["confidence_analysis_title"] = confidence_analysis_config.title
    common_params_dict["confidence_model_kwargs"] = dict(confidence_analysis_config.model_kwargs)

    # Opt-in: install SIGSEGV/SIGABRT handler + suppress Windows WER
    # popup so native crashes (e.g. XGBoost bad_malloc on too-large
    # frames) surface as Python tracebacks in Jupyter instead of hanging
    # the kernel on a modal dialog.
    if behavior_config.enable_crash_reporting:
        from mlframe.training.crash_reporting import enable_crash_reporting as _enable_crash_reporting
        _enable_crash_reporting()

    # Fix 9.4.2: report dataset-reuse capability of the installed GBDT
    # libraries. CB Pool.set_label/set_weight + CB wrapper's Pool-as-X
    # short-circuit enable reuse across weight schemas and same-type
    # targets. XGB/LGB sklearn wrappers currently lack the equivalent
    # (upstream FRs pending) -- only the per-build logging applies there.
    _dataset_reuse_caps = _detect_dataset_reuse_capabilities()
    logger.info("Dataset-reuse capabilities: %s", _dataset_reuse_caps)
    if not _dataset_reuse_caps.get("cb_pool_label_swap"):
        logger.warning(
            "  CatBoost Pool.set_label/set_weight not available in installed build -- "
            "mlframe will fall back to rebuilding the Pool on every weight schema and "
            "same-type target. Upgrade CatBoost to pick up the Pool label-swap PR."
        )

    # Fix 9.4.3: clear the process-wide CB Pool cache at every suite
    # entry. The cache is keyed on ``id(train_df)`` + column/shape
    # signature; across independent suite invocations (e.g. successive
    # pytest tests that each build their own train_df) Python can reuse
    # ids after GC, and if a later frame happens to have the same
    # columns + shape, we'd wrongly reuse a stale Pool built from
    # different underlying data. Clearing per-suite gives us per-run
    # locality without threading a session token through every strategy.
    try:
        from mlframe.training.trainer import (
            _CB_POOL_CACHE as _cb_cache,
            _CB_VAL_POOL_CACHE as _cb_val_cache,
        )
        _cb_cache.clear()
        _cb_val_cache.clear()
    except Exception:
        pass

    # Default models
    if mlframe_models is None:
        mlframe_models = ["cb", "lgb", "xgb", "mlp", "linear"]

    # Metadata for tracking
    metadata = _create_initial_metadata(
        model_name=model_name,
        target_name=target_name,
        mlframe_models=mlframe_models,
        preprocessing_config=preprocessing_config,
        pipeline_config=pipeline_config,
        split_config=split_config,
    )
    # 2026-05-10: schema bump for composite-target discovery integration.
    # v1 = no composite-target keys. v2 adds:
    # ``composite_target_specs``, ``composite_target_failures``,
    # ``baseline_diagnostics``. Existing v1 loaders that ``.get(...)``
    # the new keys with a default continue to work; loaders that hard-
    # require all-known keys must check ``schema_version >= 2``.
    metadata["schema_version"] = 2

    # ==================================================================================
    # 2. DATA LOADING & PREPROCESSING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 1: Data Loading & Preprocessing")

    # Load and prepare dataframe
    t0_phase1 = timer()
    with phase("load_and_prepare_dataframe"):
        df = load_and_prepare_dataframe(df, preprocessing_config, verbose=verbose)
    if verbose:
        logger.info("  load_and_prepare_dataframe done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_phase1))

    # Apply features_and_targets_extractor to extract targets
    if verbose:
        logger.info("Create additional features & extracting targets...")

    t0_fte = timer()
    df, target_by_type, group_ids_raw, group_ids, timestamps, artifacts, additional_columns_to_drop, sample_weights = features_and_targets_extractor.transform(
        df
    )
    if verbose:
        logger.info("  features_and_targets_extractor done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_fte))

    # Capture baseline RSS + DF size NOW -- before any downstream steps that may allocate
    # transient state (get_sequences, drop_columns, preprocess). Used by
    # maybe_clean_ram_and_gpu() at later sites to skip ~0.6s gc calls when memory
    # pressure is low. On 100GB production DFs the growth/free-RAM thresholds trip and
    # clean_ram fires; on small test DFs all sites are skipped.
    baseline_rss_mb = get_process_rss_mb()
    df_size_mb = estimate_df_size_mb(df)

    # Extract sequences for recurrent models (if not provided directly)
    if recurrent_models and sequences is None:
        extracted_sequences = features_and_targets_extractor.get_sequences(df)
        if extracted_sequences is not None:
            sequences = extracted_sequences
            if verbose:
                logger.info("Extracted %d sequences from DataFrame", len(sequences))
        elif verbose:
            logger.warning("recurrent_models specified but no sequences provided or extracted")

    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-FTE")
    if verbose:
        log_ram_usage()

    # Drop columns AFTER features_and_targets_extractor (columns might be needed by features_and_targets_extractor or created by it)
    df = drop_columns_from_dataframe(
        df,
        additional_columns_to_drop=additional_columns_to_drop,
        config_drop_columns=preprocessing_config.drop_columns,
        verbose=verbose,
    )

    # Preprocess dataframe (handle nulls, infinities, constants, dtypes)
    t0_preproc = timer()
    df = preprocess_dataframe(df, preprocessing_config, verbose=verbose)
    if verbose:
        logger.info("  preprocess_dataframe done -- %s, %s", _df_shape_str(df), _elapsed_str(t0_preproc))
        logger.info("  PHASE 1 total: %s", _elapsed_str(t0_phase1))

    # ==================================================================================
    # 3. TRAIN/VAL/TEST SPLITTING
    # ==================================================================================

    if verbose:
        log_phase("PHASE 2: Train/Val/Test Splitting")

    t0_phase2 = timer()
    if verbose:
        logger.info(f"Making train_val_test split...")
    # Auto-stratify by target for classification when no timestamps are
    # available. Without this, the unstratified shuffle path can hand
    # an unlucky val/test slice with zero minority-class rows for
    # rare imbalance ratios (fuzz default-seed c0134, seed=99 c0040 --
    # rare_1pct + binary class produces 50 positives out of 5000;
    # random 400-row val_shuf can land all-class-0). Stratification
    # preserves class proportions across train/val/test. Skipped when
    # timestamps are present (the splitter prefers temporal ordering
    # there) or for multitarget setups where picking ONE target as
    # the stratify key is arbitrary.
    _stratify_y = None
    if timestamps is None and isinstance(target_by_type, dict):
        _classification_targets = []
        _has_multilabel = False
        for _tt, _named in target_by_type.items():
            _tt_name = getattr(_tt, "name", str(_tt)).upper()
            if "MULTILABEL" in _tt_name:
                _has_multilabel = True
                break
            if "CLASS" in _tt_name and isinstance(_named, dict):
                for _tn, _tv in _named.items():
                    if _tv is not None:
                        _classification_targets.append(_tv)
        # Multilabel stratification needs the optional ``iterative-
        # stratification`` package. Skip it to avoid forcing the dep on
        # users who don't have it (the ``MultilabelStratifiedShuffleSplit``
        # branch raises ``ModuleNotFoundError`` deep in the splitter).
        # Single-label classification (binary / multiclass) uses sklearn's
        # built-in ``StratifiedShuffleSplit`` which is always available.
        if _has_multilabel:
            _stratify_y = None
        elif len(_classification_targets) == 1:
            try:
                _arr = np.asarray(_classification_targets[0])
                # Guard: only stratify when stratification is meaningful --
                # all classes have at least 2 rows, otherwise sklearn's
                # StratifiedShuffleSplit raises "least populated class has
                # only 1 member". Also limit to 1-D targets -- 2-D would
                # route to the multilabel splitter (already excluded above
                # but defense in depth).
                if _arr.ndim == 1:
                    _u, _c = np.unique(_arr, return_counts=True)
                    if len(_u) >= 2 and _c.min() >= 2:
                        _stratify_y = _arr
            except Exception:
                _stratify_y = None
    # Group-aware splitting opt-in. When the extractor produced ``group_ids``
    # (e.g. ``SimpleFeaturesAndTargetsExtractor(group_field="well_id")``) and
    # ``split_config.use_groups`` is True (default), route through
    # ``GroupShuffleSplit`` so that no well straddles train/val/test. The
    # splitter already supports this via the ``groups=`` argument; we just
    # need to wire ``group_ids`` through (previously it was extracted but
    # never reached the splitter).
    _groups = group_ids if (split_config.use_groups and group_ids is not None and len(group_ids) > 0) else None
    with phase("split_data"):
        train_idx, val_idx, test_idx, train_details, val_details, test_details = make_train_test_split(
            df=df,
            timestamps=timestamps,
            stratify_y=_stratify_y,
            groups=_groups,
            **split_config.model_dump(exclude={"use_groups"}),
        )
    if verbose:
        log_ram_usage()

    # Save artifacts
    if data_dir:
        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=group_ids_raw,
            artifacts=artifacts,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name=target_name,
            model_name=model_name,
        )

    metadata.update(
        {
            "train_details": train_details,
            "val_details": val_details,
            "test_details": test_details,
            "train_size": len(train_idx),
            "val_size": len(val_idx),
            "test_size": len(test_idx),
        }
    )

    # Pre-compute fairness subgroups from full df BEFORE splitting
    # (bins must cover all rows for train/val/test evaluation)
    fairness_subgroups, fairness_features = _compute_fairness_subgroups(df, behavior_config)
    if verbose:
        if fairness_features and fairness_subgroups is None:
            logger.warning(f"Fairness features {fairness_features} specified but subgroups could not be computed")
        elif fairness_subgroups is not None:
            logger.info("Computed %d fairness subgroups", len(fairness_subgroups))

    # Create split dataframes
    train_df, val_df, test_df = create_split_dataframes(
        df=df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    if verbose:
        logger.info("  Split shapes -- train: %s, val: %s, test: %s", _df_shape_str(train_df), _df_shape_str(val_df), _df_shape_str(test_df))
        logger.info("  PHASE 2 total: %s", _elapsed_str(t0_phase2))

    # Split sequences by train/val/test indices (for recurrent models)
    train_sequences, val_sequences, test_sequences = None, None, None
    if sequences is not None:
        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx] if val_idx is not None else None
        test_sequences = [sequences[i] for i in test_idx]
        if verbose:
            logger.info("Split sequences: train=%d, val=%d, test=%d", len(train_sequences), len(val_sequences) if val_sequences else 0, len(test_sequences))

    # Delete original df to free RAM
    if verbose:
        logger.info("Deleting original DataFrame to free RAM...")

    del df
    # Refresh baseline BEFORE the check: `del df` just freed df_size_mb worth of RAM,
    # so an unrefreshed baseline would yield negative growth and skip cleanup precisely
    # when arena-release would be most effective.
    baseline_rss_mb = get_process_rss_mb()
    baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-split (del df)")

    if verbose:
        log_ram_usage()

    # ==================================================================================
    # 4. PIPELINE FITTING & TRANSFORMATION
    # ==================================================================================

    t0_phase3 = timer()
    if verbose:
        log_phase("PHASE 3: Pipeline Fitting & Transformation")

    # Track if input is Polars before pipeline transformation
    was_polars_input = isinstance(train_df, pl.DataFrame)

    # Resolve strategies once for subsequent polars-native gating (avoids redundant lookups).
    _strategies_for_polars_check = [get_strategy(m) for m in mlframe_models] if mlframe_models else []
    all_models_polars_native = bool(_strategies_for_polars_check) and all(
        s.supports_polars for s in _strategies_for_polars_check
    )

    # Auto-skip categorical encoding when all models handle categoricals natively.
    # This avoids wasting time encoding columns that polars-native models don't need,
    # and avoids the .clone() overhead for preserving pre-pipeline originals.
    if was_polars_input and not pipeline_config.skip_categorical_encoding:
        if all_models_polars_native:
            pipeline_config = pipeline_config.model_copy(update={"skip_categorical_encoding": True})
            if verbose:
                logger.info("  All models %s support Polars natively -- skipping categorical encoding in pipeline", mlframe_models)

    # 2026-04-24 (fuzz extension): datetime columns must be decomposed
    # BEFORE the pre-pipeline clone, otherwise ``train_df_polars_pre`` and
    # friends retain the raw datetime and reach downstream (linear
    # pre_pipeline, MRMR, sklearn encoders, CB Pool) where numpy /
    # sklearn / CB all raise on DateTime64DType. Decompose once here
    # via the canonical ``create_date_features`` helper -- same
    # treatment we'd apply inside fit_and_transform_pipeline, just
    # lifted earlier so the clone inherits the decomposition.
    import numpy as _np_dt
    def _detect_datetime_cols(df_):
        if df_ is None:
            return []
        if isinstance(df_, pl.DataFrame):
            return [name for name, dt in df_.schema.items()
                    if isinstance(dt, (pl.Datetime, pl.Date))]
        if hasattr(df_, "dtypes"):
            return [c for c in df_.columns
                    if pd.api.types.is_datetime64_any_dtype(df_[c])]
        return []

    _dt_cols = _detect_datetime_cols(train_df)
    if _dt_cols:
        from mlframe.feature_engineering.basic import create_date_features
        _dt_methods = {
            "day": _np_dt.int8,
            "weekday": _np_dt.int8,
            "month": _np_dt.int8,
            "hour": _np_dt.int8,
        }
        if verbose:
            logger.info(
                "Decomposing %d datetime column(s) into numeric features "
                "(day/weekday/month/hour) before pre-pipeline clone: %s",
                len(_dt_cols), _dt_cols,
            )
        train_df = create_date_features(
            train_df, cols=_dt_cols, delete_original_cols=True,
            methods=_dt_methods,
        )
        if val_df is not None:
            v_cols = [c for c in _dt_cols if c in val_df.columns]
            if v_cols:
                val_df = create_date_features(
                    val_df, cols=v_cols, delete_original_cols=True,
                    methods=_dt_methods,
                )
        if test_df is not None:
            t_cols = [c for c in _dt_cols if c in test_df.columns]
            if t_cols:
                test_df = create_date_features(
                    test_df, cols=t_cols, delete_original_cols=True,
                    methods=_dt_methods,
                )

    # Save pre-pipeline Polars originals for the Polars fastpath.
    # Only clone when the pipeline will actually modify categorical columns;
    # when skip_categorical_encoding=True the pipeline preserves dtypes so the
    # original DF reference is sufficient (B1 optimization -- saves 100GB+ clone).
    needs_polars_pre_clone = (
        was_polars_input
        and not pipeline_config.skip_categorical_encoding
        and pipeline_config.categorical_encoding is not None
    )
    if was_polars_input:
        if needs_polars_pre_clone:
            train_df_polars_pre = train_df.clone()
            val_df_polars_pre = val_df.clone() if isinstance(val_df, pl.DataFrame) else None
            test_df_polars_pre = test_df.clone() if isinstance(test_df, pl.DataFrame) else None
            if verbose:
                logger.info(f"  Cloned pre-pipeline Polars originals (pipeline will modify categoricals)")
        else:
            # No clone needed -- pipeline won't touch categoricals, reuse references
            train_df_polars_pre = train_df
            val_df_polars_pre = val_df if isinstance(val_df, pl.DataFrame) else None
            test_df_polars_pre = test_df if isinstance(test_df, pl.DataFrame) else None
            if verbose:
                logger.info(f"  Skipped pre-pipeline clone (skip_categorical_encoding=True)")
        cat_features_polars = get_polars_cat_columns(train_df)
    else:
        train_df_polars_pre = None
        val_df_polars_pre = None
        test_df_polars_pre = None
        cat_features_polars = []

    # Pass user-specified text/embedding features to exclude from encoding/scaling.
    # Auto-detection happens later (after pipeline, when cat_features are known).
    t0_fit_pipeline = timer()
    train_df, val_df, test_df, pipeline, cat_features = fit_and_transform_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=pipeline_config,
        ensure_float32=preprocessing_config.ensure_float32_dtypes,
        verbose=verbose,
        text_features=feature_types_config.text_features,
        embedding_features=feature_types_config.embedding_features,
    )
    if verbose:
        logger.info("  fit_and_transform_pipeline done in %s", _elapsed_str(t0_fit_pipeline))

    # Track if Polars-ds pipeline was applied (to skip redundant pre_pipeline transforms later)
    polars_pipeline_applied = was_polars_input and pipeline_config.prefer_polarsds and pipeline is not None

    # Apply shared sklearn-based extensions (scaler override / poly / dim-reducer / ...).
    # When preprocessing_extensions is None (default) this is a zero-cost noop and the
    # Polars-native fastpath is preserved byte-for-byte.
    if preprocessing_extensions is not None and isinstance(preprocessing_extensions, dict):
        preprocessing_extensions = PreprocessingExtensionsConfig(**preprocessing_extensions)
    t0_ext = timer()
    train_df, val_df, test_df, extensions_pipeline = apply_preprocessing_extensions(
        train_df, val_df, test_df, preprocessing_extensions, verbose=verbose,
    )
    if verbose and preprocessing_extensions is not None:
        logger.info("  apply_preprocessing_extensions done in %s", _elapsed_str(t0_ext))
    if extensions_pipeline is not None:
        # cat_features are materialised into numeric columns by the sklearn stack.
        cat_features = []

    metadata["pipeline"] = pipeline
    metadata["extensions_pipeline"] = extensions_pipeline
    metadata["cat_features"] = cat_features
    metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns

    if verbose:
        logger.info("  Pipeline done -- train: %s, cat_features: %s", _df_shape_str(train_df), cat_features or '(none)')
        # Only emit the Polars-side list when it actually differs from the
        # post-pipeline list — when the polars fastpath skips encoding the two
        # are identical and the second log line is pure noise (one user's log
        # had 23 names duplicated back-to-back).
        if was_polars_input and cat_features_polars and list(cat_features_polars) != list(cat_features or []):
            logger.info("  Pre-pipeline Polars cat_features: %s", cat_features_polars)
        logger.info("  PHASE 3 total: %s", _elapsed_str(t0_phase3))

    # ==================================================================================
    # 4.5. AUTO-DETECT TEXT & EMBEDDING FEATURES
    # ==================================================================================

    # Use pre-pipeline DF for auto-detection (original dtypes preserved).
    detect_df = train_df_polars_pre if was_polars_input else train_df
    # Merge pipeline-detected and pre-pipeline Polars categorical columns
    raw_cat_features = list(set((cat_features or []) + (cat_features_polars or [])))
    # Honor ONLY strictly-user-declared pl.Categorical columns as already-assigned:
    # the user explicitly marked them as categorical so they must not be promoted to
    # text. pl.Utf8/String columns (which `cat_features_polars` also includes) and
    # pipeline-detected string cats are eligible for text-promotion based on
    # cardinality. `effective_cat_features` below removes promoted columns from the
    # cat list so no column double-counts.
    if was_polars_input:
        user_polars_cats = [
            c for c, dt in zip(detect_df.columns, detect_df.dtypes)
            if dt == pl.Categorical
        ]
    else:
        user_polars_cats = []
    text_features, embedding_features, auto_high_card_drop = _auto_detect_feature_types(
        detect_df, feature_types_config, user_polars_cats, verbose=verbose,
    )
    # Fix 6 correction (2026-04-22): when ``use_text_features=False``,
    # the detector populates ``auto_high_card_drop`` with columns that
    # exceed the cardinality threshold. Leaving them as cat_features
    # silently OOMs XGB's QuantileDMatrix (observed on prod 9_018_479-
    # row x ``skills_text:2_063_092``-unique run 2026-04-22) and
    # balloons CB's model artefact. Drop them from the train/val/test
    # splits AND from ``raw_cat_features`` here so every downstream
    # strategy sees the same reduced frame.
    # 2026-05-10: capture pre-drop high-card column DATA so dummy_baselines
    # per_group_mean can use these as group keys downstream. Tree models drop
    # them to avoid XGB QuantileDMatrix OOM, but a simple groupby on the same
    # column gives an excellent reference baseline (well_id with 600+ unique
    # values for well-log regression, user_id for marketplace data, etc.).
    _dropped_high_card_data = {}
    if auto_high_card_drop:
        for _col in auto_high_card_drop:
            _col_frames = {}
            for _label, _frame in (("train", train_df), ("val", val_df), ("test", test_df)):
                if _frame is None:
                    continue
                _cols = _frame.columns if hasattr(_frame, "columns") else []
                if _col not in _cols:
                    continue
                try:
                    if isinstance(_frame, pl.DataFrame):
                        _col_frames[_label] = _frame[_col].to_numpy()
                    else:
                        _col_frames[_label] = np.asarray(_frame[_col])
                except Exception:
                    continue
            if _col_frames:
                _dropped_high_card_data[_col] = _col_frames
        train_df = _drop_cols_df(train_df, auto_high_card_drop)
        val_df = _drop_cols_df(val_df, auto_high_card_drop)
        test_df = _drop_cols_df(test_df, auto_high_card_drop)
        if was_polars_input:
            if train_df_polars_pre is not None:
                train_df_polars_pre = _drop_cols_df(train_df_polars_pre, auto_high_card_drop)
            if val_df_polars_pre is not None:
                val_df_polars_pre = _drop_cols_df(val_df_polars_pre, auto_high_card_drop)
            if test_df_polars_pre is not None:
                test_df_polars_pre = _drop_cols_df(test_df_polars_pre, auto_high_card_drop)
        raw_cat_features = [c for c in raw_cat_features if c not in auto_high_card_drop]
        # Keep the metadata ``columns`` snapshot in sync with the
        # reduced frame -- load-time schema diff (_validate_input_
        # columns_against_metadata) uses this to filter serving df to
        # the trained subset, so a stale list re-introduces the
        # dropped column at inference and the model errors on shape.
        metadata["columns"] = train_df.columns.tolist() if isinstance(train_df, pd.DataFrame) else train_df.columns
    # Remove auto-detected text/embedding features from cat list (they're not categoricals)
    text_emb_set = set(text_features) | set(embedding_features)
    effective_cat_features = [c for c in raw_cat_features if c not in text_emb_set]
    _validate_feature_type_exclusivity(text_features, embedding_features, effective_cat_features)

    # CRITICAL: downstream code (select_target, strategy.build_pipeline, the CB
    # pandas-fallback path in _train_model_with_fallback, etc.) must see the
    # *deduplicated* list. Before this reassignment the unfiltered ``cat_features``
    # still contained text-promoted columns like 'category' / 'skills_text', and
    # CatBoost's pandas path then rejected the run with
    #   "column 'category' has dtype 'category' but is not in cat_features list"
    # -- the column was pd.Categorical in the pandas view (preserved from the
    # original Polars schema) AND listed in text_features, so CB's Pool
    # refused to accept it. Reassigning here flows the correct set to every
    # downstream user via the single ``cat_features`` binding.
    cat_features = effective_cat_features
    # Keep metadata in sync with the post-detection cat_features list.
    # ``metadata["cat_features"]`` was set to the un-filtered list at line
    # 1999 (before auto-detection ran); without this re-sync, Fix 6's
    # drop-list / text auto-promotion never propagates to the persisted
    # metadata, so load-time schema validation and inference consumers
    # see a stale cat_features list that includes already-removed
    # high-cardinality columns (2026-04-22 regression caught by
    # test_fix6_use_text_features_false_end_to_end_xgb_does_not_see_highcard).
    metadata["cat_features"] = cat_features

    # One-time Polars string->Categorical cast (shared across all models in the loop).
    # XGBoost's arrow bridge rejects pl.Utf8/large_string ("KeyError: DataType(large_string)");
    # HGB/LightGBM/CatBoost all prefer Categorical over raw strings when encoding is skipped.
    # Exclude text/embedding features -- those must remain as raw strings for CatBoost text/emb handling.
    if was_polars_input and all_models_polars_native and pipeline_config.skip_categorical_encoding:
        _string_types = (pl.Utf8, pl.String) if hasattr(pl, "String") else (pl.Utf8,)
        _keep_as_string = text_emb_set
        def _precast_strings(df):
            if df is None:
                return df
            str_cols = [c for c, dt in zip(df.columns, df.dtypes)
                        if dt in _string_types and c not in _keep_as_string]
            return df.with_columns([pl.col(c).cast(pl.Categorical) for c in str_cols]) if str_cols else df
        _pre_train = _precast_strings(train_df)
        if _pre_train is not train_df:
            train_df = _pre_train
            val_df = _precast_strings(val_df)
            test_df = _precast_strings(test_df)
            # Always cast the pre-pipeline Polars refs too: downstream Polars fastpath
            # uses `train_df_polars_pre` as the model input, and without the cast XGBoost
            # would hit `DataType(large_string)` again. Identity-based re-pointing is not
            # sufficient -- a no-op pipeline may return a new object that is not `is` train_df.
            train_df_polars_pre = _precast_strings(train_df_polars_pre)
            val_df_polars_pre = _precast_strings(val_df_polars_pre)
            test_df_polars_pre = _precast_strings(test_df_polars_pre)
            if verbose:
                logger.info("  Cast Polars string columns -> Categorical once (shared across model loop)")

    if verbose and (text_features or embedding_features):
        logger.info("  Feature types -- text: %s, embedding: %s, cat: %s", text_features, embedding_features, cat_features or '(none)')

    # Pre-train cardinality + val/test drift snapshot.
    #
    # Cardinality: without this, a native XGB/CB crash on
    # high-cardinality categoricals leaves us guessing at the input.
    #
    # Drift: for time-ordered splits (the common case here), val and
    # test can contain category values that never appeared in train --
    # XGB 3.x on Windows crashes silently during val IterativeDMatrix
    # construction when this happens (observed 2026-04-20 on
    # prod_jobsdetails). We compute ``val_minus_train`` /
    # ``test_minus_train`` unseen-category counts here and emit a
    # WARNING for any column with non-trivial drift, so the operator
    # sees which column is the crash suspect BEFORE the kernel dies.
    #
    # Skip if cardinality > 100k (text-sized columns): the anti-join
    # is expensive and unseen-category semantics don't cleanly apply
    # to free-text columns (CB handles them via TF-IDF, XGB drops them).
    if verbose:
        all_cat_cols = list(cat_features or []) + list(text_features or []) + list(embedding_features or [])
        if all_cat_cols and train_df is not None:
            try:
                _DRIFT_SKIP_CARD = 100_000
                is_polars = isinstance(train_df, pl.DataFrame)
                pairs = []
                for c in all_cat_cols:
                    if c not in train_df.columns:
                        continue
                    if is_polars:
                        n_unique = train_df[c].n_unique()
                    else:
                        n_unique = int(train_df[c].nunique(dropna=False))
                    pairs.append((c, n_unique))
                pairs.sort(key=lambda x: -x[1])
                summary = ", ".join(f"{c}:{n:_}" for c, n in pairs)
                logger.info("  Categorical cardinalities (train, n_unique, desc): %s", summary)

                # Drift log: val/test categories not seen in train.
                if is_polars and val_df is not None and test_df is not None and val_df.height > 0:
                    drift_rows = []
                    for c, card_train in pairs:
                        if card_train > _DRIFT_SKIP_CARD:
                            continue  # text-sized, anti-join is expensive
                        if c not in val_df.columns or c not in test_df.columns:
                            continue
                        # Polars anti-join gives count of uniques in {val|test} missing from train.
                        tr_uniq = train_df.select(pl.col(c).drop_nulls().unique().alias(c))
                        v_uniq  = val_df.select(pl.col(c).drop_nulls().unique().alias(c))
                        te_uniq = test_df.select(pl.col(c).drop_nulls().unique().alias(c))
                        val_only  = v_uniq.join(tr_uniq, on=c, how="anti").height
                        test_only = te_uniq.join(tr_uniq, on=c, how="anti").height
                        drift_rows.append((c, card_train, val_only, test_only))

                    # Log all drift stats compactly; WARN on anything non-trivial.
                    if drift_rows:
                        drift_rows.sort(key=lambda x: -x[2])  # sort by val_only desc
                        drift_summary = ", ".join(
                            f"{c}:val_only={v},test_only={t}"
                            for c, _, v, t in drift_rows if v > 0 or t > 0
                        ) or "(none)"
                        logger.info("  Category drift (val/test values missing from train): %s", drift_summary)

                        # WARN for anything where val_only is a non-trivial
                        # fraction of train cardinality -- suspect for XGB
                        # val-DMatrix native crash.
                        #
                        # IMPORTANT: auto-healing decisions below rely on
                        # ``v_only`` (val vs train) ONLY. ``t_only`` is
                        # reported above for operator visibility but is NOT
                        # used to choose an action -- using TEST to inform
                        # any preprocessing decision leaks test information
                        # into training. The suggested-action heuristics
                        # therefore look exclusively at train-vs-val drift
                        # and at train-side cardinality.
                        for c, card_tr, v_only, t_only in drift_rows:
                            if v_only == 0 and t_only == 0:
                                continue
                            v_frac = v_only / max(card_tr, 1)
                            if v_only >= 5 or v_frac >= 0.05:
                                # Pick a concrete healing suggestion based on
                                # train-side cardinality only. Thresholds are
                                # conservative and documented so operators can
                                # reproduce or override them deliberately.
                                if card_tr >= 1000:
                                    _healing = (
                                        f"        suggested actions (pick one):\n"
                                        f"          a) hash-bucket via FeatureHasher / target-encoding "
                                        f"(card {card_tr:_} >= 1 000 -> model will memorize train-only "
                                        f"values and generalize poorly on val/test);\n"
                                        f"          b) drop '{c}' from cat_features and keep only the "
                                        f"top-K most frequent (K=100-300) as one-hot, route the rest "
                                        f"into an '__OTHER__' bucket;\n"
                                        f"          c) drop '{c}' entirely if it's an identifier or "
                                        f"free-text field -- promote to text_features via use_text_features=True "
                                        f"so CatBoost handles it natively and other backends ignore it."
                                    )
                                elif card_tr >= 100:
                                    _healing = (
                                        f"        suggested actions (pick one):\n"
                                        f"          a) target-encoding (CatBoostEncoder) to collapse "
                                        f"{card_tr:_} levels into a continuous feature;\n"
                                        f"          b) keep top-K by train frequency, bucket the rest "
                                        f"into '__OTHER__' before fit (K~=30-80)."
                                    )
                                else:
                                    _healing = (
                                        f"        suggested actions (pick one):\n"
                                        f"          a) add an explicit '__UNSEEN__' bucket in the "
                                        f"Enum domain so val values absent from train resolve to a "
                                        f"known category instead of raising;\n"
                                        f"          b) widen the training window (temporal split) so "
                                        f"val_only categories are observed at fit time."
                                    )
                                logger.warning(
                                    f"  Category drift suspect: {c} -- val has {v_only} categories "
                                    f"({v_frac:.1%} of train card {card_tr:_}) that train never saw. "
                                    f"XGB/CB may crash when constructing val DMatrix with ref=train.\n"
                                    f"{_healing}"
                                )
            except Exception as _e:
                logger.warning(f"  Failed to compute categorical cardinality/drift: {_e}")

    metadata["text_features"] = text_features
    metadata["embedding_features"] = embedding_features

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
    # Actual training
    # -----------------------------------------------------------------------------------------------------------------------------------------------------

    if verbose:
        logger.info("Zero-copy conversion to pandas...")

    # Use pre-pipeline Polars originals for models that support native Polars input.
    # Post-pipeline DFs may have string/categorical columns converted to float by
    # polars-ds, losing dtype info needed by CatBoost and HGB.
    train_df_polars = train_df_polars_pre
    val_df_polars = val_df_polars_pre
    test_df_polars = test_df_polars_pre

    # Cache pandas versions for select_target (zero-copy Arrow-backed view for Polars).
    # Skip the conversion entirely when every model supports Polars natively -- the Polars
    # fastpath in process_model substitutes Polars DFs back anyway, so pandas views would
    # be unused. Saves ~1-2s on CB-only runs on small-to-medium DFs.
    # sklearn 1.4+ accepts Polars DataFrames as input (verified on 1.7.2 with
    # IsolationForest, LocalOutlierFactor novelty=True, and Pipeline wrappers);
    # _apply_outlier_detection_global's boolean-mask filter handles both pandas and
    # Polars via _filter_df_by_mask. So we can skip the conversion regardless of
    # outlier_detector presence.
    # Guard: recurrent models use fit() signatures that predate Polars support
    # (core.py passes train_df_pd as `features_train=` / `val_features=` / `features=`).
    # Force pandas conversion if recurrent_models is non-empty.
    # RFECV goes through sklearn.feature_selection.RFECV, which calls estimator.fit(X)
    # where X is indexed via integer positional slicing internally -- pandas-only path.
    # Force pandas conversion whenever any RFECV variant is requested.
    _has_rfecv = bool(rfecv_models)
    # Fix 1 (2026-04-21): when the ONLY blockers for the Polars fastpath are
    # non-native strategies (LGB, sklearn, linear), defer upfront conversion
    # and let the per-strategy lazy conversion at the non-polars-fastpath
    # branch of the training loop do it just-in-time. recurrent_models and
    # rfecv still trigger upfront conversion -- those paths use pandas-only
    # internals that predate Polars support. Savings on the user's 2026-04-21
    # run: 661 s polars->pandas + 70 GB RAM held across the 6-hour CB fit.
    _has_non_native_mlframe_strategy = was_polars_input and not all_models_polars_native
    can_skip_pandas_conv = (
        was_polars_input
        and not recurrent_models and not _has_rfecv
        and (all_models_polars_native or _has_non_native_mlframe_strategy)
    )

    # Pre-conversion size capture (Fix 3B): polars .estimated_size() is O(cols),
    # microseconds. Computing it now -- BEFORE any pandas conversion -- avoids the
    # pathological pandas memory_usage(deep=True) scan downstream in
    # configure_training_params (3 min on a 75 GB frame with millions of unique
    # object-column strings). Only used if the frame is polars here; when it's
    # already pandas (unusual entry), the downstream fallback stays on
    # get_df_memory_consumption(deep=False).
    train_df_size_bytes_cached: Optional[float] = None
    val_df_size_bytes_cached: Optional[float] = None
    if was_polars_input:
        try:
            if isinstance(train_df, pl.DataFrame):
                train_df_size_bytes_cached = float(train_df.estimated_size())
            if val_df is not None and isinstance(val_df, pl.DataFrame):
                val_df_size_bytes_cached = float(val_df.estimated_size())
        except Exception:
            # Any failure here is non-fatal -- downstream get_df_memory_consumption
            # fallback remains correct (just slower).
            train_df_size_bytes_cached = None
            val_df_size_bytes_cached = None

    if can_skip_pandas_conv:
        train_df_pd, val_df_pd, test_df_pd = train_df, val_df, test_df
        if verbose:
            if all_models_polars_native:
                logger.info("  Skipped pandas conversion -- all models are Polars-native")
            else:
                non_native = [
                    m for m, s in zip(mlframe_models or [], _strategies_for_polars_check)
                    if not s.supports_polars
                ]
                logger.info(
                    "  Deferred pandas conversion -- Polars-native models run on the fastpath; "
                    "non-native %s will convert lazily at their strategy branch.",
                    non_native,
                )
    else:
        # Diagnostic: on large high-cardinality frames the polars->pandas
        # conversion costs minutes (per-column dict rebuild for every split),
        # so if users assume fastpath is active but see the conversion fire
        # anyway they need to know *which condition* blocked the skip. Log
        # the exact reasons instead of the single non-skip signal.
        if verbose:
            reasons = []
            if not was_polars_input:
                reasons.append("input is not a Polars DataFrame")
            if not all_models_polars_native:
                non_native = [
                    m for m, s in zip(mlframe_models or [], _strategies_for_polars_check)
                    if not s.supports_polars
                ]
                reasons.append(
                    f"non-Polars-native models requested: {non_native}"
                    if non_native
                    else "all_models_polars_native=False (no strategies)"
                )
            if recurrent_models:
                reasons.append(f"recurrent_models={recurrent_models}")
            if _has_rfecv:
                reasons.append(f"rfecv_models={rfecv_models}")
            logger.info(
                "  polars->pandas conversion needed because: %s",
                "; ".join(reasons) or "unknown",
            )
        train_df_pd, val_df_pd, test_df_pd = _convert_dfs_to_pandas(train_df, val_df, test_df, verbose=verbose)

    # Ensure categorical features have pandas category dtype for CatBoost.
    # After the 2026-04-17 optimization, get_pandas_view_of_polars_df already
    # preserves Polars Categorical as pd.Categorical (int32 dict rebuild), so
    # this is usually a no-op. The call is kept for safety when pandas frames
    # come in with plain string columns that still need the category cast.
    #
    # Gate (item 11 of 2026-04-18 log triage): when the Polars fastpath is
    # active (can_skip_pandas_conv=True) the models receive the Polars DFs
    # directly -- running prepare_df_for_catboost on the *pandas views* that
    # are built for select_target only is pointless work that shows up as
    # a 2+ minute blocking step in PHASE 4 on 1Mx100 frames. Skip it in
    # that case.
    if cat_features and not can_skip_pandas_conv:
        if verbose:
            logger.info("Preparing %d categorical features for CatBoost: %s", len(cat_features), cat_features)
        for df_pd in [train_df_pd, val_df_pd, test_df_pd]:
            if df_pd is not None:
                prepare_df_for_catboost(df_pd, cat_features)
    elif cat_features and can_skip_pandas_conv and verbose:
        logger.info(
            "Skipping pandas-side CatBoost prep for %d categorical "
            "features -- Polars fastpath receives the DFs natively.",
            len(cat_features),
        )

    # B2: Release post-pipeline Polars DFs after pandas conversion.
    # Arrow-backed pandas views hold their own Arrow buffer references,
    # so the Polars objects are no longer needed. Saves ~100GB peak memory.
    if was_polars_input and needs_polars_pre_clone:
        # Only release if we cloned (otherwise train_df IS train_df_polars_pre)
        train_df = val_df = test_df = None
        baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="post-pipeline Polars release")
        if verbose:
            logger.info("  Released post-pipeline Polars DFs (pandas views retained)")

    if verbose:
        log_ram_usage()

    # ==================================================================================
    # 4.5 OUTLIER DETECTION (once, before model training loops)
    # ==================================================================================

    # Pass per-target arrays into OD so the class-balance pre-check can
    # detect when OD would eliminate the entire minority class for any
    # classification target. Flatten the {target_type: {name: values}}
    # dict to {name: values} for the inner check (target_name uniqueness
    # is enforced upstream by the FTE).
    _targets_flat_for_classbalance = {}
    for _tt, _named in target_by_type.items():
        if isinstance(_named, dict):
            for _tn, _tv in _named.items():
                _targets_flat_for_classbalance[f"{_tt}/{_tn}"] = _tv
    (filtered_train_df, filtered_val_df, filtered_train_idx, filtered_val_idx, train_od_idx, val_od_idx) = _apply_outlier_detection_global(
        train_df=train_df_pd,
        val_df=val_df_pd,
        train_idx=train_idx,
        val_idx=val_idx,
        outlier_detector=outlier_detector,
        od_val_set=od_val_set,
        verbose=verbose,
        baseline_rss_mb=baseline_rss_mb,
        df_size_mb=df_size_mb,
        targets_for_classbalance=_targets_flat_for_classbalance or None,
    )

    # Single global OD result (not per-target)
    outlier_detection_result = {
        "train_od_idx": train_od_idx,
        "val_od_idx": val_od_idx,
    }

    # Surface OD row-reduction evidence in returned metadata so callers/tests can assert
    # that OD actually ran without grepping logs. Only populated when an OD was applied.
    if outlier_detector is not None:
        n_train_pre_od = len(train_df_pd) if train_df_pd is not None else None
        n_val_pre_od = len(val_df_pd) if val_df_pd is not None else None
        n_train_post_od = int(train_od_idx.sum()) if train_od_idx is not None else n_train_pre_od
        n_val_post_od = int(val_od_idx.sum()) if val_od_idx is not None else n_val_pre_od
        n_train_dropped = (n_train_pre_od - n_train_post_od) if n_train_pre_od is not None else 0
        n_val_dropped = (n_val_pre_od - n_val_post_od) if (n_val_pre_od is not None and val_od_idx is not None) else 0
        metadata["outlier_detection"] = {
            "applied": True,
            "n_outliers_dropped_train": int(n_train_dropped),
            "n_outliers_dropped_val": int(n_val_dropped),
            "train_size_after_od": int(n_train_post_od) if n_train_post_od is not None else None,
            "val_size_after_od": int(n_val_post_od) if n_val_post_od is not None else None,
        }
    else:
        metadata["outlier_detection"] = {"applied": False}

    # Keep polars fastpath DFs in sync with pandas-filtered copies so that the
    # Polars-native training path operates on OD-filtered rows matching the
    # OD-filtered targets. Without this the downstream training call feeds the
    # unfiltered Polars DF but the OD-filtered target -> length mismatch.
    if train_od_idx is not None and train_df_polars is not None:
        train_df_polars = train_df_polars.filter(pl.Series(train_od_idx))
    if val_od_idx is not None and val_df_polars is not None:
        val_df_polars = val_df_polars.filter(pl.Series(val_od_idx))

    # ==================================================================================
    # 4.6 COMPOSITE-TARGET DISCOVERY (opt-in; default OFF)
    # ==================================================================================
    #
    # Run AFTER outlier detection so transform parameters (alpha/beta for
    # linear_residual, MAD for logratio, ...) are fitted on the same rows
    # the per-target training loop will use; run BEFORE the temporal-audit
    # batch (line ~3310) and BEFORE the per-target loop (line ~3500) so
    # the new composite-target columns flow through both. Each discovered
    # spec contributes ONE entry to target_by_type[regression] keyed by
    # ``f"{target}__{transform}__{base}"``. Specs are stored on
    # ``metadata["composite_target_specs"]`` for downstream inversion at
    # predict time (post-PR4 wiring).
    #
    # Defensive: caller may have wired the same FTE into multiple suite
    # calls; we MUST NOT mutate the FTE-returned target_by_type
    # in-place. Shallow copy the outer dict and the per-type inner dict
    # before adding entries.
    metadata["composite_target_specs"] = {}
    metadata["composite_target_failures"] = {}
    metadata["composite_target_filter_drops"] = {}
    if composite_target_discovery_config.enabled:
        # Snapshot env once per discovery run; persists on metadata
        # so a v2-loaded suite can detect numpy / sklearn / lgb / xgb
        # version drift between save time and load time.
        from ..composite import env_signature as _env_sig, detect_gpu_in_use as _detect_gpu
        metadata["composite_target_env_signature"] = _env_sig()
        # R10c bug #6 fix: GPU warning is now DEFERRED until after
        # discovery completes, so it only fires when ``K > 0`` (i.e.
        # composite specs were actually shipped). The pre-fix path
        # printed the warning unconditionally at the start of
        # discovery, causing false alarms in production runs where
        # discovery later returned 0 specs (no K-fold amplification
        # to amplify). The GPU family list is captured here for use
        # in the post-discovery emit. The actual ``logger.warning``
        # call lives further down, after the per-target discovery
        # loop, gated on ``len(kept_spec_total) > 0``.
        _gpu_families = _detect_gpu(mlframe_models or [])
        # Initialise the "total composite specs shipped" counter so
        # we can decide at end-of-loop whether the warning is warranted.
        _kept_spec_total: int = 0
    # Skip target types that aren't in scope for composite-target
    # discovery. We mark them explicitly on the failures map so
    # callers can tell "we considered this and chose not to" apart
    # from "we never looked".
    for _tt_skip, _named_skip in target_by_type.items():
        if not isinstance(_named_skip, dict):
            continue
        if _tt_skip == TargetTypes.REGRESSION:
            continue
        # Per-target skip reasons keyed by target_type stringified key.
        reason = None
        if _tt_skip == TargetTypes.LEARNING_TO_RANK:
            reason = "ltr_unsupported_pairwise_breaks_with_residual"
        elif _tt_skip == TargetTypes.MULTICLASS_CLASSIFICATION:
            reason = "multiclass_unsupported_no_residual_semantics"
        elif _tt_skip == TargetTypes.MULTILABEL_CLASSIFICATION:
            reason = "multilabel_classification_unsupported"
        elif _tt_skip == TargetTypes.QUANTILE_REGRESSION:
            reason = "quantile_regression_unsupported_per_quantile_inverse_undefined"
        elif _tt_skip == TargetTypes.BINARY_CLASSIFICATION:
            reason = "binary_classification_unsupported_init_score_logit_offset"
        if reason is not None:
            for _tn_skip in _named_skip:
                metadata["composite_target_failures"].setdefault(
                    str(_tt_skip), {})[_tn_skip] = [{
                        "name": _tn_skip, "kept": False, "rejected": True,
                        "reason": reason,
                    }]
    if (composite_target_discovery_config.enabled
            and TargetTypes.REGRESSION in target_by_type):
        target_by_type = {
            tt: dict(named) if isinstance(named, dict) else named
            for tt, named in target_by_type.items()
        }
        # R3.18: multilabel regression -- 2-D target arrays of shape
        # (n_rows, n_outputs). Expand each into n_outputs separate
        # 1-D regression targets named ``{target}_out{j}`` so the
        # rest of the pipeline (composite discovery + per-target
        # training) treats them independently. Caller can opt out
        # via ``multilabel_strategy="skip"`` to fall back to the
        # legacy "skip with metadata note" behaviour.
        _ml_strategy = str(getattr(
            composite_target_discovery_config,
            "multilabel_strategy", "per_target",
        ))
        if _ml_strategy == "per_target":
            _expanded = dict(target_by_type[TargetTypes.REGRESSION])
            _ml_expanded_map: Dict[str, List[str]] = {}
            for _tn, _tv in list(target_by_type[TargetTypes.REGRESSION].items()):
                _arr = np.asarray(_tv)
                if _arr.ndim == 2 and _arr.shape[1] >= 1:
                    sub_names = []
                    for _j in range(_arr.shape[1]):
                        _sub_name = f"{_tn}_out{_j}"
                        _expanded[_sub_name] = _arr[:, _j]
                        sub_names.append(_sub_name)
                    # Drop the original 2-D entry; sub-targets now
                    # cover it.
                    _expanded.pop(_tn, None)
                    _ml_expanded_map[_tn] = sub_names
                    logger.info(
                        "[CompositeTargetDiscovery] R3.18: multilabel "
                        "target '%s' (shape=%s) expanded into %d 1-D "
                        "sub-targets: %s",
                        _tn, _arr.shape, _arr.shape[1], sub_names,
                    )
            target_by_type[TargetTypes.REGRESSION] = _expanded
            if _ml_expanded_map:
                metadata.setdefault("multilabel_target_expansion", {})[
                    str(TargetTypes.REGRESSION)
                ] = _ml_expanded_map
        # Get the "filtered" feature columns: training feature columns at
        # the OD-filtered DataFrame. Discovery reads from filtered_train_df
        # to keep one row-set semantics.
        try:
            _disc_feature_cols = list(filtered_train_df.columns)
        except Exception:
            _disc_feature_cols = list(train_df_pd.columns)
        # Build the train_idx into the filtered frame: discovery's
        # train_idx is "indices INTO ``df``". filtered_train_df is
        # already row-aligned to filtered_train_idx, so we hand
        # discovery a contiguous range and use the filtered frame
        # directly as ``df``.
        _disc_train_idx = np.arange(len(filtered_train_df))
        # Auto-skip per-target when BaselineDiagnostics recommended
        # "unlikely_to_help" AND the user opted in to the auto-skip.
        # The diagnostic recommendation lives at:
        # ``metadata["baseline_diagnostics"][target_type][target_name]
        #   ["composite_recommendation"]``.
        # Note: BaselineDiagnostics runs INSIDE the per-target loop
        # which is BELOW our position here, so at this point the
        # metadata key does NOT yet exist. We can only auto-skip
        # for targets where the user pre-populated the diagnostic
        # (rare) OR we'd need to lift the BaselineDiagnostics call
        # above this block. For PR-time minimal scope we evaluate
        # the recommendation lazily: discovery runs the full
        # screening pipeline, but if the BaselineDiagnostics output
        # is already available (e.g. caller invoked the suite twice
        # and metadata persisted) we honour the recommendation.
        _auto_skip = bool(getattr(
            composite_target_discovery_config,
            "auto_skip_on_baseline_optimal", False,
        ))
        _existing_diags = metadata.get("baseline_diagnostics", {})
        for _tt_disc, _named_disc in list(target_by_type.items()):
            if _tt_disc != TargetTypes.REGRESSION:
                continue
            for _tname_disc, _tvals_disc in list(_named_disc.items()):
                _y_arr = np.asarray(_tvals_disc)
                if _y_arr.ndim != 1:
                    metadata["composite_target_failures"].setdefault(
                        str(_tt_disc), {})[_tname_disc] = [{
                            "name": _tname_disc, "kept": False, "rejected": True,
                            "reason": "multilabel target unsupported (R3.18 future PR)",
                        }]
                    continue  # multilabel: skip with explicit metadata
                # Auto-skip on baseline-optimal recommendation AND/OR
                # ablation hint for auto-base ranking. The per-target
                # BaselineDiagnostics runs INSIDE the per-target loop
                # further down, so on the first pass the metadata key
                # isn't populated yet. Run a lightweight inline
                # diagnostic here when EITHER signal is enabled. The
                # result is cached in metadata so the per-target loop
                # reuses it (saves the duplicate ~30-60s ablation cost).
                _use_hint = bool(getattr(
                    composite_target_discovery_config,
                    "use_baseline_diagnostics_hint", False,
                ))
                _diag = None
                if _auto_skip or _use_hint:
                    _diag = (
                        _existing_diags.get(str(_tt_disc), {}).get(_tname_disc)
                    )
                    if _diag is None:
                        try:
                            _bd_inline = BaselineDiagnostics(
                                baseline_diagnostics_config
                            )
                            _y_train_for_diag = (
                                _y_arr[filtered_train_idx]
                                if filtered_train_idx is not None else _y_arr
                            )
                            _diag_report = _bd_inline.fit_and_report(
                                train_df=filtered_train_df,
                                train_target=_y_train_for_diag,
                                feature_cols=list(filtered_train_df.columns),
                                target_type=str(_tt_disc),
                                target_name=_tname_disc,
                                cat_features=cat_features,
                            )
                            _diag = _diag_report.to_dict()
                            # Cache for per-target loop reuse.
                            metadata.setdefault("baseline_diagnostics", {}) \
                                .setdefault(str(_tt_disc), {})[_tname_disc] = _diag
                        except Exception as _bd_err:
                            logger.info(
                                "[CompositeTargetDiscovery] inline "
                                "diagnostic precompute failed for '%s': %s; "
                                "discovery proceeds without auto-skip / hint.",
                                _tname_disc, _bd_err,
                            )
                            _diag = None
                if _auto_skip:
                    if (_diag is not None
                            and _diag.get("composite_recommendation") == "unlikely_to_help"):
                        logger.info(
                            "[CompositeTargetDiscovery] auto-skip target='%s': "
                            "BaselineDiagnostics recommendation='unlikely_to_help' "
                            "(reason: %s).",
                            _tname_disc,
                            _diag.get("composite_recommendation_reason", "")[:120],
                        )
                        metadata["composite_target_failures"].setdefault(
                            str(_tt_disc), {})[_tname_disc] = [{
                                "name": _tname_disc, "kept": False, "rejected": True,
                                "reason": "auto_skip_on_baseline_optimal=True + "
                                          "diagnostic='unlikely_to_help'",
                            }]
                        continue
                # Subset y to filtered_train_df rows. y_arr is row-aligned
                # to the FULL frame; filtered_train_df is row-aligned to
                # filtered_train_idx (post split + OD).
                _y_train_aligned = _y_arr[filtered_train_idx]
                if len(_y_train_aligned) != len(filtered_train_df):
                    logger.warning(
                        "[CompositeTargetDiscovery] target='%s' row-align mismatch "
                        "(y[%d] vs filtered_train_df[%d]); skipping discovery.",
                        _tname_disc, len(_y_train_aligned), len(filtered_train_df),
                    )
                    continue
                # Build a tiny working frame: filtered_train_df + target column.
                if isinstance(filtered_train_df, pd.DataFrame):
                    _disc_df = filtered_train_df.copy(deep=False)
                    _disc_df[_tname_disc] = _y_train_aligned
                else:
                    _disc_df = filtered_train_df.with_columns(
                        pl.Series(_tname_disc, _y_train_aligned)
                    )
                # If hint is enabled and BD ran, derive a per-target
                # config copy with ``dominant_features_hint`` populated
                # from ablation top-K. ``_diag`` may be None when hint
                # was disabled or BD failed -- fall through to the
                # caller-provided config without modification.
                _disc_cfg = composite_target_discovery_config
                if _use_hint and _diag is not None:
                    _hint_top_k = max(1, int(getattr(
                        composite_target_discovery_config,
                        "baseline_diagnostics_hint_top_k", 3,
                    )))
                    _ablation = _diag.get("ablation", []) or []
                    _ablation_sorted = sorted(
                        _ablation,
                        key=lambda e: -float(e.get("delta_pct", 0.0)),
                    )
                    _hint_cols = [
                        e["feature"] for e in _ablation_sorted[:_hint_top_k]
                        if e.get("feature")
                    ]
                    # R10c bug #5: also pass per-hint ablation strength
                    # (delta_pct) so the discovery can adapt the cap
                    # to hint strength. Strong hints (>= threshold)
                    # take all top_k slots; weak hints fall back to
                    # the half-slot cap.
                    _hint_strengths = [
                        float(e.get("delta_pct", 0.0))
                        for e in _ablation_sorted[:_hint_top_k]
                        if e.get("feature")
                    ]
                    if _hint_cols:
                        try:
                            # Pydantic v2: model_copy is the preferred
                            # cloning API. Falls back silently if
                            # the user-supplied config doesn't expose
                            # the method (defensive belt-and-braces).
                            _disc_cfg = composite_target_discovery_config.model_copy(
                                update={"dominant_features_hint": _hint_cols},
                            )
                            logger.info(
                                "[CompositeTargetDiscovery] target='%s' hint "
                                "from BaselineDiagnostics ablation top-%d: "
                                "%s (max delta%%=%.1f)",
                                _tname_disc, len(_hint_cols), _hint_cols,
                                max(_hint_strengths) if _hint_strengths else 0.0,
                            )
                        except Exception as _clone_err:
                            logger.info(
                                "[CompositeTargetDiscovery] hint clone failed "
                                "for target='%s' (%s); proceeding with MI-only.",
                                _tname_disc, _clone_err,
                            )
                else:
                    _hint_strengths = None
                try:
                    _disc_instance = CompositeTargetDiscovery(_disc_cfg)
                    # Plumb per-hint strength into the discovery instance
                    # so ``_auto_base`` can decide whether to apply the
                    # full hint (strong) or cap at half-slots (weak).
                    if _use_hint and _diag is not None and _hint_strengths:
                        _disc_instance._hint_strengths_pct = _hint_strengths
                    _disc = _disc_instance.fit(
                        df=_disc_df,
                        target_col=_tname_disc,
                        feature_cols=_disc_feature_cols,
                        train_idx=_disc_train_idx,
                    )
                except Exception as _disc_err:
                    logger.warning(
                        "[CompositeTargetDiscovery] fit failed for target='%s': %s. "
                        "Per-target training continues without composite expansion.",
                        _tname_disc, _disc_err,
                    )
                    continue
                # Persist specs + report.
                metadata["composite_target_specs"].setdefault(str(_tt_disc), {})
                metadata["composite_target_specs"][str(_tt_disc)][_tname_disc] = (
                    _disc.export_specs()
                )
                metadata["composite_target_failures"].setdefault(str(_tt_disc), {})
                metadata["composite_target_failures"][str(_tt_disc)][_tname_disc] = [
                    r for r in _disc.report() if r.get("rejected")
                ]
                # Pre-MI filter drops (forbidden pattern, non-numeric,
                # constant, corr-threshold) so users can audit cases
                # where an obvious base candidate "vanished" before
                # ranking. The corr-threshold filter is the most common
                # culprit on autoregressive lag features.
                metadata.setdefault("composite_target_filter_drops", {})
                metadata["composite_target_filter_drops"].setdefault(
                    str(_tt_disc), {})
                metadata["composite_target_filter_drops"][str(_tt_disc)][
                    _tname_disc] = _disc.filter_drops()
                # Apply each discovered spec's transform to the FULL row
                # index space (train + val + test). Discovery fitted
                # transform params on filtered_train_idx ONLY (leakage
                # discipline preserved); we now apply those frozen
                # params to all rows so the per-target training loop
                # has T values for val / test rows when it slices by
                # filtered_train_idx / filtered_val_idx / test_idx.
                # NaN rows (domain violations on val / test) get
                # imputed with median(T_train) so the trainer's NaN
                # guard at trainer.py:~703 doesn't trip; the substitute
                # is biased but never propagates to predictions because
                # the inverse layer (PR5) fills via y_train_median for
                # those rows.
                from ..composite import get_transform as _get_transform_local
                for _spec in _disc.specs_:
                    _transform = _get_transform_local(_spec.transform_name)
                    _base_full = _build_full_column_from_splits(
                        _spec.base_column,
                        train_df_pd, val_df_pd, test_df_pd,
                        train_idx, val_idx, test_idx,
                        n_total=_y_arr.shape[0],
                    )
                    _valid = _transform.domain_check(_y_arr, _base_full)
                    _ct_t_full = np.full(_y_arr.shape[0], np.nan, dtype=np.float64)
                    if _valid.any():
                        _ct_t_full[_valid] = _transform.forward(
                            _y_arr[_valid], _base_full[_valid], _spec.fitted_params,
                        )
                    # Impute NaN rows (domain violations) with the
                    # train-fitted T median so trainer doesn't refuse.
                    if not np.all(np.isfinite(_ct_t_full)):
                        _t_train_for_median = _ct_t_full[filtered_train_idx]
                        _t_train_for_median = _t_train_for_median[
                            np.isfinite(_t_train_for_median)
                        ]
                        if _t_train_for_median.size > 0:
                            _ct_t_full[~np.isfinite(_ct_t_full)] = float(
                                np.median(_t_train_for_median)
                            )
                    target_by_type[_tt_disc][_spec.name] = _ct_t_full
                    logger.info(
                        "[CompositeTargetDiscovery] added composite target '%s' "
                        "to target_by_type[%s].", _spec.name, _tt_disc,
                    )
        # If any specs were discovered, log a one-liner summary so
        # operators see the impact at suite level even without
        # reading metadata.
        n_specs_total = sum(
            len(v) for tt_specs in metadata["composite_target_specs"].values()
            for v in tt_specs.values()
        )
        if n_specs_total > 0:
            logger.info(
                "[CompositeTargetDiscovery] %d composite target(s) added to "
                "target_by_type. They will be trained alongside raw targets in the "
                "per-target loop.", n_specs_total,
            )
            # R10c bug #6 fix: emit the GPU non-determinism warning
            # ONLY when composite specs actually shipped (K > 0).
            if _gpu_families:
                logger.warning(
                    "[CompositeTargetDiscovery] composite mode + GPU "
                    "training detected (%s) AND %d composite spec(s) "
                    "shipped. GPU non-determinism is amplified by the "
                    "K=%d extra fits; ensemble weights may drift across "
                    "runs even with random_state fixed. Set "
                    "deterministic=True / single_precision_histogram=True "
                    "/ force_row_wise=True on the inner estimators if "
                    "reproducibility matters.",
                    ", ".join(_gpu_families), n_specs_total, n_specs_total,
                )

    # ------------------------------------------------------------------
    # Round-11c fill: Polars Categorical cat_features with null values
    # trip CatBoost 1.2.10's fused-cpdef dispatcher
    # (``TypeError: No matching signature found`` -- see
    # ``bench_polars_cb_nullfrac.py`` and the round-11 CHANGELOG entry).
    # Fill once here on the base Polars DFs so every polars-native
    # strategy downstream (CB, XGB, HGB, ...) gets the same pre-filled
    # frame -- ``tier_polars`` views, ``prepared_train`` in the model
    # loop, predict-time conversions, all uniformly null-free on cats.
    # Keeps the fastpath alive for every polars-capable model.
    #
    # Single-pass null detection via ``df.null_count()`` -- computes
    # per-column counts in one scan instead of one query per column.
    # Fill expr list is built once and reused across train/val/test
    # so the category sentinel codes are consistent across splits.
    if train_df_polars is not None:
        from mlframe.training.trainer import (
            _polars_nullable_categorical_cols,
            _polars_fill_null_in_categorical,
        )
        # Detect nullable cats in train AND val AND test -- union the
        # sets. Previously we only inspected ``train_df_polars``, which
        # missed a class of bug: a column with 0 nulls in train but 100+
        # in val (common on time-ordered splits where new records
        # introduce new null-paradigms) was NOT filled, leaving val
        # with nulls in a Polars Categorical -> CB/XGB native crash at
        # val-DMatrix construction. Union ensures fill applies if ANY
        # split has nulls; ``__MISSING__`` sentinel then consistently
        # lands in train/val/test's category dicts.
        train_null_cats = set(_polars_nullable_categorical_cols(
            train_df_polars, cat_features=cat_features,
        ))
        val_null_cats = set(_polars_nullable_categorical_cols(
            val_df_polars, cat_features=cat_features,
        )) if val_df_polars is not None else set()
        test_null_cats = set(_polars_nullable_categorical_cols(
            test_df_polars, cat_features=cat_features,
        )) if test_df_polars is not None else set()
        nullable_cats = sorted(train_null_cats | val_null_cats | test_null_cats)
        if nullable_cats:
            # Spotlight columns where val/test have nulls but train doesn't --
            # that's the exact scenario that used to escape the pre-fill.
            val_only = sorted((val_null_cats | test_null_cats) - train_null_cats)
            if verbose:
                logger.info(
                    "  Pre-fit fill_null('__MISSING__') on %d nullable Polars "
                    "Categorical cat_feature(s) [union train/val/test]: %s. "
                    "Keeps CB 1.2.x's Polars fastpath alive (avoids the "
                    "~15-min pandas-path detour) and gives XGB/HGB the "
                    "same pre-filled frame.",
                    len(nullable_cats), nullable_cats,
                )
                if val_only:
                    logger.warning(
                        "  val/test introduced nulls in %d cat_feature(s) that "
                        "train never had: %s. Without pre-fill these would "
                        "slip into the model's val DMatrix as raw nulls and "
                        "crash CB/XGB native layer.",
                        len(val_only), val_only,
                    )
            train_df_polars = _polars_fill_null_in_categorical(train_df_polars, nullable_cats)
            if val_df_polars is not None:
                val_df_polars = _polars_fill_null_in_categorical(val_df_polars, nullable_cats)
            if test_df_polars is not None:
                test_df_polars = _polars_fill_null_in_categorical(test_df_polars, nullable_cats)

    # -----------------------------------------------------------------
    # Round 18: align Polars Categorical dicts across train/val/test.
    #
    # Empirically prevents a silent process kill on Windows when XGB
    # 3.x constructs val IterativeDMatrix with ref=train on large
    # frames (7.3M+ rows, 15+ cat features) -- observed 2026-04-20 on
    # prod_jobsdetails. Mechanism is not fully understood but the
    # leading theory: ``pl.Categorical`` assigns physical codes
    # per-Series (order-of-first-occurrence), so the same string can
    # have different physical codes in train vs val vs test. XGB's
    # native layer at large scale appears to treat val's physical
    # codes as indices into train's bin structure without re-reading
    # the Arrow dict, corrupting memory. ``pl.Enum(list)`` enforces a
    # shared dict by construction so physical codes are consistent
    # across splits.
    #
    # Small-scale probe (``D:/Temp/xgb_unseen_cat_probe.py``) did NOT
    # reproduce the crash on 2000 rows x 1 cat feature, even with
    # deliberate val/train dict mismatch. Prod repro confirmed the
    # fix works end-to-end (MakeCuts time dropped 50x after
    # alignment: 0.9s -> 18ms, consistent with XGB taking a different
    # code path).
    #
    # Opt out via ``behavior_config.align_polars_categorical_dicts=False``.
    # Skipped for columns with cardinality > 50_000 (text-sized --
    # expensive to align; typically already promoted to text_features).
    if (train_df_polars is not None and cat_features
            and behavior_config.align_polars_categorical_dicts):
        _DICT_ALIGN_SKIP_CARD = 50_000
        aligned_cols: list = []
        skipped_cols: list = []
        for col in cat_features:
            if col not in train_df_polars.columns:
                continue
            dt = train_df_polars.schema[col]
            # Only align Categorical/Enum (string-like) columns.
            is_cat_like = (
                dt == pl.Categorical
                or (hasattr(pl, "Enum") and isinstance(dt, pl.Enum))
            )
            if not is_cat_like:
                continue
            try:
                # Collect unique values from each split (drop nulls --
                # fill_null above already handled them; nulls mustn't
                # enter an Enum's category list).
                #
                # 2026-04-22 future-leakage fix: ONLY train + val
                # contribute to the Enum vocabulary. test_df categories
                # MUST NOT leak back -- the whole purpose of a held-out
                # test is to represent "truly unseen" data, and seeding
                # the model's categorical vocabulary with test values
                # leaks future information into training-time
                # preprocessing (the Enum object is part of the pipeline
                # the model commits to at fit). Test-only categories
                # are cast with ``strict=False`` -> OOV values land as
                # nulls, which XGB / CB treat identically to any other
                # missing value (consistent with production-time
                # behaviour: at inference we have no mechanism to
                # retroactively grow the Enum either).
                tr_u = train_df_polars.select(pl.col(col).drop_nulls().unique())[col]
                v_u  = val_df_polars.select(pl.col(col).drop_nulls().unique())[col] if val_df_polars is not None else None
                # Union + stable-sorted for reproducibility.
                union = set(tr_u.to_list())
                if v_u is not None:
                    union |= set(v_u.to_list())
                if len(union) > _DICT_ALIGN_SKIP_CARD:
                    skipped_cols.append((col, len(union)))
                    continue
                union_sorted = sorted(union)
                enum_dt = pl.Enum(union_sorted)
                # train + val: known to be fully within the vocabulary,
                # safe to cast strictly (default).
                train_df_polars = train_df_polars.with_columns(pl.col(col).cast(enum_dt))
                if val_df_polars is not None:
                    val_df_polars = val_df_polars.with_columns(pl.col(col).cast(enum_dt))
                # test: may have OOV; cast with strict=False to null those
                # out rather than crashing or leaking them into the Enum.
                if test_df_polars is not None:
                    test_df_polars = test_df_polars.with_columns(
                        pl.col(col).cast(enum_dt, strict=False)
                    )
                aligned_cols.append((col, len(union_sorted)))
            except Exception as _e:
                logger.warning(
                    "  Failed to align category dict for %s: %s. "
                    "XGB/CB may crash on val-DMatrix if val has "
                    "unseen categories.",
                    col, _e,
                )

        if verbose and aligned_cols:
            aligned_summary = ", ".join(f"{c}:{n}" for c, n in aligned_cols)
            logger.info(
                "  Aligned Categorical dicts across train/val/test for %d "
                "cat_feature(s) via pl.Enum(union): %s. Prevents XGB/CB "
                "native crash on val-DMatrix construction when val has "
                "categories absent from train.",
                len(aligned_cols), aligned_summary,
            )
        if verbose and skipped_cols:
            skipped_summary = ", ".join(f"{c}:{n}" for c, n in skipped_cols)
            logger.warning(
                "  Skipped dict alignment for %d high-cardinality "
                "cat_feature(s) (union > %d): %s. These columns are "
                "still at risk of XGB/CB val-DMatrix crash.",
                len(skipped_cols), _DICT_ALIGN_SKIP_CARD, skipped_summary,
            )

    # 2026-04-23 (fuzz c0088/c0121): when ``can_skip_pandas_conv=True`` the
    # top-level ``train_df_pd / filtered_train_df / etc.`` were aliased to
    # the ORIGINAL polars frames (line ~2354) BEFORE the fill_null /
    # Enum-alignment block above. ``common_params["train_df"]`` (built by
    # ``configure_training_params`` below) picks up ``filtered_train_df``,
    # so the lazy pandas conversion at the non-polars-native strategy branch
    # (line ~3034) converts the UNFILLED frame -- nulls become NaN in the
    # pandas Categorical, CB's fallback Pool build raises
    # ``Invalid type for cat_feature ... =NaN``. Re-point the aliases to
    # the filled+aligned ``train_df_polars`` now that those treatments
    # have been applied, so every downstream consumer (select_target,
    # common_params, lazy-pandas-conversion) sees the sentinel-filled
    # values. ``train_df_polars`` already carries the OD row filter
    # (applied at line ~2479), so the row sets stay aligned.
    if can_skip_pandas_conv and train_df_polars is not None:
        train_df_pd = train_df_polars
        filtered_train_df = train_df_polars
        if val_df_polars is not None:
            val_df_pd = val_df_polars
            filtered_val_df = val_df_polars
        if test_df_polars is not None:
            test_df_pd = test_df_polars

    # 2026-04-23 (fuzz driver seed 20260430, pattern cb/hgb/lgb/xgb +
    # polars_utf8 + ncats>=1): when input is polars_utf8 and alignment
    # didn't touch the column (either skipped or align_polars_categorical_dicts=False),
    # cat_features stay as pl.Utf8 / pl.String. Non-polars-native
    # strategies (LGB, Linear) later trigger the lazy pandas conversion;
    # pl.Utf8 -> pandas ``object`` dtype (not ``category``). XGBClassifier's
    # sklearn wrapper then raises
    #     ValueError: DataFrame.dtypes for data must be int, float, bool
    #     or category. ... Invalid columns:cat_0: object
    # Cast remaining Utf8/String cat_features to pl.Categorical after
    # fill+align so the pandas conversion produces ``category`` dtype.
    # Gate on ``was_polars_input`` + non-empty ``cat_features`` so the
    # pandas-input path stays a byte-for-byte no-op.
    if was_polars_input and cat_features:
        def _cast_utf8_cats_to_categorical(df_):
            if not isinstance(df_, pl.DataFrame):
                return df_
            exprs = []
            for c in cat_features:
                if c in df_.columns and df_.schema[c] in (pl.Utf8, pl.String):
                    exprs.append(pl.col(c).cast(pl.Categorical))
            return df_.with_columns(exprs) if exprs else df_
        train_df_polars = _cast_utf8_cats_to_categorical(train_df_polars)
        val_df_polars = _cast_utf8_cats_to_categorical(val_df_polars)
        test_df_polars = _cast_utf8_cats_to_categorical(test_df_polars)
        if can_skip_pandas_conv:
            train_df_pd = train_df_polars if train_df_polars is not None else train_df_pd
            filtered_train_df = train_df_polars if train_df_polars is not None else filtered_train_df
            if val_df_polars is not None:
                val_df_pd = val_df_polars
                filtered_val_df = val_df_polars
            if test_df_polars is not None:
                test_df_pd = test_df_polars

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

    # 2026-04-26 Session 7 batch 2: precompute the temporal target audit
    # ONCE for ALL (target_type, target_name) pairs in a single polars
    # multi-aggregation pass. Per-target use inside the loop just looks
    # up the prebuilt result. This is ~5× faster than the loop-per-call
    # approach for N>1 targets on >1M-row datasets (benchmarked: 1M×5
    # targets, 6.0s loop → 1.3s batch).
    _all_target_audits: Dict[Any, Dict[str, Any]] = {}
    # 2026-04-27: auto-detect timestamp column from the
    # features_and_targets_extractor's ``ts_field``. The user almost
    # always already configured a timestamp on the FTE for splitting /
    # date-feature extraction; making them re-state it on
    # behavior_config to enable the audit was friction.
    #
    # Resolution order:
    #   1. behavior_config.target_temporal_audit_column = "<col>"  -> explicit opt-in to <col>
    #   2. behavior_config.target_temporal_audit_column = ""       -> explicit opt-out (audit disabled)
    #   3. behavior_config.target_temporal_audit_column = None     -> fall through to FTE.ts_field
    #   4. FTE.ts_field set + column present in df                 -> auto-detect (audit fires)
    #   5. neither                                                  -> audit silent
    _audit_ts_override = getattr(behavior_config, "target_temporal_audit_column", None) if behavior_config else None
    if _audit_ts_override is None:
        # ``df`` is deleted earlier in this function (Phase 2 cleanup
        # ``del df`` around line 2207). The check below tolerates a
        # NameError on df via locals() so the auto-detect doesn't blow
        # up; the actual ts source is the FTE-returned ``timestamps``
        # ndarray captured before del, with df-column lookup as a
        # secondary fallback when the original df is still in scope.
        _fte_ts = getattr(features_and_targets_extractor, "ts_field", None)
        # Prefer the FTE-returned timestamps ndarray (always present
        # after FTE.transform when ts_field is set, regardless of
        # downstream df.del). Fall back to df[col] only if df is
        # still bound and has the column.
        _src_df = locals().get("df", None)
        _ts_in_df = (
            _src_df is not None and hasattr(_src_df, "columns") and _fte_ts in _src_df.columns
        )
        if _fte_ts and (timestamps is not None or _ts_in_df):
            _audit_ts_col = _fte_ts
            logger.info(
                "target_temporal_audit: auto-detected timestamp column '%s' "
                "from features_and_targets_extractor.ts_field. To override, "
                "set behavior_config.target_temporal_audit_column='<col>'; "
                "to disable, set it to ''.",
                _audit_ts_col,
            )
        else:
            _audit_ts_col = None
    else:
        # Operator explicitly set the override (could be a column name
        # OR empty string for "disable"). Honour as-is; empty string
        # falls through the truthy check below and silently skips.
        _audit_ts_col = _audit_ts_override
    if _audit_ts_col:
        try:
            from ..target_temporal_audit import (
                audit_targets_over_time as _audit_targets_over_time,
                format_temporal_audit_report as _format_temporal_audit_report,
                plot_target_over_time as _plot_target_over_time,
            )
            # Resolve the timestamp source. Prefer the column in df
            # when present; fall back to the ``timestamps`` ndarray
            # that FTE.transform returned alongside targets when the
            # column has been dropped from df (the prod pattern: user
            # lists ``ts_field`` in ``columns_to_drop`` because the
            # raw datetime shouldn't be a model feature, but the
            # split / audit code still needs the values).
            # 2026-04-27 Session 7 batch 7: this fallback is what makes
            # the FTE-auto-detect path actually fire under prod-style
            # configs (where ts_field is in columns_to_drop).
            _audit_ts_values = None
            _audit_src_kind = None
            _src_df_local = locals().get("df", None)
            if _src_df_local is not None and hasattr(_src_df_local, "columns") and _audit_ts_col in _src_df_local.columns:
                _audit_ts_values = _src_df_local[_audit_ts_col]
                _audit_src_kind = "df_column"
            elif timestamps is not None:
                _audit_ts_values = timestamps
                _audit_src_kind = "fte_timestamps"
                logger.info(
                    "target_temporal_audit: column '%s' was dropped from df "
                    "(likely via columns_to_drop) — using FTE-returned "
                    "timestamps ndarray as fallback.",
                    _audit_ts_col,
                )
            if _audit_ts_values is None:
                logger.warning(
                    "target_temporal_audit: column '%s' not found in df and "
                    "FTE returned no timestamps — audit skipped. Either "
                    "set behavior_config.target_temporal_audit_column to a "
                    "column present in df, or configure ts_field on your "
                    "FeaturesAndTargetsExtractor.",
                    _audit_ts_col,
                )
            else:
                # Build the {audit_key: (col, target_type)} mapping for
                # the batch call. ``audit_key`` is unique across
                # target_types so the same target_name under two
                # target_types doesn't collide.
                _audit_input_cols: Dict[str, np.ndarray] = {}
                _audit_targets_spec: Dict[str, Tuple[str, str]] = {}
                _audit_keys_by_pair: Dict[Tuple[Any, str], str] = {}
                for _tt_outer, _named in target_by_type.items():
                    for _tname, _tvals in _named.items():
                        _audit_key = f"{slugify(str(_tt_outer).lower())}__{slugify(_tname)}"
                        _audit_col = f"__audit_target_{_audit_key}"
                        _arr = np.asarray(_tvals)
                        # Skip multilabel (2-D) — not supported by the
                        # binary/regression audit; would need per-label
                        # decomposition (future work).
                        if _arr.ndim != 1:
                            continue
                        _audit_input_cols[_audit_col] = _arr
                        _audit_targets_spec[_audit_key] = (
                            _audit_col,
                            "regression" if str(_tt_outer) == "regression" else "binary_classification",
                        )
                        _audit_keys_by_pair[(_tt_outer, _tname)] = _audit_key
                if _audit_targets_spec:
                    if _audit_src_kind == "df_column" and isinstance(_src_df_local, pl.DataFrame):
                        # Polars fastpath: timestamp column still in df
                        _batch_input = _src_df_local.select([_audit_ts_col]).with_columns([
                            pl.Series(name, arr) for name, arr in _audit_input_cols.items()
                        ])
                    elif _audit_src_kind == "df_column":
                        # Pandas: timestamp column still in df
                        _batch_input = pd.DataFrame({
                            _audit_ts_col: _src_df_local[_audit_ts_col].values,
                            **_audit_input_cols,
                        })
                    else:
                        # FTE-timestamps fallback: build a fresh frame
                        # from the stand-alone timestamps ndarray.
                        _ts_arr = np.asarray(_audit_ts_values)
                        _batch_input = pd.DataFrame({
                            _audit_ts_col: _ts_arr,
                            **_audit_input_cols,
                        })
                    _gran = getattr(behavior_config, "target_temporal_audit_granularity", "auto")
                    _batch_results = _audit_targets_over_time(
                        _batch_input,
                        timestamp_col=_audit_ts_col,
                        targets=_audit_targets_spec,
                        granularity=_gran,
                    )
                    # Index the results back by (target_type, target_name)
                    # for the per-target loop to look up.
                    for (_tt_pair, _tname_pair), _key in _audit_keys_by_pair.items():
                        _all_target_audits.setdefault(_tt_pair, {})[_tname_pair] = _batch_results.get(_key)
                    logger.info(
                        "target_temporal_audit: batched %d target(s) in one polars multi-aggregation pass.",
                        len(_audit_targets_spec),
                    )
        except Exception as _audit_err:
            logger.warning(
                "target_temporal_audit batch failed (timestamp_col='%s'): %s. Training continues without audit.",
                _audit_ts_col, _audit_err,
            )

    for target_type, targets in tqdmu_lazy_start(target_by_type.items(), desc="target type"):
        # Store original target_type mapping
        slug_to_original_target_type[slugify(str(target_type).lower())] = target_type

        # !TODO ! optimize for creation of inner feature matrices of cb,lgb,xgb here. They should be created once per featureset, not once per target.
        for cur_target_name, cur_target_values in tqdmu_lazy_start(targets.items(), desc="target"):
            # Store original cur_target_name mapping
            slug_to_original_target_name[slugify(cur_target_name)] = cur_target_name
            # Initialize rfecv_models_params before conditional to avoid NameError if mlframe_models is empty
            rfecv_models_params = {}
            if mlframe_models:
                # Set up directories for charts and models
                plot_file, model_file = _setup_model_directories(
                    target_name=target_name,
                    model_name=model_name,
                    target_type=target_type,
                    cur_target_name=cur_target_name,
                    data_dir=data_dir,
                    models_dir=models_dir,
                    save_charts=save_charts,
                )

                # Subset targets using pre-filtered indices (OD already applied globally)
                current_train_target = (
                    cur_target_values[filtered_train_idx]
                    if isinstance(cur_target_values, (np.ndarray, pl.Series))
                    else cur_target_values.iloc[filtered_train_idx]
                )
                current_val_target = None
                if filtered_val_idx is not None:
                    current_val_target = (
                        cur_target_values[filtered_val_idx]
                        if isinstance(cur_target_values, (np.ndarray, pl.Series))
                        else cur_target_values.iloc[filtered_val_idx]
                    )
                # Test target extraction for the drift report. test_idx is
                # NOT subset by the outlier-detector (test never gets
                # OD-filtered) so we use the raw test_idx here.
                current_test_target = None
                if test_idx is not None:
                    current_test_target = (
                        cur_target_values[test_idx]
                        if isinstance(cur_target_values, (np.ndarray, pl.Series))
                        else cur_target_values.iloc[test_idx]
                    )

                # 2026-04-25 Session 7: label-distribution drift report.
                # Computes per-split P(y) (binary), per-class rate
                # (multiclass), per-label rate (multilabel), or mean/std
                # (regression), and warns when train/val/test priors
                # diverge beyond the threshold. Logged BEFORE training so
                # operators catch selection-bias / temporal-prior-shift
                # without waiting hours for a miscalibrated model. Stored
                # on the metadata dict for retrospective inspection.
                _drift_report = compute_label_distribution_drift(
                    train_target=current_train_target,
                    val_target=current_val_target,
                    test_target=current_test_target,
                    target_type=str(target_type),
                )
                logger.info(format_drift_report(_drift_report, target_name=cur_target_name))
                metadata.setdefault("label_distribution_drift", {}) \
                    .setdefault(str(target_type), {})[cur_target_name] = _drift_report

                # 2026-05-10: baseline diagnostics. Cheap pre-training
                # pass that surfaces (a) headline metric of a quick fit,
                # (b) top-K feature ablation deltas, (c) init_score
                # native-residual baseline. Output stored on metadata
                # so composite-target discovery (future component) can
                # gate its expensive screening loops on the
                # ``composite_recommendation`` flag.
                try:
                    # Reuse the inline BaselineDiagnostics result if
                    # composite-discovery already computed one for this
                    # (target_type, target_name). Saves the ~30-60s
                    # ablation cost when both subsystems are enabled.
                    _existing_bd = (
                        metadata.get("baseline_diagnostics", {})
                        .get(str(target_type), {})
                        .get(cur_target_name)
                    )
                    if _existing_bd is not None:
                        logger.info(
                            "[BaselineDiagnostics] target='%s' reusing cached "
                            "diagnostic from composite-discovery precompute "
                            "(saved ~30-60s).", cur_target_name,
                        )
                    elif baseline_diagnostics_config.enabled and (
                        str(target_type) in baseline_diagnostics_config.apply_to_target_types
                    ):
                        _bd = BaselineDiagnostics(baseline_diagnostics_config)
                        _bd_report = _bd.fit_and_report(
                            train_df=filtered_train_df,
                            train_target=current_train_target,
                            feature_cols=list(filtered_train_df.columns),
                            target_type=str(target_type),
                            target_name=cur_target_name,
                            cat_features=cat_features,
                        )
                        logger.info(format_baseline_diagnostics_report(
                            _bd_report, target_name=cur_target_name,
                        ))
                        metadata.setdefault("baseline_diagnostics", {}) \
                            .setdefault(str(target_type), {})[cur_target_name] = _bd_report.to_dict()
                except Exception as _bd_err:
                    logger.warning(
                        "baseline_diagnostics failed for target='%s' (%s): %s. "
                        "Training continues without diagnostics.",
                        cur_target_name, target_type, _bd_err,
                    )

                # 2026-05-10: dummy / trivial-baseline floor (sit-alongside
                # BaselineDiagnostics). One verdict line at INFO, full table
                # at DEBUG; per-strongest overlay plot. Wrapped in try/except
                # — failure to compute the floor must never block training.
                # D7: phase qualifier per-target_type for [wall-share]
                # debuggability.
                try:
                    if dummy_baselines_config.enabled and (
                        str(target_type) in dummy_baselines_config.apply_to_target_types
                    ):
                        from ..dummy_baselines import compute_dummy_baselines

                        _ts_train = (
                            timestamps[filtered_train_idx]
                            if timestamps is not None and filtered_train_idx is not None
                            else None
                        )
                        _ts_val = (
                            timestamps[filtered_val_idx]
                            if timestamps is not None and filtered_val_idx is not None
                            else None
                        )
                        _ts_test = (
                            timestamps[test_idx]
                            if timestamps is not None and test_idx is not None
                            else None
                        )
                        # Re-attach high-card cat cols dropped earlier so per_group_mean
                        # can use them as group keys (e.g. well_id with 623 unique values).
                        _dummy_train_X = filtered_train_df
                        _dummy_val_X = filtered_val_df
                        _dummy_test_X = test_df_pd
                        _dummy_cat_features = list(cat_features or [])
                        if _dropped_high_card_data:
                            try:
                                _dummy_train_X, _dummy_val_X, _dummy_test_X, _added = _augment_with_dropped_high_card_cols(
                                    _dropped_high_card_data,
                                    filtered_train_df, filtered_val_df, test_df_pd,
                                    train_od_idx=train_od_idx, val_od_idx=val_od_idx,
                                )
                                if _added:
                                    _dummy_cat_features.extend(_added)
                                    logger.debug(
                                        "[dummy-baselines] re-attached %d auto-dropped high-card cat col(s) for per_group_mean: %s",
                                        len(_added), _added,
                                    )
                            except Exception as _aug_err:
                                logger.debug(
                                    "[dummy-baselines] failed to re-attach dropped high-card cat cols (%s); per_group_mean may be missing",
                                    _aug_err,
                                )
                        # Auto-pick quantile_alphas from QuantileRegressionConfig
                        # so the operator doesn't have to restate them per call.
                        # Only fires when target_type matches; for non-quantile
                        # targets the kwarg is ignored by compute_dummy_baselines.
                        _q_alphas = None
                        if str(target_type) == "quantile_regression":
                            _q_alphas = list(getattr(
                                quantile_regression_config, "alphas", ()
                            ) or ())
                            if not _q_alphas:
                                _q_alphas = None
                        with phase(f"dummy_baselines:{str(target_type)}", target=cur_target_name):
                            _db_report = compute_dummy_baselines(
                                target_type=str(target_type),
                                target_name=cur_target_name,
                                train_X=_dummy_train_X,
                                val_X=_dummy_val_X,
                                test_X=_dummy_test_X,
                                train_y=current_train_target,
                                val_y=current_val_target,
                                test_y=current_test_target,
                                timestamps_train=_ts_train,
                                timestamps_val=_ts_val,
                                timestamps_test=_ts_test,
                                cat_features=_dummy_cat_features,
                                quantile_alphas=_q_alphas,
                                config=dummy_baselines_config,
                                plot_file_prefix=(plot_file or ""),
                            )
                        logger.info(_db_report.format_text())
                        logger.debug(
                            "[dummy-baselines] target='%s' full table:\n%s",
                            cur_target_name, _db_report.table.to_string(),
                        )
                        # 2026-05-11 (round 2): report the strongest
                        # dummy baseline through the SAME
                        # ``report_model_perf`` pipeline that all
                        # real models go through. Yields the same
                        # text report (MAE/RMSE/MaxError/R2 +
                        # residual_audit verdict) AND the same chart
                        # (regression scatter + residual hist for
                        # regression; calibration + prob-hist for
                        # classification) as cb/xgb/lgb/linear. The
                        # earlier standalone overlay helper was
                        # cosmetically different from the model
                        # reports; user wants ONE format. Gated on
                        # ``dummy_baselines_config.plot_strongest``
                        # (default ON).
                        if (getattr(dummy_baselines_config,
                                    "plot_strongest", True)
                                and _db_report.strongest is not None):
                            try:
                                from ..evaluation import report_model_perf
                                _strongest_val_raw = _db_report.extras.get(
                                    "strongest_val_preds",
                                )
                                _strongest_test_raw = _db_report.extras.get(
                                    "strongest_test_preds",
                                )
                                # 2026-05-12 (user request): mirror the real-
                                # model title format so plots are not
                                # anonymous. Real models hit
                                # ``{ModelClass} {target_name} {model_name} {target_col} {MT*-tag}``
                                # (see train_eval._build_chart_title);
                                # the dummy gets the same suffix so the
                                # operator can see which target / experiment
                                # the scatter belongs to.
                                from .._format import format_metric as _dummy_fmt
                                _dummy_is_composite = (
                                    "__linear_residual__" in cur_target_name
                                    or "__linear_residual_multi__" in cur_target_name
                                    or "__linear_residual_grouped__" in cur_target_name
                                    or "__diff__" in cur_target_name
                                    or "__ratio__" in cur_target_name
                                    or "__logratio__" in cur_target_name
                                )
                                _dummy_mt_tag = (
                                    "MTRESID" if _dummy_is_composite else "MTTR"
                                )
                                try:
                                    if current_train_target is not None and len(current_train_target) > 0:
                                        _dummy_mt_val = float(
                                            np.asarray(current_train_target).mean()
                                        )
                                        _dummy_mt_suffix = (
                                            f" {_dummy_mt_tag}={_dummy_fmt(_dummy_mt_val)}"
                                        )
                                    else:
                                        _dummy_mt_suffix = ""
                                except Exception:
                                    _dummy_mt_suffix = ""
                                _dummy_name = (
                                    f"DummyBaseline:{_db_report.strongest} "
                                    f"{target_name} {model_name} {cur_target_name}"
                                    f"{_dummy_mt_suffix}"
                                )

                                def _split_preds_probs(arr):
                                    """Regression: 1-D ``preds``;
                                    classification: 2-D ``probs`` +
                                    derived 1-D ``preds`` via argmax."""
                                    if arr is None:
                                        return None, None
                                    a = np.asarray(arr)
                                    if a.ndim == 2:
                                        # Classification dummy: arr
                                        # is per-class probability.
                                        return np.argmax(a, axis=1), a
                                    return a, None

                                _common = dict(
                                    columns=list(
                                        getattr(filtered_train_df,
                                                "columns", []) or []
                                    ),
                                    df=None, model=None,
                                    model_name=_dummy_name,
                                    plot_outputs=getattr(
                                        reporting_config,
                                        "plot_outputs", None,
                                    ),
                                    plot_dpi=getattr(
                                        reporting_config,
                                        "plot_dpi", None,
                                    ),
                                    show_fi=False,
                                    target_type=str(target_type),
                                )
                                if (_strongest_val_raw is not None
                                        and current_val_target is not None):
                                    _vp, _vpr = _split_preds_probs(
                                        _strongest_val_raw,
                                    )
                                    _common_val = dict(_common)
                                    if plot_file:
                                        _common_val["plot_file"] = (
                                            f"{plot_file}"
                                            f"_dummy_{_db_report.strongest}_val"
                                        )
                                    report_model_perf(
                                        targets=current_val_target,
                                        preds=_vp, probs=_vpr,
                                        report_title="VAL (DUMMY) ",
                                        **_common_val,
                                    )
                                if (_strongest_test_raw is not None
                                        and current_test_target is not None):
                                    _tp, _tpr = _split_preds_probs(
                                        _strongest_test_raw,
                                    )
                                    _common_test = dict(_common)
                                    if plot_file:
                                        _common_test["plot_file"] = (
                                            f"{plot_file}"
                                            f"_dummy_{_db_report.strongest}_test"
                                        )
                                    report_model_perf(
                                        targets=current_test_target,
                                        preds=_tp, probs=_tpr,
                                        report_title="TEST (DUMMY) ",
                                        **_common_test,
                                    )
                            except Exception as _plot_err:
                                logger.warning(
                                    "[dummy-baselines] target='%s' "
                                    "report_model_perf for dummy "
                                    "failed: %s. Training continues "
                                    "without pre-training floor "
                                    "report.",
                                    cur_target_name, _plot_err,
                                )
                        metadata.setdefault("dummy_baselines", {}) \
                            .setdefault(str(target_type), {})[cur_target_name] = _db_report.to_dict()
                        # 2026-05-11: y-scale dummy for composite targets.
                        # If ``cur_target_name`` matches a composite spec,
                        # the dummy report above is on the T-scale (e.g.
                        # ``median(T_train)``) which is apples-to-oranges
                        # vs raw-target dummy (median(y_train)) and ALSO
                        # vs the wrapped composite model whose RMSE is
                        # reported y-scale post-wrap. Invert the strongest
                        # dummy predictions to y-scale via the spec's
                        # ``transform.inverse`` and recompute RMSE / MAE
                        # against the raw y so the suite-end verdict block
                        # compares both numbers on the same scale.
                        _specs_for_tt = (
                            metadata.get("composite_target_specs", {})
                            .get(str(target_type), {})
                        )
                        _matching_spec = None
                        for _tname_specs in _specs_for_tt.values():
                            for _s in _tname_specs or []:
                                if _s.get("name") == cur_target_name:
                                    _matching_spec = _s
                                    break
                            if _matching_spec is not None:
                                break
                        if (_matching_spec is not None
                                and _db_report.strongest is not None
                                and _db_report.extras.get("strongest_val_preds")
                                is not None):
                            try:
                                from ..composite import get_transform
                                _tf = get_transform(
                                    _matching_spec["transform_name"]
                                )
                                _fp = _matching_spec["fitted_params"]
                                _base_col = _matching_spec["base_column"]
                                _raw_target_col = _matching_spec["target_col"]
                                _raw_y_full = target_by_type.get(
                                    target_type, {}
                                ).get(_raw_target_col)
                                _y_scale_dummy_metrics: Dict[str, Dict[str, float]] = {}
                                for _split_name, _split_df, _split_idx, _T_preds_key in (
                                    ("val", filtered_val_df, filtered_val_idx,
                                     "strongest_val_preds"),
                                    ("test", test_df_pd, test_idx,
                                     "strongest_test_preds"),
                                ):
                                    _T_preds = _db_report.extras.get(_T_preds_key)
                                    if (_T_preds is None or _split_df is None
                                            or _split_idx is None
                                            or _raw_y_full is None
                                            or _base_col not in _split_df.columns):
                                        continue
                                    _base_split = np.asarray(
                                        _split_df[_base_col], dtype=np.float64,
                                    )
                                    _y_dummy_split = _tf.inverse(
                                        np.asarray(_T_preds, dtype=np.float64),
                                        _base_split, _fp,
                                    )
                                    _y_true_split = np.asarray(
                                        _raw_y_full, dtype=np.float64,
                                    )[_split_idx]
                                    _diff = (
                                        _y_dummy_split.astype(np.float64)
                                        - _y_true_split
                                    )
                                    _finite = np.isfinite(_diff)
                                    if _finite.sum() == 0:
                                        continue
                                    _y_scale_dummy_metrics[_split_name] = {
                                        "RMSE": float(np.sqrt(np.mean(
                                            _diff[_finite] * _diff[_finite]
                                        ))),
                                        "MAE": float(np.mean(np.abs(
                                            _diff[_finite]
                                        ))),
                                        "n_rows_finite": int(_finite.sum()),
                                    }
                                if _y_scale_dummy_metrics:
                                    metadata["dummy_baselines"][str(target_type)][
                                        cur_target_name
                                    ]["y_scale_strongest_metrics"] = (
                                        _y_scale_dummy_metrics
                                    )
                                    _ys_log_parts = [
                                        f"{k.upper()}=RMSE_y:{v['RMSE']:.4g} "
                                        f"MAE_y:{v['MAE']:.4g}"
                                        for k, v in _y_scale_dummy_metrics.items()
                                    ]
                                    logger.info(
                                        "[DUMMY_BASELINES] composite='%s' "
                                        "strongest='%s' y-scale metrics "
                                        "(inverted from T via %s): %s",
                                        cur_target_name, _db_report.strongest,
                                        _matching_spec["transform_name"],
                                        " | ".join(_ys_log_parts),
                                    )
                            except Exception as _yscale_err:
                                logger.warning(
                                    "[DUMMY_BASELINES] failed to compute "
                                    "y-scale dummy for composite '%s': %s. "
                                    "T-scale metrics remain in metadata.",
                                    cur_target_name, _yscale_err,
                                )
                except Exception as _db_err:
                    logger.warning(
                        "[DUMMY_BASELINES] FAILED target='%s' (%s): %s. "
                        "Training continues without baseline floor.",
                        cur_target_name, target_type, _db_err,
                    )
                    metadata.setdefault("dummy_baselines_failures", {}) \
                        .setdefault(str(target_type), {})[cur_target_name] = str(_db_err)

                # Look up the precomputed temporal audit (built ONCE
                # for all targets above the loop via the batch API).
                _audit = _all_target_audits.get(target_type, {}).get(cur_target_name)
                if _audit is not None:
                    try:
                        logger.info(_format_temporal_audit_report(_audit))
                        if (getattr(behavior_config, "target_temporal_audit_save_plot", True)
                                and plot_file):
                            _plot_path = f"{plot_file}_target_temporal_audit.png"
                            _plot_target_over_time(_audit, save_path=_plot_path)
                        metadata.setdefault("target_temporal_audit", {}) \
                            .setdefault(str(target_type), {})[cur_target_name] = _audit.to_dict()
                    except Exception as _audit_err:
                        logger.warning(
                            "target_temporal_audit (per-target render) failed for "
                            "target='%s': %s. Training continues.",
                            cur_target_name, _audit_err,
                        )

                if verbose:
                    logger.info(f"select_target...")

                t0_select_target = timer()
                # Build common_params and behavior_config for select_target
                od_common_params, current_behavior_config = _build_common_params_for_target(
                    common_params_dict=common_params_dict,
                    trainset_features_stats=trainset_features_stats,
                    plot_file=plot_file,
                    train_od_idx=train_od_idx,
                    val_od_idx=val_od_idx,
                    current_train_target=current_train_target,
                    current_val_target=current_val_target,
                    outlier_detector=outlier_detector,
                    behavior_config=behavior_config,
                    fairness_subgroups=fairness_subgroups,
                )

                common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs = select_target(
                    model_name=f"{target_name} {model_name} {cur_target_name}",
                    target=cur_target_values,  # Full target (for test_target extraction)
                    target_type=target_type,
                    df=None,
                    train_df=filtered_train_df,  # Use pre-filtered DataFrame
                    val_df=filtered_val_df,  # Use pre-filtered DataFrame
                    test_df=test_df_pd,  # Test set is not filtered by outlier detector
                    train_idx=filtered_train_idx,  # Use pre-filtered indices
                    val_idx=filtered_val_idx,  # Use pre-filtered indices
                    test_idx=test_idx,
                    train_details=train_details,
                    val_details=val_details,
                    test_details=test_details,
                    group_ids=group_ids,
                    cat_features=cat_features,
                    text_features=text_features,
                    embedding_features=embedding_features,
                    hyperparams_config=hyperparams_config,
                    behavior_config=current_behavior_config,
                    common_params=od_common_params,
                    mlframe_models=mlframe_models,
                    linear_model_config=linear_model_config,
                    # Fix 3B: forward pre-conversion Polars-side size so
                    # configure_training_params skips the 3-min pandas
                    # memory_usage(deep=...) scan on frames with high-
                    # cardinality object columns. Slight approximation
                    # acceptable (only feeds GPU-RAM-fit heuristic);
                    # outlier-filter shrinkage is small vs total size.
                    train_df_size_bytes=train_df_size_bytes_cached,
                    val_df_size_bytes=val_df_size_bytes_cached,
                    multilabel_dispatch_config=multilabel_dispatch_config,
                )

            if verbose:
                logger.info("  select_target done in %s", _elapsed_str(t0_select_target))
                log_ram_usage()

            # Build list of pre-pipelines (feature selection methods) to try
            pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
                use_ordinary_models=use_ordinary_models,
                rfecv_models=rfecv_models,
                rfecv_models_params=rfecv_models_params,
                use_mrmr_fs=use_mrmr_fs,
                mrmr_kwargs=mrmr_kwargs,
                custom_pre_pipelines=custom_pre_pipelines,
            )

            # Initialize pipeline cache ONCE - preprocessing output is reused across pre_pipelines.
            # Since custom transformers run AFTER preprocessing, the preprocessing output is the same
            # for ordinary models and all custom pipelines with the same model type (linear, tree, etc).
            pipeline_cache = PipelineCache()

            for pre_pipeline, pre_pipeline_name in tqdmu_lazy_start(zip(pre_pipelines, pre_pipeline_names), desc="pre_pipeline", total=len(pre_pipelines)):
                # Skip CatBoost RFECV pipeline with metamodel_func due to sklearn clone issue
                if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, behavior_config):
                    continue
                ens_models = [] if use_mlframe_ensembles else None
                orig_pre_pipeline = pre_pipeline

                # Build weight schemas from extractor output
                if sample_weights:
                    weight_schemas = sample_weights
                    if "uniform" in sample_weights:
                        logger.info("Using %d weighting schema(s) from extractor: %s", len(weight_schemas), list(weight_schemas.keys()))
                    else:
                        logger.info("Using %d weighting schema(s) from extractor: %s. Note: uniform weighting not included.", len(weight_schemas), list(weight_schemas.keys()))
                else:
                    weight_schemas = {"uniform": None}
                    logger.info("No weighting schemas from extractor, defaulting to uniform weighting.")

                # Conflict check: backward val placement + non-uniform weighting.
                # Backward split puts val BEFORE train on the timeline; recency
                # weighting makes train bias toward the NEWEST rows. The two
                # together optimise "fit the newest rows and validate on the
                # oldest ones" -- the opposite of the concept-drift proxy that
                # motivated choosing backward in the first place. Warn at suite
                # entry (rather than silently let the suite run) so the user
                # either disables recency on the extractor or reverts to
                # forward placement deliberately.
                _val_placement = getattr(split_config, "val_placement", "forward")
                if _val_placement == "backward":
                    _non_uniform = [k for k in weight_schemas.keys() if k != "uniform"]
                    if _non_uniform:
                        logger.warning(
                            "  val_placement='backward' is combined with %d non-"
                            "uniform weighting schema(s) %s. Backward val is "
                            "designed to approximate DEPLOYMENT error under "
                            "drift by mirroring the val->train gap against the "
                            "train->prod gap, while recency-style weights bias "
                            "training toward the newest rows. Together they "
                            "optimise 'fit newest, validate on oldest' -- which "
                            "contradicts the drift-proxy intent of backward. "
                            "Consider disabling use_recency_weighting on the "
                            "extractor (runs will fall back to uniform only) "
                            "or switching back to val_placement='forward'.",
                            len(_non_uniform), _non_uniform,
                        )

                # -----------------------------------------------------------------------
                # MODEL LOOP: Train each model type with all weight variations
                # Models sorted by feature tier (most features first) so that
                # text/embedding columns are dropped once per tier, not per model.
                # -----------------------------------------------------------------------
                # Resolve strategies once and reuse -- avoids re-calling get_strategy() inside
                # sort key, main loop, and tier-transition check (was ~3x redundant calls).
                # Keyed by id(m) so unhashable estimator instances / tuples don't break lookup,
                # and two identical-by-value-but-distinct-by-identity entries stay distinct.
                strategy_by_model = {id(m): get_strategy(m) for m in mlframe_models}
                sorted_models = sorted(
                    mlframe_models,
                    key=lambda m: strategy_by_model[id(m)].feature_tier(),
                    reverse=True,  # (True, True) before (False, False)
                )
                tier_dfs_cache = {}  # feature_tier -> {train_df, val_df, test_df}
                # 2026-04-28: Enum-map cache parallel to tier_dfs_cache.
                # XGBoostStrategy.prepare_polars_dataframe needs a leak-free
                # pl.Enum dict built from the train+val UNION (test EXCLUDED
                # to avoid label-time leakage). The map only depends on
                # (feature_tier, strategy class) -- invariant across the
                # weight-schema loop and across multiple models that share
                # a tier. Cache here so we compute it at most once per
                # (target, tier, strategy) instead of once per model.
                tier_enum_map_cache: Dict[tuple, Optional[Dict[str, Any]]] = {}
                prev_tier = None

                _total_models_in_run = len(list(sorted_models))
                _model_idx_in_run = 0
                for mlframe_model_name in tqdmu_lazy_start(sorted_models, desc="mlframe model"):
                    # Skip CatBoost model with metamodel_func due to sklearn clone issue
                    if _should_skip_catboost_metamodel(mlframe_model_name, target_type, behavior_config):
                        continue
                    _model_idx_in_run += 1
                    # 2026-05-10 (rec a): mid-run phase progress so the
                    # operator sees fractional progress AND per-model RAM
                    # snapshot in real time, instead of waiting for the
                    # phase summary at suite end. Logged once at MODEL
                    # entry (before the weight-schemas inner loop); the
                    # existing per-(model, weight) DONE line at the
                    # bottom of the inner loop already carries the
                    # post-fit duration + RAM. Together they bracket
                    # each model so the operator can see "model 2/4
                    # cb starting at RAM=7.2 GB" in the live tail.
                    if verbose:
                        try:
                            import psutil as _ps
                            _ram_gb_now = _ps.Process().memory_info().rss / (1024 ** 3)
                        except Exception:
                            _ram_gb_now = 0.0
                        logger.info(
                            "  process_model(%s) START -- model %d/%d, RAM=%.1fGB",
                            mlframe_model_name,
                            _model_idx_in_run, _total_models_in_run,
                            _ram_gb_now,
                        )

                    if mlframe_model_name not in models_params:
                        logger.warning(f"mlframe model {mlframe_model_name} not known, skipping...")
                        continue

                    # Use strategy pattern to determine pipeline and cache key
                    strategy = strategy_by_model[id(mlframe_model_name)]

                    # B5 release (upfront, 2026-04-23 fix): drop the
                    # pre-pipeline Polars originals as soon as we hit the
                    # FIRST strategy that can't consume them, regardless of
                    # ``feature_tier`` change. The original release block
                    # below (post-iteration) only fires on a tier transition
                    # -- but XGB and LGB share ``tier=(False,False)``, so a
                    # ``cb -> xgb -> lgb`` suite kept ``train_df_polars`` /
                    # ``val_df_polars`` / ``test_df_polars`` alive through
                    # the LGB iteration. Right after that LGB triggered the
                    # lazy polars->pandas conversion, holding **both** the
                    # 29 GB polars frames and the 57 GB pandas copies
                    # simultaneously -- peak 86 GB on the 2026-04-23 prod
                    # run, instead of ~57 GB. Releasing here, before the
                    # lazy conversion, halves peak RAM in mixed suites.
                    if (
                        not strategy.supports_polars
                        and train_df_polars is not None
                    ):
                        del train_df_polars, val_df_polars, test_df_polars
                        train_df_polars = val_df_polars = test_df_polars = None
                        tier_dfs_cache.clear()
                        tier_enum_map_cache.clear()
                        baseline_rss_mb = maybe_clean_ram_and_gpu(
                            baseline_rss_mb, df_size_mb,
                            verbose=verbose,
                            reason="non-polars-native strategy entry",
                        )
                        if verbose:
                            logger.info(
                                "  Released pre-pipeline Polars originals before "
                                "%s (non-polars-native strategy) -- frees ~30 %% "
                                "of peak RAM before lazy pandas conversion.",
                                mlframe_model_name,
                            )

                    # Clone base_pipeline per model so each iteration gets a
                    # fresh un-fitted selector (MRMR / RFECV). Previously, a
                    # single fitted ``orig_pre_pipeline`` was shared across
                    # all models in the suite -- if CB fit it on train_df
                    # and selected 3 of 4 cols, the following Linear iter
                    # saw a pipeline where MRMR was already fitted but
                    # encoder/imputer/scaler were not. ``_is_fitted`` mis-
                    # reported True -> code took the transform-only branch
                    # -> imputer.transform raised "feature names should
                    # match those passed during fit". Cloning isolates each
                    # strategy's fit state. sklearn.base.clone handles
                    # feature selectors (they're BaseEstimator-compatible)
                    # without copying fitted data -- resets parameters only.
                    _base_for_strategy = orig_pre_pipeline
                    if _base_for_strategy is not None:
                        try:
                            _base_for_strategy = clone(_base_for_strategy)
                        except Exception:
                            # Some custom base pipelines (non-BaseEstimator)
                            # don't clone; keep the original reference.
                            pass
                    pre_pipeline = strategy.build_pipeline(
                        base_pipeline=_base_for_strategy,
                        cat_features=cat_features,
                        category_encoder=category_encoder if cat_features else None,
                        imputer=imputer,
                        scaler=scaler,
                    )
                    # Cache key composition (2026-04-19 round-9 verify):
                    #   1. strategy.cache_key (e.g. "tree", "hgb", "neural")
                    #   2. pre_pipeline_name (MRMR vs RFECV vs ordinary)
                    #   3. feature_tier() -- CRITICAL: CB/LGB/XGB all inherit
                    #      cache_key="tree", but CB has tier=(True,True) while
                    #      LGB/XGB have tier=(False,False). Without tier in the
                    #      key, CB (running first by tier-desc sort) cached its
                    #      polars DF with text/embedding cols under "tree";
                    #      LGB/XGB then retrieved that cache and received cols
                    #      they can't handle (either polars when they need
                    #      pandas, or text cols they don't support). Adding
                    #      feature_tier() partitions the cache so same-tier
                    #      models still share, different-tier models don't
                    #      collide.
                    _tier_suffix = f"_tier{strategy.feature_tier()}"
                    # Container-kind suffix (2026-04-23 fix): CB, XGB, and LGB
                    # all inherit ``cache_key="tree"`` and XGB/LGB share the
                    # same ``feature_tier``. Without the kind qualifier, XGB
                    # (Polars-native) stored the *polars* post-pipeline frame
                    # under ``tree_..._tier(False,False)`` and LGB pulled that
                    # polars frame out on its next iteration -- undoing the
                    # strategy-loop lazy pandas conversion a few lines later
                    # and triggering a duplicate 224 s conversion in trainer's
                    # self-heal (2026-04-23 prod log, ~38 min wasted). Mirror
                    # the kind-tagging already used by ``_build_tier_dfs`` so
                    # pandas-consumers and polars-consumers never collide on a
                    # shared tier key.
                    _kind_suffix = f"_kind{'pl' if strategy.supports_polars else 'pd'}"
                    cache_key = (
                        f"{strategy.cache_key}_{pre_pipeline_name}{_tier_suffix}{_kind_suffix}"
                        if pre_pipeline_name else
                        f"{strategy.cache_key}{_tier_suffix}{_kind_suffix}"
                    )

                    # Polars fastpath: substitute original Polars DataFrames for models
                    # that support native Polars input (e.g. CatBoost >= 1.2.7, HGB).
                    polars_fastpath_active = train_df_polars is not None and strategy.supports_polars

                    # B3: Prepare Polars DFs once per model (outside weight loop).
                    # prepare_polars_dataframe() calls .with_columns() which allocates --
                    # doing it per weight schema wastes memory for 100GB+ DataFrames.
                    if polars_fastpath_active:
                        if verbose:
                            logger.info("  Polars fastpath active for %s (strategy=%s)", mlframe_model_name, type(strategy).__name__)
                        # Use ``cat_features`` (the post-promotion, deduplicated
                        # list from the Phase 3.5 reassignment at line ~1526) --
                        # NOT the stale ``cat_features_polars`` captured at the
                        # start of Phase 3 before auto-detection ran.
                        # Production 2026-04-19 bug: the old
                        #   _cat_features = cat_features_polars or cat_features or []
                        # short-circuit picked the stale 13-column list even
                        # after text-promotion removed 4 of them. Those 4 were
                        # later cast Categorical->String for CatBoost's text
                        # path, so CB saw cat_features=[..., 'skills_text'] with
                        # skills_text being pl.String -- its fused-cpdef
                        # ``_set_features_order_data_polars_categorical_column``
                        # has no String overload and raised "No matching
                        # signature found", burning 22 s + a 150 s pandas
                        # fallback on every run.
                        _cat_features = list(cat_features or [])

                        # Build tier-specific DFs with text/embedding columns dropped for non-supporting models
                        tier_base = {
                            "train_df": train_df_polars,
                            "val_df": val_df_polars,
                            "test_df": test_df_polars,
                        }
                        tier_polars = _build_tier_dfs(
                            tier_base, strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                        )

                        # 2026-04-28: leak-free pl.Enum map from train+val
                        # UNION of unique categorical values (test EXCLUDED).
                        # Strategy-class+tier-keyed cache so the map is built
                        # once per (target, tier, strategy) and reused across
                        # the weight-schema loop and any sibling-tier models.
                        _enum_cache_key = (strategy.feature_tier(), type(strategy).__name__)
                        if _enum_cache_key in tier_enum_map_cache:
                            _xgb_category_map = tier_enum_map_cache[_enum_cache_key]
                        elif hasattr(strategy, "build_polars_enum_map"):
                            try:
                                _xgb_category_map = strategy.build_polars_enum_map(
                                    tier_polars["train_df"],
                                    tier_polars.get("val_df"),
                                    _cat_features,
                                )
                            except Exception as _emb_exc:
                                logger.warning(
                                    "build_polars_enum_map failed for %s; "
                                    "falling back to per-DF Enum cast: %s",
                                    type(strategy).__name__, _emb_exc,
                                )
                                _xgb_category_map = None
                            tier_enum_map_cache[_enum_cache_key] = _xgb_category_map
                        else:
                            _xgb_category_map = None
                            tier_enum_map_cache[_enum_cache_key] = None

                        def _prep_polars(_df):
                            if _df is None:
                                return None
                            if _xgb_category_map is not None:
                                return strategy.prepare_polars_dataframe(_df, _cat_features, category_map=_xgb_category_map)
                            return strategy.prepare_polars_dataframe(_df, _cat_features)

                        prepared_train = _prep_polars(tier_polars["train_df"])
                        prepared_val = _prep_polars(tier_polars.get("val_df"))
                        prepared_test = _prep_polars(tier_polars.get("test_df"))

                        # Null-fill text features for CatBoost (requires no nulls in text columns).
                        # Also cast Polars Categorical/Enum to pl.String: CatBoost's Polars
                        # fastpath for text_features (`_set_features_order_data_polars_text_column`)
                        # rejects Categorical with
                        #   "Unsupported data type Categorical for a text feature column".
                        # This happens whenever a column was auto-promoted from cat_features to
                        # text_features (done in `_auto_detect_feature_types`) but its backing
                        # dtype was left as Categorical -- CB's text path requires plain string.
                        if text_features and mlframe_model_name == "cb":
                            text_cols_present = filter_existing(prepared_train, text_features)
                            if text_cols_present:
                                # Determine which of the text columns need a dtype cast.
                                needs_cast = [
                                    c for c in text_cols_present
                                    if prepared_train.schema[c] == pl.Categorical
                                    or isinstance(prepared_train.schema[c], pl.Enum)
                                ]
                                prep_exprs = []
                                for c in text_cols_present:
                                    expr = pl.col(c)
                                    if c in needs_cast:
                                        expr = expr.cast(pl.String)
                                    prep_exprs.append(expr.fill_null(""))
                                prepared_train = prepared_train.with_columns(prep_exprs)
                                if prepared_val is not None:
                                    prepared_val = prepared_val.with_columns(prep_exprs)
                                if prepared_test is not None:
                                    prepared_test = prepared_test.with_columns(prep_exprs)
                                if needs_cast and verbose:
                                    logger.info(
                                        "  Cast %d text feature(s) from Polars Categorical to String "
                                        "for CatBoost: %s",
                                        len(needs_cast), needs_cast,
                                    )

                        # Null-in-Categorical fix: applied UPSTREAM once
                        # at pre-loop level on train_df_polars/val/test
                        # (see the ``_polars_nullable_categorical_cols`` +
                        # ``_polars_fill_null_in_categorical`` block near
                        # OD-filter, keyword ``__MISSING__``). Every
                        # polars-capable strategy -- CB, XGB, HGB -- now
                        # operates on the same pre-filled frame. No
                        # per-model fill needed in the loop.

                        polars_fastpath_skip_preprocessing = strategy.requires_encoding
                    else:
                        polars_fastpath_skip_preprocessing = False

                        # Lazy pandas conversion for non-Polars-native strategies.
                        # The "deferred" half of the 2026-04-21 design: when all
                        # blockers for the Polars fastpath are non-native
                        # strategies, the upfront ``_convert_dfs_to_pandas`` is
                        # skipped and per-strategy conversion happens here,
                        # preserving RAM when CB/XGB can run natively on polars.
                        # In mixed suites (cb+xgb+lgb) this fires once when
                        # lgb's iteration is reached.
                        # B3 fix (2026-05-11): the previous message blamed the strategy ("non-Polars-native strategy CatBoostStrategy") but the strategy may BE polars-native; the actual trigger is that the polars frames were released earlier in the run (e.g. between targets). Two distinct cases now reported differently:
                        #   (a) ``strategy.supports_polars`` is FALSE — really non-native (lgb, linear, etc.).
                        #   (b) ``strategy.supports_polars`` is TRUE but ``train_df_polars`` was released — falling back to pandas because the polars originals are gone.
                        _logged_lazy_conv = False
                        for df_key in ("train_df", "val_df", "test_df"):
                            df_ = common_params.get(df_key)
                            if isinstance(df_, pl.DataFrame):
                                if not _logged_lazy_conv and verbose:
                                    if strategy.supports_polars:
                                        _reason = (
                                            "Polars originals released "
                                            "(common_params still carries "
                                            "polars frames; converting to "
                                            "pandas for inner predict path)"
                                        )
                                    else:
                                        _reason = (
                                            f"non-Polars-native strategy "
                                            f"{type(strategy).__name__}"
                                        )
                                    logger.info(
                                        "  Lazy pandas conversion for %s -- %s",
                                        mlframe_model_name, _reason,
                                    )
                                    _logged_lazy_conv = True
                                common_params[df_key] = get_pandas_view_of_polars_df(df_)

                        # Defense-in-depth (2026-04-23): immediately after the
                        # lazy conversion ran above, EVERY ``common_params``
                        # DF key must be non-polars. If a Polars frame is
                        # still sitting here it means either (a) the loop
                        # above silently missed a key (bug in this function)
                        # or (b) some later step replaced the converted
                        # frame with a polars original between iterations
                        # -- which is exactly the 2026-04-23 prod regression
                        # that pipeline_cache's kind-suffix fix addresses.
                        # The trainer-level hard-raise also catches this,
                        # but failing HERE surfaces the bug one function up
                        # the stack, closer to the cause (cheaper to debug
                        # -- you see ``strategy.__class__.__name__`` and the
                        # live ``common_params`` state, not just a model
                        # type at fit time).
                        for df_key in ("train_df", "val_df", "test_df"):
                            df_ = common_params.get(df_key)
                            if isinstance(df_, pl.DataFrame):
                                raise RuntimeError(
                                    f"Lazy pandas conversion produced incomplete "
                                    f"state for non-Polars-native strategy "
                                    f"{type(strategy).__name__} ({mlframe_model_name}): "
                                    f"common_params[{df_key!r}] is still pl.DataFrame "
                                    f"(shape={df_.shape}, id={id(df_)}). The lazy-"
                                    f"conversion hook iterated over train/val/test but "
                                    f"this key escaped. Likely cause: a ``common_params`` "
                                    f"override between lazy-conversion and here, or "
                                    f"pipeline_cache cross-stream leakage (see core.py "
                                    f"kind-suffix in cache_key)."
                                )

                        # For non-Polars models, build tier DFs from pandas common_params
                        tier_pandas = _build_tier_dfs(
                            {"train_df": common_params.get("train_df"), "val_df": common_params.get("val_df"), "test_df": common_params.get("test_df")},
                            strategy, text_features, embedding_features, tier_dfs_cache, verbose=verbose,
                        )

                    # --- WEIGHT SCHEMA LOOP: Train with each sample weighting ---
                    for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items(), desc="weighting schema"):
                        # Create model name with weight suffix
                        model_name_with_weight = common_params["model_name"]
                        model_file_name=f"{mlframe_model_name}"
                        if weight_name != "uniform":
                            model_name_with_weight += f" w={weight_name}"
                            model_file_name +=f"_{weight_name}"

                        # Shallow copy common_params - only sample_weight changes per iteration
                        current_common_params = common_params.copy()
                        current_common_params["sample_weight"] = weight_values

                        # Apply tier DFs (text/embedding columns dropped for non-supporting models)
                        if polars_fastpath_active:
                            current_common_params["train_df"] = prepared_train
                            if prepared_val is not None:
                                current_common_params["val_df"] = prepared_val
                            if prepared_test is not None:
                                current_common_params["test_df"] = prepared_test
                        else:
                            current_common_params["train_df"] = tier_pandas["train_df"]
                            if tier_pandas.get("val_df") is not None:
                                current_common_params["val_df"] = tier_pandas["val_df"]
                            if tier_pandas.get("test_df") is not None:
                                current_common_params["test_df"] = tier_pandas["test_df"]

                        # Fix 8: compute per-model input-schema fingerprint and
                        # append to model_file_name so two runs with different
                        # feature-type configs don't silently overwrite each
                        # other. Hashes realised schema (sorted cols + canonical
                        # dtypes + roles) of the DF just assigned above -- the
                        # actual DF the strategy feeds to model.fit(). Uses the
                        # outer cat/text/embedding lists because the per-model
                        # fit_params are built later (see extra_fit near the CB
                        # branch) and the roles at the DF level are stable across
                        # strategies for a given DataFrame.
                        _schema_hash, _input_schema = compute_model_input_fingerprint(
                            current_common_params.get("train_df"),
                            cat_features=cat_features,
                            text_features=text_features,
                            embedding_features=embedding_features,
                        )
                        if getattr(behavior_config, "model_file_hash_suffix", True):
                            model_file_name += f"__sch_{_schema_hash}"

                        # Append weight_name to plot_file for non-uniform weights
                        if weight_name != "uniform" and current_common_params.get("plot_file"):
                            current_common_params["plot_file"] = current_common_params["plot_file"] + weight_name + "_"

                        # Check if we have cached transformed DataFrames for this pipeline type
                        cached_dfs = pipeline_cache.get(cache_key)

                        # ============================================================================
                        # INTENTIONAL: Clone model for EACH weight schema iteration.
                        # DO NOT "OPTIMIZE" BY MOVING CLONE OUTSIDE THE LOOP!
                        # ============================================================================
                        # Each weight schema produces a DIFFERENT trained model that gets stored
                        # separately in models[type][target]. Without cloning per iteration:
                        #   - All entries would point to the SAME sklearn object (last-trained state)
                        #   - In-memory model.model.predict() would give WRONG results
                        #   - Only saved .dump files would work correctly (they capture snapshots)
                        # The clone() cost is negligible compared to training time.
                        # ============================================================================
                        original_model = models_params[mlframe_model_name]["model"]
                        try:
                            cloned_model = clone(original_model)
                        except RuntimeError:
                            # CatBoost wraps custom eval_metric objects internally, causing sklearn's
                            # identity check (param1 is not param2) to fail. Fall back to direct
                            # constructor call with the same params, which produces an equivalent
                            # fresh unfitted model without the verification step.
                            cloned_model = type(original_model)(**original_model.get_params())
                        except TypeError:
                            # NGBoost: get_params() exposes attributes (validation_fraction,
                            # early_stopping_rounds) that __init__ doesn't accept. Filter get_params
                            # to those actually in the signature.
                            import inspect as _inspect
                            _cls = type(original_model)
                            _sig_params = set(_inspect.signature(_cls.__init__).parameters) - {"self"}
                            _raw = original_model.get_params(deep=False)
                            cloned_model = _cls(**{k: v for k, v in _raw.items() if k in _sig_params})
                        # Preserve the mlframe posthoc calibration tag across clone.
                        # sklearn.clone() strips non-param attributes, which would
                        # drop the calibration directive and silently revert to an
                        # uncalibrated model. Fix 2026-04-15.
                        if getattr(original_model, "_mlframe_posthoc_calibrate", False):
                            try:
                                cloned_model._mlframe_posthoc_calibrate = True
                            except Exception:
                                pass
                        # Same problem class for the polars-fastpath sticky flag
                        # (set defensively in trainer.configure_training_params
                        # for every freshly constructed CatBoost instance -- see
                        # the 2026-04-24 prod log analysis). sklearn.clone()
                        # would otherwise blank it on every weight-schema iter,
                        # forcing the dispatch-miss + retry on the FIRST predict
                        # of CB recency / CB uniform alike. Re-assert here.
                        if getattr(original_model, "_mlframe_polars_fastpath_broken", False):
                            try:
                                cloned_model._mlframe_polars_fastpath_broken = True
                            except Exception:
                                pass
                        # XGB DMatrix-reuse shim cache (2026-04-24) AND
                        # LGB Dataset-reuse shim cache (2026-05-08). Both
                        # shims cache the heavy binned dataset on instance
                        # attributes (``_cached_train_dmatrix`` /
                        # ``_cached_train_dataset`` and their val/key
                        # siblings); sklearn.clone() blanks them. Without
                        # this hand-off, the weight-schema loop (uniform ->
                        # recency on the same train_df) would rebuild the
                        # dataset from scratch on every iteration -- defeating
                        # the whole point of the shim. Hand the cache forward
                        # so reused dataset sees consecutive ``set_label`` /
                        # ``set_weight`` swaps in place.
                        for _attr in (
                            "_cached_train_dmatrix",
                            "_cached_train_key",
                            "_cached_val_dmatrix",
                            "_cached_val_key",
                            "_cached_train_dataset",
                            "_cached_val_dataset",
                        ):
                            if hasattr(original_model, _attr):
                                try:
                                    setattr(cloned_model, _attr, getattr(original_model, _attr))
                                except Exception:
                                    pass
                        current_model_params = models_params[mlframe_model_name].copy()
                        current_model_params["model"] = cloned_model

                        # Polars fastpath: update cat_features/text_features/embedding_features
                        # in fit_params for CatBoost only.
                        # XGBoost/HGB auto-detect pl.Categorical via enable_categorical=True
                        # and do NOT accept cat_features/text_features/embedding_features as fit() params.
                        if polars_fastpath_active and mlframe_model_name == "cb" and "fit_params" in current_model_params:
                            extra_fit = {}
                            if _cat_features:
                                _valid_cat = _filter_polars_cat_features_by_dtype(
                                    prepared_train, _cat_features
                                )
                                if _valid_cat:
                                    extra_fit["cat_features"] = _valid_cat
                            if text_features:
                                cb_text = filter_existing(prepared_train, text_features)
                                if cb_text:
                                    extra_fit["text_features"] = cb_text
                            if embedding_features:
                                cb_emb = filter_existing(prepared_train, embedding_features)
                                if cb_emb:
                                    extra_fit["embedding_features"] = cb_emb
                            if extra_fit:
                                current_model_params["fit_params"] = {**current_model_params["fit_params"], **extra_fit}

                        # Build process_model kwargs using helper
                        process_model_kwargs = _build_process_model_kwargs(
                            model_file=model_file,
                            model_name_with_weight=model_name_with_weight,
                            model_file_name=model_file_name,
                            target_type=target_type,
                            pre_pipeline=pre_pipeline,
                            pre_pipeline_name=pre_pipeline_name,
                            cur_target_name=cur_target_name,
                            models=models,
                            model_params=current_model_params,
                            common_params=current_common_params,
                            ens_models=ens_models,
                            trainset_features_stats=trainset_features_stats,
                            verbose=verbose,
                            cached_dfs=cached_dfs,
                            # Fix 11 (2026-04-22): per-strategy compute of
                            # whether the Polars-ds pipeline already did the
                            # preprocessing this strategy would have needed.
                            #
                            # Previously ``polars_pipeline_applied or
                            # polars_fastpath_skip_preprocessing`` accumulated
                            # globally across the suite loop. The initial
                            # value at core.py:1995 is True when the input is
                            # Polars and the polars-ds pipeline exists -- so
                            # every later iteration (including Linear, which
                            # really DOES need the encoder/scaler/imputer)
                            # inherited True and skipped its pre_pipeline
                            # via train_eval.py:675 -> trainer.py:775
                            # ``elif skip_preprocessing: feature_selector
                            # only`` branch. LogReg then received raw
                            # pd.Categorical and crashed on 'HOURLY'.
                            #
                            # Correct semantics: the preprocessing is
                            # already done for THIS strategy iff
                            #   (a) the initial polars-ds pipeline ran at
                            #       the suite level (``polars_pipeline_applied``
                            #       as seeded before the loop), AND
                            #   (b) this strategy itself can consume
                            #       Polars frames (``supports_polars``).
                            #
                            # requires_encoding=True is NOT a sufficient
                            # trigger to re-run preprocessing: HGB declares
                            # requires_encoding=True for its pandas-fallback
                            # path only, but on the Polars fastpath HGB
                            # consumes pl.Categorical natively (no encoder
                            # needed). Gating on supports_polars alone is
                            # correct -- only the non-Polars-native
                            # strategies (Linear, Neural, sklearn-generic,
                            # LGB-via-bridge) fall through to their own
                            # pre_pipeline run in trainer.py.
                            #
                            # 2026-04-23 extension (fuzz c0003 / HGB +
                            # polars_enum + prefer_polarsds=False):
                            # when the shared polars-ds pipeline does NOT
                            # run, Fix 11's left-hand side collapses to
                            # False -- the gate was then False for HGB too,
                            # forcing pre_pipeline (with a sklearn
                            # category_encoders encoder) to fit on a
                            # pl.DataFrame and crash at convert_inputs.
                            # The polars fastpath being ACTIVE for this
                            # strategy is itself a sufficient reason to
                            # skip sklearn preprocessing -- the strategy
                            # consumes the Polars frame natively, so the
                            # encoder/scaler/imputer would both be
                            # redundant and crash on Polars input.
                            # ``polars_fastpath_active`` already encodes
                            # (train_df_polars is not None AND
                            # strategy.supports_polars), so this disjunct
                            # keeps the supports_polars-required invariant.
                            polars_pipeline_applied=(
                                (polars_pipeline_applied and strategy.supports_polars)
                                or polars_fastpath_active
                            ),
                            mlframe_model_name=mlframe_model_name,
                            metadata_columns=metadata.get("columns"),
                        )

                        t0_model = timer()
                        try:
                            with phase("process_model", model=mlframe_model_name, weight=weight_name):
                                trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                                    **process_model_kwargs
                                )
                        except Exception as model_err:
                            # Skip-and-continue only when the caller explicitly opted in.
                            # KeyboardInterrupt is intentionally NOT caught -- Ctrl-C must
                            # still abort the run. A native SIGSEGV that kills the process
                            # also won't be caught here; only Python-level exceptions
                            # (XGBoostError, CatBoostError, MemoryError, ...) are.
                            if not behavior_config.continue_on_model_failure:
                                raise
                            logger.error(
                                f"  process_model({mlframe_model_name}, w={weight_name}) FAILED after "
                                f"{_elapsed_str(t0_model)} -- {type(model_err).__name__}: {model_err}. "
                                f"continue_on_model_failure=True -> skipping and moving on.",
                                exc_info=True,
                            )
                            metadata.setdefault("failed_models", []).append({
                                "model": mlframe_model_name,
                                "weighting": weight_name,
                                "error_type": type(model_err).__name__,
                                "error_message": str(model_err),
                            })
                            continue  # next weight_name in the inner loop
                        if verbose:
                            logger.info("  process_model(%s, w=%s) done -- %s", mlframe_model_name, weight_name, _elapsed_str(t0_model))

                        # XGB DMatrix-reuse shim cache (2026-04-24, second
                        # half). The cache lives on the cloned_model that
                        # ``process_model`` actually fit. Hand it back to the
                        # template (``original_model`` / ``models_params[...]
                        # ["model"]``) so the NEXT weight-schema iteration's
                        # ``clone()`` carries the cache forward (via the
                        # forward-transfer block above). Without this, the
                        # cache would be born and die inside one weight-
                        # schema iteration -- wasted because the SAME train
                        # frame is consumed by the next iteration with only
                        # a different sample_weight. Symmetric counterpart
                        # to the forward-transfer block in this loop.
                        for _attr in (
                            "_cached_train_dmatrix",
                            "_cached_train_key",
                            "_cached_val_dmatrix",
                            "_cached_val_key",
                            "_cached_train_dataset",
                            "_cached_val_dataset",
                        ):
                            if hasattr(cloned_model, _attr):
                                _val = getattr(cloned_model, _attr)
                                if _val is not None:
                                    try:
                                        setattr(original_model, _attr, _val)
                                    except Exception:
                                        pass

                        # Fix 8: record this model's input-schema fingerprint
                        # in metadata so load-time can verify or at least diff
                        # against the serving data. Keyed by the final
                        # model_file_name (already includes weight + hash), so
                        # repeat weights don't collide.
                        # 2026-04-24: extend with target_type / n_classes /
                        # multilabel_strategy + schema_version to support
                        # multi-output target inference at load time. Old
                        # artifacts without these fields fall through to
                        # legacy binary inference (load_mlframe_suite).
                        _record = {
                            "schema_hash": _schema_hash,
                            "input_schema": _input_schema,
                            "mlframe_model": mlframe_model_name,
                            "weight_name": weight_name,
                            # Multi-output extensions:
                            "target_type": str(target_type) if target_type is not None else None,
                            "schema_version": 2,  # 1=legacy, 2=multi-output-aware
                        }
                        try:
                            from ..configs import TargetTypes as _TT
                            if target_type == _TT.MULTILABEL_CLASSIFICATION:
                                # Number of label outputs and dispatch strategy
                                _record["n_classes"] = (
                                    int(train_y.shape[1])
                                    if hasattr(train_y, "shape") and train_y.ndim == 2
                                    else None
                                )
                                _record["multilabel_strategy"] = "native" if (
                                    hasattr(strategy, "supports_native_multilabel") and strategy.supports_native_multilabel
                                ) else "wrapper"
                            elif target_type == _TT.MULTICLASS_CLASSIFICATION:
                                _record["n_classes"] = (
                                    int(len(np.unique(np.asarray(train_y))))
                                    if hasattr(train_y, "shape") else None
                                )
                                _record["multilabel_strategy"] = None
                            else:
                                _record["n_classes"] = None
                                _record["multilabel_strategy"] = None
                        except Exception:
                            # Defensive -- never fail the metadata write because
                            # of an introspection problem on multi-output fields.
                            pass
                        metadata.setdefault("model_schemas", {})[model_file_name] = _record

                        # Cache the transformed DataFrames if not already cached
                        if cached_dfs is None:
                            pipeline_cache.set(cache_key, train_df_transformed, val_df_transformed, test_df_transformed)

                    # Update orig_pre_pipeline for tree models only.
                    # Tree models return just the base_pipeline (feature selector) from build_pipeline(),
                    # so after process_model() fits it, we preserve the fitted version for subsequent models.
                    # Non-tree models wrap base_pipeline in a full Pipeline (with encoder/imputer/scaler),
                    # which we don't want to use as the base for other model types.
                    # For optimal performance, list tree models first in mlframe_models.
                    if cache_key.startswith("tree"):
                        orig_pre_pipeline = pre_pipeline

                    # Release XGB / LGB shim cache memory at strategy-iter end
                    # (2026-04-24 follow-up; LGB shim added 2026-05-08).
                    # Both shims cache the heavy binned dataset on
                    # ``_cached_train_*`` / ``_cached_val_*`` instance attrs
                    # as a fit-only scratchpad used by the inner
                    # weight-schema loop (uniform -> recency swap in place
                    # via ``set_label`` / ``set_weight``).
                    # Once the inner loop has finished, nothing downstream
                    # reads those attrs:
                    #   * ensemble scoring (below) uses pre-computed probs
                    #     stored on the SimpleNamespace wrappers in
                    #     ``ens_models`` -- NOT ``.predict()`` calls;
                    #   * ``.predict`` / ``.predict_proba`` route through
                    #     ``_Booster`` (attached at fit end), not the cache;
                    #   * model save goes through pickle / joblib, which
                    #     our ``__getstate__`` override strips the cache
                    #     from anyway.
                    # The prod log showed a 7.3M x 105 frame holding ~8 GB
                    # of QuantileDMatrix memory per XGB iteration; LGB
                    # Dataset binning is similarly heavy. Releasing this
                    # cache between strategies frees ~30 % of peak RAM.
                    # Duck-typed: only the shims expose ``clear_cache()``.
                    # CB / sklearn estimators skip harmlessly via the
                    # ``callable`` check.
                    def _maybe_clear_shim_cache(est):
                        fn = getattr(est, "clear_cache", None)
                        if callable(fn):
                            try:
                                fn()
                            except Exception:
                                pass
                    # Template held under models_params; release its cache.
                    _maybe_clear_shim_cache(original_model)
                    # Each previously-fitted model snapshot parked in
                    # ``ens_models`` may also hold a cache ref (forward-
                    # transfer at clone copied the reference, not moved it).
                    # Release on all -- their probs are already recorded
                    # and the cache is not read anywhere else.
                    if ens_models:
                        for _ens_ns in ens_models:
                            _maybe_clear_shim_cache(getattr(_ens_ns, "model", None))

                    # B5: Release Polars originals after all tier-1 (Polars-native) models finish.
                    # When transitioning to a lower tier, pre-pipeline Polars DFs are no longer needed.
                    cur_tier = strategy.feature_tier()
                    if prev_tier is not None and cur_tier != prev_tier and not strategy.supports_polars:
                        if train_df_polars is not None:
                            del train_df_polars, val_df_polars, test_df_polars
                            train_df_polars = val_df_polars = test_df_polars = None
                            tier_dfs_cache.clear()
                            tier_enum_map_cache.clear()
                            baseline_rss_mb = maybe_clean_ram_and_gpu(baseline_rss_mb, df_size_mb, verbose=verbose, reason="tier transition")
                            if verbose:
                                logger.info("  Released pre-pipeline Polars originals (tier transition)")
                    prev_tier = cur_tier

                if ens_models and len(ens_models) > 1:
                    if verbose:
                        logger.info(f"evaluating simple ensembles...")
                    # Get feature count from transformed DataFrame for display
                    ens_n_features = train_df_transformed.shape[1] if train_df_transformed is not None else None
                    # Name the ensemble by its actual members so post-hoc log
                    # grep shows *which* models participated. The old
                    # ``"{N}models "`` label hid dropouts -- in the 2026-04-23
                    # prod log LGB silently failed, yet the ensemble was
                    # reported as "2models" with no hint that it was just
                    # CB+XGB. Cap the list to keep headers readable:
                    #   <=4 members -> "[cb+xgb+lgb]"
                    #   >4        -> "[N=5]" (avoid bloated titles)
                    # 2026-05-11 (user request): delegate to the shared ``short_model_tag`` helper, which ALSO strips the internal shim suffixes (``WithDMatrixReuse`` / ``WithDatasetReuse``) so ``LinearRegression`` stays as ``LinearRegression`` in the label, but ``XGBRegressorWithDMatrixReuse`` collapses to ``xgb`` (was already correct for tree families via prefix match; this also handles any future shimmed non-tree class cleanly).
                    from .._format import short_model_tag as _short_tag_fn
                    def _short_model_tag(ns):
                        return _short_tag_fn(getattr(ns, "model", ns))
                    _member_tags = [_short_model_tag(m) for m in ens_models]
                    if len(_member_tags) <= 4:
                        _members_label = "[" + "+".join(_member_tags) + "]"
                    else:
                        _members_label = f"[N={len(_member_tags)}]"
                    # 2026-05-11 (user request): honour ``behavior_config.confidence_ensemble_quantile`` so users can disable the "Conf Ensemble" output entirely by setting the quantile to 0.0 -- previously hard-coded at the ``score_ensemble`` default 0.1 which produced 6 flavor x 2 split = 12 noisy log blocks per ensemble pass.
                    _conf_q = float(getattr(behavior_config, "confidence_ensemble_quantile", 0.1))
                    _ensembles = score_ensemble(  # Result used for side effects (logging/metrics)
                        models_and_predictions=ens_models,
                        ensemble_name=f"{pre_pipeline_name}{_members_label} ",
                        n_features=ens_n_features,
                        uncertainty_quantile=_conf_q,
                        **common_params,
                    )

    # ==================================================================================
    # 6. RECURRENT MODEL TRAINING
    # ==================================================================================

    if recurrent_models and (train_sequences is not None or train_df is not None):
        if verbose:
            log_phase("PHASE 5: Recurrent Model Training")

        from ..trainer import _configure_recurrent_params

        # Determine if this is a regression task
        use_regression = TargetTypes.REGRESSION in target_by_type

        # Configure recurrent model parameters
        recurrent_params = _configure_recurrent_params(
            recurrent_models=recurrent_models,
            recurrent_config=recurrent_config,
            sequences_train=train_sequences,
            features_train=train_df_pd if train_df_pd is not None else train_df,
            use_regression=use_regression,
        )

        # Train recurrent models
        for recurrent_model_name in tqdmu_lazy_start(recurrent_models, desc="recurrent model"):
            model_name_lower = recurrent_model_name.lower()
            if model_name_lower not in recurrent_params:
                logger.warning(f"Recurrent model {recurrent_model_name} not configured, skipping...")
                continue

            recurrent_model = recurrent_params[model_name_lower]["model"]

            # Iterate over target types and targets
            for target_type, targets in target_by_type.items():
                for cur_target_name, target_values in targets.items():
                    if verbose:
                        logger.info("Training %s for target %s...", recurrent_model_name, cur_target_name)

                    # Extract train/val/test targets
                    train_target = target_values[train_idx] if hasattr(target_values, '__getitem__') else target_values.iloc[train_idx]
                    val_target = target_values[val_idx] if val_idx is not None and hasattr(target_values, '__getitem__') else None
                    test_target = target_values[test_idx] if hasattr(target_values, '__getitem__') else target_values.iloc[test_idx]

                    # Convert to numpy if needed
                    if hasattr(train_target, 'to_numpy'):
                        train_target = train_target.to_numpy()
                    elif hasattr(train_target, 'values'):
                        train_target = train_target.values

                    if val_target is not None:
                        if hasattr(val_target, 'to_numpy'):
                            val_target = val_target.to_numpy()
                        elif hasattr(val_target, 'values'):
                            val_target = val_target.values

                    if hasattr(test_target, 'to_numpy'):
                        test_target = test_target.to_numpy()
                    elif hasattr(test_target, 'values'):
                        test_target = test_target.values

                    # Clone model for this target
                    model_clone = clone(recurrent_model)

                    try:
                        # Fit the model
                        model_clone.fit(
                            sequences=train_sequences,
                            features=train_df_pd if train_df_pd is not None else None,
                            labels=train_target,
                            val_sequences=val_sequences,
                            val_features=val_df_pd if val_df_pd is not None else None,
                            val_labels=val_target,
                        )

                        # Store the trained model
                        models[target_type][cur_target_name].append(model_clone)

                        if verbose:
                            logger.info("Successfully trained %s for %s", recurrent_model_name, cur_target_name)

                    except Exception as e:
                        logger.error(f"Failed to train {recurrent_model_name} for {cur_target_name}: {e}")
                        continue

    if verbose:
        log_phase(f"Training suite completed for {model_name}, {sum(len(v) for targets in models.values() for v in targets.values())} models.")
        log_ram_usage()

    # Aggregate per-model fairness_report into metadata so callers can access it without
    # re-walking the models dict. Trainer stores fairness_report in model.metrics[split].
    # Fix date 2026-04-15: bug B (fairness_report not propagated into suite metadata).
    fairness_reports: Dict[str, Any] = {}
    for _ttype, _targets in models.items():
        for _tname, _model_list in _targets.items():
            for _m in _model_list:
                _m_metrics = getattr(_m, "metrics", None)
                if not isinstance(_m_metrics, dict):
                    continue
                for _split in ("test", "val", "train"):
                    _split_metrics = _m_metrics.get(_split)
                    if isinstance(_split_metrics, dict) and "fairness_report" in _split_metrics:
                        _key = f"{_ttype}__{_tname}__{getattr(_m, 'model_name', type(getattr(_m, 'model', _m)).__name__)}__{_split}"
                        fairness_reports[_key] = _split_metrics["fairness_report"]
    if fairness_reports:
        metadata["fairness_report"] = fairness_reports

    # Save metadata again with slug-to-original name mappings (for load_mlframe_suite)
    _finalize_and_save_metadata(
        metadata=metadata,
        outlier_detector=outlier_detector,
        outlier_detection_result=outlier_detection_result,
        trainset_features_stats=trainset_features_stats,
        data_dir=data_dir,
        models_dir=models_dir,
        target_name=target_name,
        model_name=model_name,
        verbose=0,  # Silent to avoid duplicate log messages
        slug_to_original_target_type=slug_to_original_target_type,
        slug_to_original_target_name=slug_to_original_target_name,
    )

    if verbose:
        logger.info("[phases] Top phases by wall-clock time:\n%s", format_phase_summary())

        # 2026-05-10 (rec f): top-N wall-share so the reader immediately
        # sees where the time went vs total. Reads from the same phase
        # registry as ``format_phase_summary``; percentages computed
        # against the longest-running phase (effectively the suite root).
        # Helps spot plot/render-bound vs train-bound runs at a glance.
        try:
            from ..phases import phase_snapshot
            _snap = phase_snapshot()
            if _snap:
                _root_wall = _snap[0][1] if _snap else 0.0
                if _root_wall > 0:
                    _share_str = ", ".join(
                        f"{p}={tot/_root_wall*100:.1f}%"
                        for p, tot, _ in _snap[:8]
                    )
                    logger.info("[wall-share] top: %s", _share_str)
        except Exception:
            pass

        # Kaleido oneshot fallback summary (rec b cont'd): if any plotly
        # PNG/SVG/PDF saves took the slow oneshot path, surface the
        # cumulative cost so the reader knows the ROI of upgrading
        # kaleido. The per-call warning was suppressed (idempotent),
        # so this is the canonical place to learn about it.
        try:
            from mlframe.reporting.renderers.plotly import (
                get_kaleido_oneshot_stats, reset_kaleido_oneshot_stats,
            )
            _kal_n, _kal_wall = get_kaleido_oneshot_stats()
            if _kal_n > 0:
                logger.info(
                    "[plotly-render] kaleido oneshot fallback fired %d times "
                    "(cumulative %.1fs wall, %.0fms/call avg). Persistent "
                    "sync-server path would be ~10-100x faster -- upgrade "
                    "kaleido (>=1.x ships ``start_sync_server``) to enable.",
                    _kal_n, _kal_wall, (_kal_wall / _kal_n) * 1000,
                )
            reset_kaleido_oneshot_stats()
        except Exception:
            pass

    # Surface the selected-features list per trained model so callers
    # can introspect feature-selection outputs (MRMR / RFECV) without
    # walking the nested entry namespace. Aggregate every entry's
    # ``columns`` (post-pipeline feature list) under
    # ``metadata['selected_features']`` keyed by ``f"{target_type}/{target_name}/{model_name}"``
    # plus a flat ``metadata['all_selected_features']`` union for the
    # common bizvalue pattern of "did any model keep the informative
    # feature?". 2026-04-27 (batch 3): unblocks the three xfails in
    # ``tests/training/test_bizvalue_feature_selection.py`` that
    # previously said "selected features not surfaced on suite outputs".
    _selected_features_per_model: dict = {}
    _selected_features_union: set = set()
    for _tt, _by_name in (models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        for _tn, _entries in _by_name.items():
            if not isinstance(_entries, list):
                continue
            for _entry in _entries:
                _cols = getattr(_entry, "columns", None)
                _mn = getattr(_entry, "model_name", None) or ""
                if _cols is None:
                    continue
                _key = f"{_tt}/{_tn}/{_mn}" if _mn else f"{_tt}/{_tn}"
                _selected_features_per_model[_key] = list(_cols)
                _selected_features_union.update(_cols)
                # Also expose ``selected_features_`` directly on the
                # entry so the standard sklearn-style probe finds it
                # (matches ``getattr(entry, 'selected_features_')``).
                try:
                    _entry.selected_features_ = list(_cols)
                except Exception:
                    pass
    if _selected_features_per_model:
        # Flat sorted union (column names) -- matches the existing
        # ``_collect_selected_features`` probe in the bizvalue tests
        # which does ``list(metadata['selected_features'])`` and
        # checks INFORMATIVE_NAMES membership. Per-model breakdown
        # surfaces under a separate key so callers wanting the
        # diagnostic detail can find it.
        metadata["selected_features"] = sorted(_selected_features_union)
        metadata["selected_features_per_model"] = _selected_features_per_model

    # ==================================================================================
    # 6. COMPOSITE-TARGET WRAPPING (post-fit, y-scale predictions)
    # ==================================================================================
    #
    # The per-target loop trained models on the T-scale composite
    # target (e.g. ``T = y - alpha*base - beta`` for linear_residual).
    # Predict-time those models return T-scale, which is useless for
    # downstream consumers expecting the original y-scale. Wrap each
    # fitted composite-target model in a ``CompositeTargetEstimator``
    # so ``model.predict(X)`` automatically applies the inverse
    # transform and returns y-scale.
    #
    # Wrapping is post-hoc (no re-training): the wrapper takes the
    # ALREADY-fitted inner model + the spec's fitted_params and adds
    # the inverse / clip / fallback machinery on top.
    composite_specs_by_target_type = metadata.get("composite_target_specs", {}) or {}
    # Train-prediction cache shared across the y-scale-metrics block
    # (post-wrap) and the cross-target-ensemble RMSE block. Key is
    # ``id(model)`` of the wrapped / raw component; value is the
    # y-scale prediction array on ``filtered_train_df``. The y-scale-
    # metrics block fills it; the ensemble block reads from it before
    # falling back to a fresh predict call. Saves K predict calls per
    # target on the LightGBM/XGB hot path.
    _train_pred_cache: Dict[int, np.ndarray] = {}
    if composite_specs_by_target_type:
        from ..composite import CompositeTargetEstimator as _CTE
        for _tt_w, _by_name in (models or {}).items():
            if not isinstance(_by_name, dict):
                continue
            _tt_specs = composite_specs_by_target_type.get(str(_tt_w), {})
            if not _tt_specs:
                continue
            # Build ``composite_name -> (original_target, spec)`` lookup
            # so wrapping is O(K) per pass.
            _name_to_spec: Dict[str, Tuple[str, Dict[str, Any]]] = {}
            for _orig_tname, _spec_list in _tt_specs.items():
                for _spec in _spec_list:
                    _name_to_spec[_spec["name"]] = (_orig_tname, _spec)
            for _composite_name, _entries in list(_by_name.items()):
                if _composite_name not in _name_to_spec:
                    continue  # not a composite target
                _orig_tname, _spec = _name_to_spec[_composite_name]
                # ``y_train`` for wrapping = original y values (NOT T)
                # at the train rows the wrapper saw at fit time.
                _y_full = target_by_type.get(_tt_w, {}).get(_orig_tname)
                if _y_full is None:
                    logger.warning(
                        "[CompositeTargetEstimator] missing original target '%s' "
                        "in target_by_type for composite='%s'; skipping wrap. "
                        "Predictions will remain in T-scale.",
                        _orig_tname, _composite_name,
                    )
                    continue
                try:
                    _y_train_for_wrap = np.asarray(_y_full)[filtered_train_idx]
                except Exception as _y_err:
                    logger.warning(
                        "[CompositeTargetEstimator] cannot align y_train for '%s': %s. "
                        "Skipping wrap.",
                        _composite_name, _y_err,
                    )
                    continue
                if not isinstance(_entries, list):
                    continue
                for _i, _entry in enumerate(_entries):
                    # Entries may be plain estimators OR wrapper objects
                    # carrying the model on ``.model``. Try ``.model``
                    # first, fall back to the entry itself.
                    _inner = getattr(_entry, "model", None) or _entry
                    if not hasattr(_inner, "predict"):
                        # Not an estimator -- skip (e.g. metadata-only
                        # placeholder entry).
                        continue
                    try:
                        _wrapper = _CTE.from_fitted_inner(
                            fitted_inner=_inner,
                            transform_name=_spec["transform_name"],
                            base_column=_spec["base_column"],
                            transform_fitted_params=_spec["fitted_params"],
                            y_train=_y_train_for_wrap,
                        )
                    except Exception as _wrap_err:
                        logger.warning(
                            "[CompositeTargetEstimator] wrap failed for '%s' (entry %d): %s. "
                            "Predictions will remain in T-scale.",
                            _composite_name, _i, _wrap_err,
                        )
                        continue
                    # Replace the inner model on the entry (preserve
                    # auxiliary metadata: columns, model_name, metrics).
                    if hasattr(_entry, "model"):
                        try:
                            _entry.model = _wrapper
                        except Exception:
                            # Read-only attribute: replace the entry
                            # itself with the wrapper.
                            _entries[_i] = _wrapper
                    else:
                        _entries[_i] = _wrapper
                logger.info(
                    "[CompositeTargetEstimator] wrapped %d model(s) for composite "
                    "target '%s'; predictions now y-scale.",
                    len(_entries), _composite_name,
                )
                # Compute parallel y-scale RMSE / MAE for each wrapped
                # entry on train + val + test slices. The per-target
                # loop's metrics were computed on T-scale BEFORE wrap;
                # this block fills the y-scale gap so callers can
                # compare composite to raw on the same scale.
                _metrics_dict = metadata.setdefault(
                    "composite_target_y_scale_metrics", {},
                ).setdefault(str(_tt_w), {}).setdefault(_composite_name, [])
                _metrics_dict.clear()
                _y_full_metric = target_by_type.get(_tt_w, {}).get(_orig_tname)
                if _y_full_metric is None:
                    continue
                _y_arr_metric = np.asarray(_y_full_metric)
                for _entry in _entries:
                    _wrapper_for_score = getattr(_entry, "model", None) or _entry
                    _entry_y_scores: Dict[str, Dict[str, float]] = {}
                    for _split_name, _split_idx, _split_df in (
                        ("train", filtered_train_idx, filtered_train_df),
                        ("val", filtered_val_idx, filtered_val_df),
                        ("test", test_idx, test_df_pd),
                    ):
                        if _split_idx is None or _split_df is None:
                            continue
                        try:
                            _y_split = _y_arr_metric[_split_idx]
                            _y_pred = np.asarray(
                                _wrapper_for_score.predict(_split_df),
                                dtype=np.float64,
                            ).reshape(-1)
                            # F6 diagnostic (2026-05-11): suspicious RMSE_y values in the 05:03 TVT run (MLP-wrapped composite gave 0.49 on a target where init_score AR(1) baseline gives RMSE=11.12 -- impossibly good). Sample-log the first 3 (y_pred, y_true) pairs per split so the next run reveals whether the wrapper is returning y-scale predictions correctly OR there is a leakage / contract mismatch. Single line per entry x split keeps the spam bounded.
                            if _split_idx is not None and len(_y_split) > 0:
                                _n_dbg = min(3, len(_y_split))
                                _pairs = ", ".join(
                                    f"({_y_pred[_i]:.3f}, {_y_split[_i]:.3f})"
                                    for _i in range(_n_dbg)
                                )
                                _outer_dbg = getattr(_entry, "model", None) or _entry
                                _inner_dbg = getattr(_outer_dbg, "base_estimator", None) or getattr(_outer_dbg, "estimator_", None) or _outer_dbg
                                logger.debug(
                                    "[CompositeTargetEstimator.diag] inner=%s split=%s sample(y_hat, y_true) = %s",
                                    type(_inner_dbg).__name__, _split_name, _pairs,
                                )
                            # Cache the train prediction for the
                            # cross-target ensemble RMSE block to
                            # avoid a second predict call on the same
                            # data.
                            if _split_name == "train":
                                _train_pred_cache[id(_wrapper_for_score)] = _y_pred
                            _diff = _y_pred - _y_split.astype(np.float64)
                            _finite = np.isfinite(_diff)
                            if _finite.sum() == 0:
                                continue
                            # R^2 = 1 - SS_res / SS_tot. When the split's
                            # y has zero variance R^2 is undefined; we
                            # emit NaN so the summary explicitly marks
                            # the degenerate case rather than 0.0.
                            _y_finite = _y_split.astype(np.float64)[_finite]
                            _ss_tot = float(np.sum(
                                (_y_finite - _y_finite.mean()) ** 2
                            ))
                            _ss_res = float(np.sum(
                                _diff[_finite] * _diff[_finite]
                            ))
                            _r2 = (1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")
                            _entry_y_scores[_split_name] = {
                                "RMSE": float(
                                    np.sqrt(np.mean(_diff[_finite] * _diff[_finite]))
                                ),
                                "MAE": float(np.mean(np.abs(_diff[_finite]))),
                                "R2": _r2,
                                "n_rows_finite": int(_finite.sum()),
                            }
                        except Exception:
                            # Best-effort: any predict failure simply
                            # omits that split's y-scale metrics.
                            continue
                    _metrics_dict.append({
                        "model_name": getattr(_entry, "model_name", None),
                        "metrics": _entry_y_scores,
                    })
                    # User-facing fix (2026-05-11): the per-target loop
                    # printed RMSE / MAE / R^2 on the T-scale (composite
                    # target before inverse), which is apples-to-oranges
                    # vs raw-target models. Log a y-scale summary here
                    # so the user sees the COMPARABLE numbers in the
                    # script output for each wrapped composite model.
                    if _entry_y_scores:
                        from .._format import (
                            format_metric as _fmt,
                            strip_shim_suffix as _strip,
                        )
                        _y_summary_parts: List[str] = []
                        for _split_name in ("train", "val", "test"):
                            _s = _entry_y_scores.get(_split_name)
                            if not _s:
                                continue
                            _y_summary_parts.append(
                                f"{_split_name.upper()}=RMSE_y:{_fmt(_s['RMSE'])} "
                                f"MAE_y:{_fmt(_s['MAE'])} "
                                f"R2_y:{_fmt(_s.get('R2', float('nan')), 4)}"
                            )
                        if _y_summary_parts:
                            # B1-v2 fix (2026-05-11): drill into the WRAPPED model. After wrapping, ``_entry.model`` IS the CompositeTargetEstimator -- using its type name in the log gives the unhelpful ``model='CompositeTargetEstimator'`` (5 entries in a row, all identical). Look one level deeper at ``_entry.model.base_estimator`` (the actual inner cb / xgb / lgb / linear / mlp) for the diagnostic name.
                            _mn = getattr(_entry, "model_name", None)
                            if not _mn:
                                _outer = getattr(_entry, "model", None) or _entry
                                _inner_actual = getattr(_outer, "base_estimator", None) or getattr(_outer, "estimator_", None) or _outer
                                _mn = _strip(type(_inner_actual).__name__)
                            else:
                                _mn = _strip(_mn)
                            logger.info(
                                "[CompositeTargetEstimator] composite='%s' "
                                "model='%s' y-scale metrics (post-inverse, "
                                "comparable to raw): %s",
                                _composite_name, _mn,
                                " | ".join(_y_summary_parts),
                            )

    # ==================================================================================
    # 7. CROSS-TARGET ENSEMBLE (post-wrap; opt-in via config)
    # ==================================================================================
    #
    # After every composite-target model is wrapped to y-scale we can
    # combine them into one final predictor per original target. The
    # ensemble is OPT-IN via ``cross_target_ensemble_strategy`` and
    # produces a SimpleNamespace entry under
    # ``models[type][f"_CT_ENSEMBLE__{original_target}"]`` so downstream
    # consumers can pick it without having to know which composite to
    # trust.
    _ce_strategy = getattr(
        composite_target_discovery_config, "cross_target_ensemble_strategy", "off",
    )
    # Diagnostic: emit a one-line state banner whenever the user has
    # composite discovery enabled, regardless of whether the gate
    # actually opens. Without this, users who set
    # ``cross_target_ensemble_strategy="nnls_stack"`` but get no
    # ``[CompositeCrossTargetEnsemble] ...`` lines have no way to tell
    # whether the gate was closed (strategy=off, no specs) or whether
    # the build silently failed for every target. Emitting the banner
    # unconditionally turns "no log lines" into a debuggable signal.
    if composite_target_discovery_config.enabled:
        _n_specs_total = sum(
            sum(len(v) for v in _tt_specs.values())
            for _tt_specs in (composite_specs_by_target_type or {}).values()
        )
        logger.info(
            "[CompositeCrossTargetEnsemble] entry: strategy='%s', "
            "target_types=%d, composite_specs=%d",
            _ce_strategy,
            len(composite_specs_by_target_type or {}),
            _n_specs_total,
        )
    if (composite_target_discovery_config.enabled
            and _ce_strategy != "off"
            and composite_specs_by_target_type):
        from ..composite import CompositeCrossTargetEnsemble as _CrossEns

        # R10c bug #4 fix: cross-target ensemble must call
        # ``inner.predict`` on input that has already been transformed
        # by the inner's ``pre_pipeline`` (SimpleImputer + StandardScaler
        # for linear models, identity for tree models). Without this,
        # LinearRegression / Ridge components blow up on raw frames
        # with NaN because the imputer never ran. Wrap each raw
        # component in a thin pipeline-aware shim that applies the
        # entry's pre_pipeline before delegating to the model.
        # Composite-target components (wrapped via
        # ``CompositeTargetEstimator``) handle this internally and
        # don't need the shim.
        class _PrePipelinePredictShim:
            __slots__ = ("_model", "_pre_pipeline", "_name")

            def __init__(self, model, pre_pipeline, name):
                self._model = model
                self._pre_pipeline = pre_pipeline
                self._name = name

            def predict(self, X):
                X_in = X
                if self._pre_pipeline is not None:
                    try:
                        X_in = self._pre_pipeline.transform(X)
                    except Exception:
                        # Some pipelines reject pandas vs polars
                        # mismatches at the boundary; fall through to
                        # the inner predict which will raise a more
                        # descriptive error.
                        X_in = X
                return self._model.predict(X_in)

            def __repr__(self):
                return f"_PrePipelinePredictShim({self._name})"

        for _tt_e, _tt_specs in composite_specs_by_target_type.items():
            if not _tt_specs:
                continue
            # StrEnum: ``models.get(str_key)`` is hash-equivalent to
            # ``models.get(enum_key)`` so a plain string key works here.
            # Guard with explicit-skip log when the target type has no
            # trained models (e.g. dropped at split time) so users see
            # WHY the ensemble didn't fire for that type rather than a
            # silent skip.
            if _tt_e not in (models or {}):
                logger.info(
                    "[CompositeCrossTargetEnsemble] target_type='%s': no models "
                    "registered; ensemble skipped.", _tt_e,
                )
                continue
            for _orig_tname, _spec_list in _tt_specs.items():
                # Collect all wrapped composite-target entries plus the
                # raw-target entries for this original target. The raw
                # entries are at ``models[tt][orig_tname]``.
                _components: List[Any] = []
                _component_names: List[str] = []
                _orig_entries = (models or {}).get(_tt_e, {}).get(_orig_tname, []) or []
                for _i, _entry in enumerate(_orig_entries):
                    _inner = getattr(_entry, "model", None) or _entry
                    if not hasattr(_inner, "predict"):
                        continue
                    # Raw components: apply entry's pre_pipeline before
                    # predict. ``model_obj.pre_pipeline`` may be None for
                    # tree models (no preprocessing needed).
                    _pp = getattr(_entry, "pre_pipeline", None)
                    _name = f"raw#{_i}"
                    _components.append(
                        _PrePipelinePredictShim(_inner, _pp, _name)
                    )
                    _component_names.append(_name)
                for _spec in _spec_list:
                    _composite_entries = (models or {}).get(_tt_e, {}).get(
                        _spec["name"], []
                    ) or []
                    for _i, _entry in enumerate(_composite_entries):
                        _inner = getattr(_entry, "model", None) or _entry
                        if not hasattr(_inner, "predict"):
                            continue
                        # Composite entries: CompositeTargetEstimator
                        # wrappers already manage their own transform;
                        # pre_pipeline (if any) is the OUTER frame-prep
                        # that should also be applied. Same shim.
                        _pp = getattr(_entry, "pre_pipeline", None)
                        _name = f"{_spec['name']}#{_i}"
                        _components.append(
                            _PrePipelinePredictShim(_inner, _pp, _name)
                        )
                        _component_names.append(_name)
                if len(_components) < 2:
                    logger.info(
                        "[CompositeCrossTargetEnsemble] target='%s': only %d "
                        "component(s); ensemble skipped.",
                        _orig_tname, len(_components),
                    )
                    continue
                # Score every component on the train slice in y-scale.
                # Wrapped composite-target components predict in y-scale
                # via their inverse layer; raw-target components predict
                # y-scale directly. Use the same train rows the wrappers
                # were fitted against to keep the comparison fair.
                _y_full_for_rmse = target_by_type.get(_tt_e, {}).get(_orig_tname)
                _component_train_rmses: List[float] = []
                if _y_full_for_rmse is not None:
                    _y_train_for_rmse = np.asarray(_y_full_for_rmse)[filtered_train_idx]
                    for _comp, _name in zip(_components, _component_names):
                        try:
                            # I1 fix (2026-05-11): cache key is the INNER model id, not the shim id. The wrap pass populated ``_train_pred_cache`` keyed by the wrapper / inner ``id()``; the ensemble pass builds NEW shim instances so ``id(_comp)`` never hits. Look up via the inner instead and only fall back to ``id(_comp)`` for safety.
                            _inner_for_cache = getattr(_comp, "_model", _comp)
                            _pred = _train_pred_cache.get(id(_inner_for_cache))
                            if _pred is None:
                                _pred = _train_pred_cache.get(id(_comp))
                            if _pred is None:
                                _pred = np.asarray(
                                    _comp.predict(filtered_train_df),
                                    dtype=np.float64,
                                ).reshape(-1)
                                _train_pred_cache[id(_inner_for_cache)] = _pred
                            _diff = _pred - _y_train_for_rmse.astype(np.float64)
                            _component_train_rmses.append(
                                float(np.sqrt(np.mean(_diff * _diff)))
                            )
                        except Exception as _rmse_err:
                            logger.warning(
                                "[CompositeCrossTargetEnsemble] could not score "
                                "component '%s' on train: %s. Skipping in "
                                "ensemble weighting.", _name, _rmse_err,
                            )
                            _component_train_rmses.append(float("nan"))
                else:
                    _component_train_rmses = [float("nan")] * len(_components)
                _rmse_arr = np.asarray(_component_train_rmses, dtype=np.float64)
                _finite = np.isfinite(_rmse_arr)
                if _finite.sum() == 0:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] target='%s': no "
                        "component scored on train; ensemble skipped.",
                        _orig_tname,
                    )
                    continue
                if not _finite.all():
                    _rmse_arr[~_finite] = float(np.median(_rmse_arr[_finite]))
                # If oof_holdout_frac > 0, replace train-RMSE proxy
                # with honest holdout predictions: re-fit each
                # component on (1-frac) of train and predict on the
                # held-out frac. This is the correct objective for
                # ensemble weighting / stacking; the cost is one
                # extra fit per component.
                _oof_frac = float(getattr(
                    composite_target_discovery_config, "oof_holdout_frac", 0.0,
                ))
                _oof_y_full = _y_full_for_rmse
                _oof_pred_matrix = None
                _oof_y_holdout = None
                _oof_components = _components
                _oof_names = _component_names
                _oof_rmses = _rmse_arr  # train-RMSE proxy by default
                if _oof_frac > 0.0 and _oof_y_full is not None:
                    from ..composite import compute_oof_holdout_predictions
                    # Build per-spec base column on filtered_train_df
                    # rows (composite components need this for the
                    # transform.forward step inside the OOF helper).
                    _base_full_per_spec: Dict[str, np.ndarray] = {}
                    for _spec_for_oof in _spec_list:
                        _b = _build_full_column_from_splits(
                            _spec_for_oof["base_column"],
                            train_df_pd, val_df_pd, test_df_pd,
                            train_idx, val_idx, test_idx,
                            n_total=len(_oof_y_full),
                        )
                        _base_full_per_spec[_spec_for_oof["base_column"]] = (
                            _b[filtered_train_idx]
                        )
                    # Build the spec-or-None list parallel to components.
                    _component_specs: List[Optional[Dict[str, Any]]] = []
                    for _name in _component_names:
                        if _name.startswith("raw#"):
                            _component_specs.append(None)
                        else:
                            # Composite name format "{compname}#{i}".
                            _comp_name = _name.split("#", 1)[0]
                            _matching = next(
                                (s for s in _spec_list
                                 if s["name"] == _comp_name), None,
                            )
                            _component_specs.append(_matching)
                    try:
                        _oof_pred_matrix, _oof_y_holdout, _surviving = (
                            compute_oof_holdout_predictions(
                                component_models=_components,
                                component_names=_component_names,
                                component_specs=_component_specs,
                                train_X=filtered_train_df,
                                y_train_full=np.asarray(_oof_y_full)[filtered_train_idx],
                                base_train_full_per_spec=_base_full_per_spec,
                                holdout_frac=_oof_frac,
                                random_state=getattr(
                                    composite_target_discovery_config,
                                    "oof_random_state", 42,
                                ),
                            )
                        )
                    except Exception as _oof_err:
                        logger.warning(
                            "[CompositeCrossTargetEnsemble] OOF computation failed "
                            "for target='%s': %s. Falling back to train-RMSE proxy.",
                            _orig_tname, _oof_err,
                        )
                        _oof_pred_matrix, _oof_y_holdout, _surviving = (
                            None, None, [],
                        )
                    if _oof_pred_matrix is not None and _oof_pred_matrix.shape[1] > 0:
                        # Re-align components / names / rmses to the
                        # surviving set returned by the OOF helper.
                        _surviving_set = set(_surviving)
                        _oof_components = [
                            c for c, n in zip(_components, _component_names)
                            if n in _surviving_set
                        ]
                        _oof_names = list(_surviving)
                        # Compute holdout RMSE per surviving component.
                        _oof_rmses_list = []
                        for _i_col in range(_oof_pred_matrix.shape[1]):
                            _diff = _oof_pred_matrix[:, _i_col] - _oof_y_holdout
                            _finite = np.isfinite(_diff)
                            if _finite.sum() == 0:
                                _oof_rmses_list.append(float("nan"))
                            else:
                                _oof_rmses_list.append(float(np.sqrt(np.mean(
                                    _diff[_finite] * _diff[_finite]
                                ))))
                        _oof_rmses = np.asarray(_oof_rmses_list, dtype=np.float64)
                        logger.info(
                            "[CompositeCrossTargetEnsemble] target='%s' using "
                            "honest OOF holdout (frac=%.2f, n=%d) for ensemble "
                            "weights / stacking.",
                            _orig_tname, _oof_frac, len(_oof_y_holdout),
                        )

                try:
                    if _ce_strategy == "mean":
                        _ensemble = _CrossEns.from_uniform_weights(
                            component_models=_oof_components,
                            component_names=_oof_names,
                        )
                    elif _ce_strategy in ("linear_stack", "nnls_stack"):
                        # Use OOF holdout predictions if available
                        # (honest stacking), otherwise the train-set
                        # predictions (biased but always available).
                        if _oof_pred_matrix is not None and _oof_pred_matrix.shape[1] > 0:
                            _pred_matrix = _oof_pred_matrix
                            _y_for_stack = _oof_y_holdout
                        else:
                            _y_for_stack = (
                                np.asarray(_oof_y_full)[filtered_train_idx]
                                if _oof_y_full is not None else None
                            )
                            if _y_for_stack is None:
                                raise RuntimeError(
                                    "stacking requires train target alignment"
                                )
                            _pred_matrix_cols = []
                            for _comp, _name in zip(_oof_components, _oof_names):
                                # I1 (2026-05-11): inner-keyed cache lookup; see twin block above for the rationale. Shim-id keying never hit because shims are created per-pass.
                                _inner_for_cache = getattr(_comp, "_model", _comp)
                                _pred = _train_pred_cache.get(id(_inner_for_cache))
                                if _pred is None:
                                    _pred = _train_pred_cache.get(id(_comp))
                                if _pred is None:
                                    _pred = np.asarray(
                                        _comp.predict(filtered_train_df),
                                        dtype=np.float64,
                                    ).reshape(-1)
                                    _train_pred_cache[id(_inner_for_cache)] = _pred
                                _pred_matrix_cols.append(_pred)
                            _pred_matrix = np.column_stack(_pred_matrix_cols)
                        if _ce_strategy == "linear_stack":
                            _ensemble = _CrossEns.from_linear_stack(
                                component_models=_oof_components,
                                component_names=_oof_names,
                                component_predictions=_pred_matrix,
                                y_train=_y_for_stack,
                            )
                        else:  # nnls_stack
                            _ensemble = _CrossEns.from_nnls_stack(
                                component_models=_oof_components,
                                component_names=_oof_names,
                                component_predictions=_pred_matrix,
                                y_train=_y_for_stack,
                            )
                    else:  # "oof_weighted"
                        _ensemble = _CrossEns.from_train_metrics(
                            component_models=_oof_components,
                            component_names=_oof_names,
                            component_train_rmse=_oof_rmses.tolist(),
                            baseline_train_rmse=None,
                        )
                    # True OOF validation gate: if we have honest
                    # holdout predictions, compare ensemble holdout
                    # RMSE vs best single holdout RMSE. If the
                    # ensemble is worse, fall back to the best single.
                    if (_oof_pred_matrix is not None
                            and _oof_pred_matrix.shape[1] > 0
                            and isinstance(_ensemble, _CrossEns)):
                        try:
                            _ens_pred = _ensemble.predict(filtered_train_df)
                            # Use the SAME train rows used for OOF;
                            # ensemble.predict on filtered_train_df is
                            # in-sample for raw-target components but
                            # the comparison is component-fair (all
                            # components see the same X at same rows).
                            # NB: this is approximate -- the proper
                            # check would predict on stack_holdout. We
                            # do that next:
                            # Recompute ensemble preds on stack_holdout
                            # by weighted-combining the cached
                            # _oof_pred_matrix with the ensemble's
                            # weights.
                            _w_full = np.asarray(_ensemble.weights, dtype=np.float64)
                            if _ce_strategy == "linear_stack":
                                _intercept = float(getattr(
                                    _ensemble, "_linear_stack_intercept", 0.0,
                                ))
                                _ens_holdout = (
                                    (_oof_pred_matrix * _w_full[None, :]).sum(axis=1)
                                    + _intercept
                                )
                            else:
                                _w_norm = _w_full / max(_w_full.sum(), 1e-12)
                                _ens_holdout = (
                                    _oof_pred_matrix * _w_norm[None, :]
                                ).sum(axis=1)
                            _ens_diff = _ens_holdout - _oof_y_holdout
                            _ens_rmse = float(np.sqrt(np.mean(_ens_diff ** 2)))
                            _best_single_rmse = float(np.nanmin(_oof_rmses))
                            if _ens_rmse > _best_single_rmse:
                                _best_idx = int(np.nanargmin(_oof_rmses))
                                logger.warning(
                                    "[CompositeCrossTargetEnsemble] target='%s' "
                                    "honest OOF gate fired: ensemble RMSE %.4g > "
                                    "best single '%s' RMSE %.4g. Falling back to "
                                    "best single component.",
                                    _orig_tname, _ens_rmse,
                                    _oof_names[_best_idx], _best_single_rmse,
                                )
                                _ensemble = _oof_components[_best_idx]
                        except Exception as _gate_err:
                            logger.info(
                                "[CompositeCrossTargetEnsemble] OOF gate check "
                                "skipped (%s); ensemble retained.", _gate_err,
                            )
                except Exception as _ens_err:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] target='%s' build failed: "
                        "%s. Skipping.", _orig_tname, _ens_err,
                    )
                    continue
                # Optionally cap to the top-N components by weight
                # for online-latency-bounded serving. Configured via
                # ``max_inference_components``; 0 / None preserves
                # the full ensemble.
                _max_components = getattr(
                    composite_target_discovery_config,
                    "max_inference_components", None,
                )
                if (_max_components is not None and _max_components > 0
                        and isinstance(_ensemble, _CrossEns)):
                    _ensemble = _ensemble.cap_inference_components(
                        int(_max_components)
                    )
                # Wrap as a SimpleNamespace entry so downstream
                # iterators that expect ``.model`` / ``.columns`` keep
                # working. ``columns`` = union of inner columns; we
                # leave it empty -- the ensemble itself does not need
                # a fixed feature list (each component knows its own).
                from types import SimpleNamespace as _SN
                _ens_entry = _SN(
                    model=_ensemble,
                    model_name="CT_ENSEMBLE",
                    columns=None,
                    pre_pipeline=None,
                    metrics={},
                )
                # Dedicated key with prefix ``_CT_ENSEMBLE__`` so
                # downstream code that loops over composite-target
                # entries can trivially detect the ensemble entry and
                # skip / pick it.
                _ens_key = f"_CT_ENSEMBLE__{_orig_tname}"
                _by_name = models.setdefault(_tt_e, {})
                _by_name[_ens_key] = [_ens_entry]
                metadata.setdefault("composite_target_ensemble", {}) \
                    .setdefault(str(_tt_e), {})[_orig_tname] = (
                    _ensemble.export_metadata()
                    if hasattr(_ensemble, "export_metadata")
                    else {"strategy": "single_best_fallback"}
                )
                logger.info(
                    "[CompositeCrossTargetEnsemble] target='%s' built strategy='%s' "
                    "over %d component(s); stored at models[%s][%s].",
                    _orig_tname, _ce_strategy, len(_components),
                    _tt_e, _ens_key,
                )

                # 2026-05-12 (user request): route the cross-target ensemble
                # through the SAME ``report_model_perf`` pipeline that every
                # per-target model goes through. Previously the entry was
                # stored with ``metrics={}`` and no chart / log lines were
                # emitted, so users had no visual confirmation that the
                # ensemble was even built. Each split (val + test) gets a
                # scatter + residual chart + one-line metrics in the log,
                # using the SAME look as the real models. Guarded with a
                # broad try/except because an ensemble that has a
                # component shim that doesn't accept the suite's frame
                # shape would otherwise abort the whole suite.
                try:
                    from ..evaluation import report_model_perf
                    _ens_orig_y = target_by_type.get(_tt_e, {}).get(_orig_tname)
                    if _ens_orig_y is not None:
                        _ens_y_arr = np.asarray(_ens_orig_y)
                        _ens_model_name = (
                            f"CT_ENSEMBLE[{_ce_strategy}] {target_name} "
                            f"{model_name} {_orig_tname}"
                        )
                        _ens_columns = (
                            list(getattr(filtered_train_df, "columns", []) or [])
                        )
                        _ens_common = dict(
                            columns=_ens_columns,
                            df=None, model=None,
                            model_name=_ens_model_name,
                            plot_outputs=getattr(reporting_config, "plot_outputs", None),
                            plot_dpi=getattr(reporting_config, "plot_dpi", None),
                            show_fi=False,
                            target_type=str(_tt_e),
                        )
                        for _split_name, _report_title, _split_idx, _split_df in (
                            ("val", "VAL (CT_ENSEMBLE) ", filtered_val_idx, filtered_val_df),
                            ("test", "TEST (CT_ENSEMBLE) ", test_idx, test_df_pd),
                        ):
                            if _split_idx is None or _split_df is None:
                                continue
                            try:
                                _y_split = _ens_y_arr[_split_idx]
                                _ens_preds = np.asarray(
                                    _ensemble.predict(_split_df),
                                    dtype=np.float64,
                                ).reshape(-1)
                                _common_split = dict(_ens_common)
                                if plot_file:
                                    _common_split["plot_file"] = (
                                        f"{plot_file}_ct_ensemble_{_orig_tname}_{_split_name}"
                                    )
                                report_model_perf(
                                    targets=_y_split,
                                    preds=_ens_preds, probs=None,
                                    report_title=_report_title,
                                    **_common_split,
                                )
                            except Exception as _split_err:
                                logger.warning(
                                    "[CompositeCrossTargetEnsemble] target='%s' "
                                    "split='%s' report_model_perf failed: %s. "
                                    "Continuing without ensemble chart for this split.",
                                    _orig_tname, _split_name, _split_err,
                                )
                except Exception as _ens_report_err:
                    logger.warning(
                        "[CompositeCrossTargetEnsemble] target='%s' could not emit "
                        "scatter / log charts: %s. The ensemble entry is still "
                        "stored at models[%s][%s] for downstream consumers.",
                        _orig_tname, _ens_report_err, _tt_e, _ens_key,
                    )

    # 2026-05-10: suite-end dummy-baselines summary (D6) — cross-target
    # verdict block — canonical UPPERCASE WARN tokens.
    try:
        if metadata.get("dummy_baselines"):
            from ..dummy_baselines import format_suite_end_summary
            # Build {(target_type, target_name): {primary_metric: best_val,
            # "model_name": ...}} from the trained models. The model
            # metrics dict is keyed by metric NAME (e.g. "RMSE"); the
            # dummy primary_metric is split-prefixed (e.g. "val_RMSE").
            # Strip the "val_" prefix and look up via _entry_metric.
            _best_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for _tt, _by_name in metadata.get("dummy_baselines", {}).items():
                for _tname, _rep_dict in _by_name.items():
                    _pm = _rep_dict.get("primary_metric")
                    if not _pm or not _pm.startswith("val_"):
                        continue
                    _metric_name = _pm[len("val_"):]  # "val_RMSE" -> "RMSE"
                    _model_list = models.get(_tt, {}).get(_tname, [])
                    if not _model_list:
                        continue
                    # Pick best model by primary metric. Minimize for
                    # RMSE/MAE/log_loss/pinball; maximize for everything
                    # else (NDCG / AUC).
                    _is_minimize = (
                        "RMSE" in _metric_name or "MAE" in _metric_name
                        or "log_loss" in _metric_name or "pinball" in _metric_name
                    )
                    # For composite targets: prefer the y-scale model
                    # metric (post-inverse, comparable to raw / y-scale
                    # dummy) over the T-scale ``_entry_metric`` value
                    # that was computed during the per-target loop on
                    # the unwrapped inner model. The y-scale numbers
                    # live in metadata["composite_target_y_scale_metrics"]
                    # populated by the wrap pass at section 6.
                    _yscale_entries = (
                        metadata.get("composite_target_y_scale_metrics", {})
                        .get(str(_tt), {})
                        .get(_tname, [])
                    )
                    _best_val: Optional[float] = None
                    _best_name = "-"
                    if _yscale_entries:
                        # y-scale path: iterate stored entries.
                        for _ye in _yscale_entries:
                            _split_metric = _ye.get("metrics", {}).get("val", {})
                            _v = _split_metric.get(_metric_name)
                            if _v is None or not np.isfinite(_v):
                                continue
                            if (
                                _best_val is None
                                or (_is_minimize and _v < _best_val)
                                or (not _is_minimize and _v > _best_val)
                            ):
                                _best_val = float(_v)
                                _best_name = _ye.get("model_name") or "Composite"
                    else:
                        for _m in _model_list:
                            _v = _entry_metric(_m, "val", _metric_name)
                            if not np.isfinite(_v):
                                continue
                            if (
                                _best_val is None
                                or (_is_minimize and _v < _best_val)
                                or (not _is_minimize and _v > _best_val)
                            ):
                                _best_val = _v
                                _best_name = getattr(_m, "model_name", None) or type(
                                    getattr(_m, "model", _m)
                                ).__name__
                    if _best_val is not None:
                        _best_metrics[(str(_tt), str(_tname))] = {
                            _pm: _best_val,
                            "model_name": _best_name,
                        }
            # B2 (2026-05-11): build composite->raw target map so the verdict block uses the raw target's dummy (median(y_raw) constant) as the true trivial baseline -- not the inverted-T fake baseline that uses fitted alpha.
            _composite_to_raw: Dict[Tuple[str, str], str] = {}
            for _tt_str, _by_tname in metadata.get(
                "composite_target_specs", {}
            ).items():
                for _raw_tname, _spec_list in _by_tname.items():
                    for _s in _spec_list or []:
                        _comp_name = _s.get("name")
                        if _comp_name:
                            _composite_to_raw[(_tt_str, _comp_name)] = _raw_tname
            _summary_text = format_suite_end_summary(
                dummy_baselines_metadata=metadata.get("dummy_baselines", {}),
                failures_metadata=metadata.get("dummy_baselines_failures", {}),
                best_model_metrics_by_target=_best_metrics if _best_metrics else None,
                min_lift=dummy_baselines_config.best_model_min_lift,
                composite_to_raw_target_map=_composite_to_raw if _composite_to_raw else None,
            )
            if _summary_text:
                logger.info(_summary_text)
    except Exception as _db_summary_err:
        logger.warning(
            "[DUMMY_BASELINES] suite-end summary failed: %s",
            _db_summary_err,
        )

    # Release captured high-card column data — for typical well-log shapes
    # (5M rows * 1-2 dropped cols * 4-8 bytes per cell) this frees ~40-80MB
    # before the final metadata save / pickle path. Cheap memory hygiene.
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