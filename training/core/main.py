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
    _apply_plot_style_overrides,
    _auto_detect_feature_types,
    _augment_with_dropped_high_card_cols,
    _build_common_params_for_target,
    _build_full_column_from_splits,
    _build_pre_pipelines,
    _build_process_model_kwargs,
    _build_suite_common_params_dict,
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
    # 2026-05-13 (user request): apply matplotlib style + rcParams +
    # plotly template overrides from reporting_config. Process-wide;
    # ``None`` keeps the user's pre-suite settings intact.
    _apply_plot_style_overrides(
        matplotlib_style=getattr(reporting_config, "matplotlib_style", None),
        matplotlib_rcparams=getattr(reporting_config, "matplotlib_rcparams", None),
        plotly_template=getattr(reporting_config, "plotly_template", None),
        verbose=bool(verbose),
    )
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
        # 2026-05-12 Wave 31: _short_circuit_active was previously ONLY logged
        # ("ACTIVE (renders skipped)") but the flag was NEVER passed to the
        # rendering code. The suite still rendered 100+ calibration plots and
        # saved them to a temp dir that was immediately discarded. On a 1M-row
        # multiclass combo (c0124), this wasted 42 show_calibration_plot calls
        # × 576 ms = 24 s, plus 46 savefig calls × 383 ms = 17.6 s, plus
        # 18 s of thread-lock contention from the parallel chart save threads.
        # All ~60 s of rendering work was thrown away.
        #
        # Fix: when short-circuit is active, clear the model-level plot_file
        # so show_calibration_plot's ``show_plots and not plot_file`` guard
        # (metrics.py:1559) fires and returns immediately. The output_config
        # is not mutated (the string "" is semantic, not a path mutation).
        if _short_circuit_active:
            output_config.plot_file = ""
            logger.info(
                "[reporting] save_charts=%s, interactive=%s -- "
                "cal-plot short-circuit ACTIVE: clearing plot_file so "
                "chart rendering is skipped entirely",
                save_charts, _is_interactive_logp,
            )
        else:
            logger.info(
                "[reporting] save_charts=%s, plot_dir=%s, interactive=%s -- "
                "cal-plot short-circuit INACTIVE (charts will render)",
                save_charts, _plot_dir, _is_interactive_logp,
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

    # 2026-05-12: build common_params_dict (the dict that ferries ReportingConfig
    # + PreprocessingConfig.{scaler,imputer,category_encoder} + ConfidenceAnalysisConfig
    # fields down to the deep dict-key consumers in trainer.py) via a dedicated
    # helper. Pure read-only extraction of values from the three configs.
    common_params_dict = _build_suite_common_params_dict(
        reporting_config=reporting_config,
        preprocessing_config=preprocessing_config,
        confidence_analysis_config=confidence_analysis_config,
    )

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
                                from ..composite_transforms import is_composite_target_name
                                _dummy_is_composite = is_composite_target_name(cur_target_name)
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

                # 2026-05-13 (user request): skip identity-equivalent
                # pre_pipelines when an ordinary (no-pipeline) branch or
                # another identity-equivalent selector already covered this
                # target. The marker is set by _apply_pre_pipeline_transforms
                # after the first fit-transform.  If the same MRMR/RFECV
                # instance was identity on a previous target, the marker
                # survives → skip before any model trains.
                _pp_name_stripped = pre_pipeline_name.strip()
                if _pp_name_stripped and getattr(
                    pre_pipeline, "_mlframe_identity_equivalent", False
                ):
                    logger.info(
                        "[Dedup] Skipping pre_pipeline '%s' -- "
                        "identity-equivalent to ordinary (cached from "
                        "prior target/iteration); models already covered.",
                        _pp_name_stripped,
                    )
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
                    key=lambda m: (
                        is_neural_model(m),  # push neural to end
                        tuple(
                            -int(t) for t in strategy_by_model[id(m)].feature_tier()
                        ),  # rich-feature models first
                    ),
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

                # 2026-05-13 (user request): neural max_time defaults to
                # the 95th percentile of non-neural model train times so
                # MLP doesn't run 2h while boosting models finish in 5min.
                # Accumulated as each non-neural model completes.
                _non_neural_train_times: List[float] = []

                _total_models_in_run = len(list(sorted_models))
                _model_idx_in_run = 0
                _break_model_loop = False
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

                        # 2026-05-13 (user request): neural-model max_time
                        # defaults to the P95 of all prior non-neural model
                        # train times (seconds). If no non-neural model has
                        # trained yet, keep the config-level max_time (2h).
                        _is_neural = is_neural_model(mlframe_model_name)
                        if _is_neural and _non_neural_train_times:
                            _p95 = float(np.percentile(_non_neural_train_times, 95))
                            # Floor: 5 min (300 s) so LightGBM/CB don't
                            # produce a sub-minute P95 that rounds to
                            # 0h0m → Lightning stops immediately.
                            _max_sec = max(int(round(_p95)), 300)
                            _dd = _max_sec // 86400
                            _hh = (_max_sec % 86400) // 3600
                            _mm = (_max_sec % 3600) // 60
                            _ss = _max_sec % 60
                            _max_time_dict = {"days": _dd, "hours": _hh, "minutes": _mm, "seconds": _ss}
                            # MLP is Pipeline(StandardScaler, TTR(PytorchLightningRegressor(...)))
                            _neural_model = current_model_params.get("model")
                            if _neural_model is not None:
                                _inner = getattr(_neural_model, "regressor", None)
                                if _inner is None and hasattr(_neural_model, "named_steps"):
                                    for _step in _neural_model.named_steps.values():
                                        if hasattr(_step, "regressor"):
                                            _inner = _step.regressor
                                            break
                                if _inner is not None and hasattr(_inner, "trainer_params"):
                                    _inner.trainer_params["max_time"] = _max_time_dict
                                    if verbose:
                                        logger.info(
                                            "  [NeuralTimeout] %s max_time=%dh%02dm%02ds "
                                            "(P95 of %d prior non-neural train times: %.0fs)",
                                            mlframe_model_name, _hh, _mm, _ss,
                                            len(_non_neural_train_times), _p95,
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
                        if not _is_neural and t0_model is not None:
                            _non_neural_train_times.append(timer() - t0_model)

                        # 2026-05-13 (user request): after the FIRST model
                        # under this pre_pipeline completes, check whether
                        # the pre_pipeline is identity-equivalent (kept all
                        # columns, created none).  If it is AND ordinary
                        # models are in the suite, every remaining model
                        # would train on identical data → skip.
                        if (
                            _model_idx_in_run == 1
                            and _pp_name_stripped
                            and use_ordinary_models
                            and feature_selection_config.skip_identity_equivalent_pre_pipelines
                            and getattr(
                                pre_pipeline, "_mlframe_identity_equivalent", False
                            )
                        ):
                            _skip_remaining = _total_models_in_run - 1
                            if _skip_remaining > 0:
                                logger.info(
                                    "[Dedup] pre_pipeline '%s' is "
                                    "identity-equivalent to ordinary (kept "
                                    "all %d columns); skipping remaining "
                                    "%d model(s) for this target.",
                                    _pp_name_stripped,
                                    train_df_transformed.shape[1]
                                    if train_df_transformed is not None
                                    else 0,
                                    _skip_remaining,
                                )
                            _break_model_loop = True
                            break  # exit weight_schema loop

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
                    # 2026-05-13: if the first model under this pre_pipeline
                    # was identity-equivalent, break the model loop now.
                    if _break_model_loop:
                        break

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
    # 6-7. COMPOSITE POST-PROCESSING (wrapping + cross-target ensemble + suite-end summary) -- extracted helper
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
        plot_file=plot_file,
        verbose=verbose,
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
