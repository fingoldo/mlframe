"""
Phase 1: configuration setup.

Converts raw config dicts to Pydantic objects, resolves process-wide overrides,
pre-warms optional kernels, clears caches, and builds initial metadata.
"""
from __future__ import annotations

import logging
import os as _os
from typing import Any, Dict, List, Optional, Union

from ..configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    ConfidenceAnalysisConfig,
    DummyBaselinesConfig,
    FeatureSelectionConfig,
    FeatureTypesConfig,
    ModelHyperparamsConfig,
    OutlierDetectionConfig,
    OutputConfig,
    PreprocessingBackendConfig,
    PreprocessingConfig,
    QuantileRegressionConfig,
    ReportingConfig,
    TrainingBehaviorConfig,
    TrainingSplitConfig,
)
from ..phases import log_phase
from ._training_context import TrainingContext
from .utils import (
    _apply_plot_style_overrides,
    _build_suite_common_params_dict,
    _create_initial_metadata,
    _detect_dataset_reuse_capabilities,
    _ensure_config,
)

logger = logging.getLogger(__name__)


def setup_configuration(
    *,
    # Raw config parameters (dicts or Pydantic objects)
    preprocessing_config,
    pipeline_config,
    feature_types_config,
    split_config,
    hyperparams_config,
    behavior_config,
    reporting_config,
    output_config,
    outlier_detection_config,
    feature_selection_config,
    confidence_analysis_config,
    baseline_diagnostics_config,
    dummy_baselines_config,
    quantile_regression_config,
    composite_target_discovery_config,
    feature_handling_config,
    # Other inputs
    model_name: str,
    target_name: str,
    mlframe_models: Optional[List[str]],
    verbose: int,
):
    """Convert and validate all configs, return processed state dict.

    Returns a dict with all processed configs, scalars, and derived values
    consumed by the downstream orchestrator.
    """
    if verbose:
        log_phase(f"Starting mlframe training suite: {model_name}")

    # Phase Q: FeatureHandlingConfig validation (consumer wiring lands in F-J phases)
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

    # Convert dict configs to Pydantic
    preprocessing_config = _ensure_config(preprocessing_config, PreprocessingConfig, {})
    pipeline_config = _ensure_config(pipeline_config, PreprocessingBackendConfig, {})
    feature_types_config = _ensure_config(feature_types_config, FeatureTypesConfig, {})
    split_config = _ensure_config(split_config, TrainingSplitConfig, {})
    hyperparams_config = _ensure_config(hyperparams_config, ModelHyperparamsConfig, {})
    behavior_config = _ensure_config(behavior_config, TrainingBehaviorConfig, {})
    reporting_config = _ensure_config(reporting_config, ReportingConfig, {})

    # Propagate residual-audit toggle from behavior_config into evaluation module.
    # Module-level override; subsequent stand-alone calls keep the historical default
    # since the override only flips while a suite run is in progress.
    from ..evaluation import _set_residual_audit_enabled as _set_resid_audit
    _set_resid_audit(getattr(behavior_config, "report_residual_audit", True))

    # Honor ReportingConfig.plot_inline_display via process-wide env var.
    # None = clear override (auto-detect via __IPYTHON__ / sys.ps1); True/False = explicit.
    try:
        from mlframe.reporting.renderers.save import set_inline_display_mode as _set_idm
        _set_idm(getattr(reporting_config, "plot_inline_display", None))
    except Exception:
        pass

    # Apply matplotlib style + rcParams + plotly template overrides from reporting_config.
    # Process-wide; None keeps the user's pre-suite settings intact.
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
    baseline_diagnostics_config = _ensure_config(baseline_diagnostics_config, BaselineDiagnosticsConfig, {})
    dummy_baselines_config = _ensure_config(dummy_baselines_config, DummyBaselinesConfig, {})
    quantile_regression_config = _ensure_config(quantile_regression_config, QuantileRegressionConfig, {})

    # Pre-warm dummy_baselines numba kernels so first call doesn't pay 6-10s JIT cold-start.
    if dummy_baselines_config.enabled:
        try:
            from ..dummy_baselines import _warmup_numba_kernels
            _warmup_numba_kernels()
        except Exception:
            pass

    composite_target_discovery_config = _ensure_config(
        composite_target_discovery_config, CompositeTargetDiscoveryConfig, {}
    )
    # Production kill-switch: MLFRAME_DISABLE_COMPOSITE env var overrides to OFF.
    if _os.environ.get("MLFRAME_DISABLE_COMPOSITE", "").lower() in {"1", "true", "yes"}:
        composite_target_discovery_config = _ensure_config(
            {"enabled": False}, CompositeTargetDiscoveryConfig, {}
        )
        logger.info("[CompositeTargetDiscovery] disabled by MLFRAME_DISABLE_COMPOSITE env var.")

    # Pull scalar fields out of typed configs for downstream code that takes plain locals.
    data_dir = output_config.data_dir
    models_dir = output_config.models_dir
    save_charts = output_config.save_charts

    # Surface reporting-knob resolution at suite entry so the reader knows up-front
    # whether plots will be saved / rendered / short-circuited.
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
        # When short-circuit is active, clear plot_file so cal-plot rendering is skipped
        # entirely. Previously the suite still rendered 100+ calibration plots to a temp
        # dir that was immediately discarded (~60s wasted on 1M-row multiclass).
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
        # Perf advisory: plotly[png] via kaleido spawns ~12-15s per chart on Chromium
        # reload. On large datasets this can dominate wall-time by minutes.
        try:
            _po = getattr(reporting_config, "plot_outputs", "") or ""
        except NameError:
            _po = ""
        if save_charts and "plotly" in _po and "png" in _po:
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

    # Extract scalar feature-selection knobs
    outlier_detector = outlier_detection_config.detector
    od_val_set = outlier_detection_config.apply_to_val
    use_mrmr_fs = feature_selection_config.use_mrmr_fs
    mrmr_kwargs = feature_selection_config.mrmr_kwargs
    rfecv_models = feature_selection_config.rfecv_models
    custom_pre_pipelines = feature_selection_config.custom_pre_pipelines if feature_selection_config.custom_pre_pipelines else None

    # Build common_params_dict: ferries ReportingConfig + PreprocessingConfig scaler/imputer/
    # category_encoder + ConfidenceAnalysisConfig fields down to dict-key consumers in trainer.py.
    common_params_dict = _build_suite_common_params_dict(
        reporting_config=reporting_config,
        preprocessing_config=preprocessing_config,
        confidence_analysis_config=confidence_analysis_config,
    )

    # Opt-in: install SIGSEGV/SIGABRT handler so native crashes surface as Python tracebacks
    if behavior_config.enable_crash_reporting:
        from mlframe.training.crash_reporting import enable_crash_reporting as _enable_crash_reporting
        _enable_crash_reporting()

    # Report dataset-reuse capability of installed GBDT libraries
    _dataset_reuse_caps = _detect_dataset_reuse_capabilities()
    logger.info("Dataset-reuse capabilities: %s", _dataset_reuse_caps)
    if not _dataset_reuse_caps.get("cb_pool_label_swap"):
        logger.warning(
            "  CatBoost Pool.set_label/set_weight not available in installed build -- "
            "mlframe will fall back to rebuilding the Pool on every weight schema and "
            "same-type target. Upgrade CatBoost to pick up the Pool label-swap PR."
        )

    # Clear process-wide CB Pool cache at every suite entry to prevent stale Pool reuse
    # when Python reuses ids across independent suite invocations.
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

    # Initial metadata
    metadata = _create_initial_metadata(
        model_name=model_name,
        target_name=target_name,
        mlframe_models=mlframe_models,
        preprocessing_config=preprocessing_config,
        pipeline_config=pipeline_config,
        split_config=split_config,
    )
    metadata["schema_version"] = 2  # v2 adds composite_target_specs, baseline_diagnostics

    ctx = TrainingContext(
        model_name=model_name,
        target_name=target_name,
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
        data_dir=data_dir,
        models_dir=models_dir,
        save_charts=save_charts,
        outlier_detector=outlier_detector,
        od_val_set=od_val_set,
        use_mrmr_fs=use_mrmr_fs,
        mrmr_kwargs=mrmr_kwargs,
        rfecv_models=rfecv_models,
        custom_pre_pipelines=custom_pre_pipelines,
        common_params_dict=common_params_dict,
        mlframe_models=mlframe_models,
        metadata=metadata,
    )
    return ctx
