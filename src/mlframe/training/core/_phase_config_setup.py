"""Configuration setup: convert raw config dicts to Pydantic, resolve process-wide overrides, pre-warm kernels, build initial metadata."""
from __future__ import annotations

import logging
import os as _os
import sys
from typing import Any

from ..configs import (
    BaselineDiagnosticsConfig,
    CompositeTargetDiscoveryConfig,
    ConfidenceAnalysisConfig,
    ConformalConfig,
    DummyBaselinesConfig,
    FeatureSelectionConfig,
    FeatureTypesConfig,
    ModelHyperparamsConfig,
    OutlierDetectionConfig,
    OutputConfig,
    PreprocessingBackendConfig,
    PreprocessingConfig,
    QuantileRegressionConfig,
    RegressionCalibrationConfig,
    ReportingConfig,
    TrainingBehaviorConfig,
    TrainingSplitConfig,
)
from ..utils import log_phase
from ._training_context import TrainingContext
from .utils import (
    _apply_plot_style_overrides,
    _build_suite_common_params_dict,
    _create_initial_metadata,
    _detect_dataset_reuse_capabilities,
    _ensure_config,
)

logger = logging.getLogger(__name__)


def _detect_interactive_mode() -> bool:
    """Detect IPython/REPL session once at module import.

    Probing on every suite invocation was wasted work: the environment doesn't switch
    between interactive and non-interactive between back-to-back calls in the same
    process. Cached so the kaleido / cal-plot short-circuit check is a constant-time
    attribute read.
    """
    try:
        return bool(__IPYTHON__)  # type: ignore[name-defined]
    except NameError:
        return hasattr(sys, "ps1")


_MLFRAME_INTERACTIVE = _detect_interactive_mode()


def _heavy_libs_needed(mlframe_models, recurrent_models, reporting_config) -> bool:
    """True when this run will actually reach the neural / SHAP import stack.

    The prewarm imports lightning, which pulls torchmetrics -> transformers -> a ``find_spec("tensorflow")``
    probe, and separately shap. A production fit with ``mlframe_models=['cb']`` spent 382.73s on that for
    models it never builds. Any recurrent model, any neural tag in the model list, or SHAP being enabled
    means the cost is real work rather than waste; an unreadable config answers True, because paying the
    import needlessly is a slow run while skipping it wrongly is a slow first fit plus a confusing profile.
    """
    try:
        if recurrent_models:
            return True
        if mlframe_models:
            from ..models import is_neural_model

            if any(is_neural_model(m) for m in mlframe_models):
                return True
        return bool(getattr(reporting_config, "use_shap", False))
    except Exception as exc:
        logger.debug("could not decide whether heavy libs are needed (%s); warming them", exc)
        return True


def setup_configuration(
    *,
    preprocessing_config: Any,
    pipeline_config: Any,
    feature_types_config: Any,
    split_config: Any,
    hyperparams_config: Any,
    behavior_config: Any,
    reporting_config: Any,
    output_config: Any,
    outlier_detection_config: Any,
    feature_selection_config: Any,
    confidence_analysis_config: Any,
    baseline_diagnostics_config: Any,
    dummy_baselines_config: Any,
    quantile_regression_config: Any,
    conformal_config: Any = None,
    regression_calibration_config: Any = None,
    composite_target_discovery_config: Any,
    feature_handling_config: Any,
    model_name: str,
    target_name: str,
    mlframe_models: list[str] | None,
    # Read only to decide whether the neural import stack is worth prewarming; the suite sets ctx.recurrent_models itself.
    recurrent_models: list[str] | None = None,
    verbose: int,
    # These get plumbed into the TrainingContext so dispatchers downstream
    # (LTR ranker-suite, pre-pipeline builder, ensemble-builder) see the
    # caller's intent. Defaults match the public-API defaults in
    # train_mlframe_models_suite so older call-sites that don't pass them
    # explicitly preserve current behaviour.
    ranking_config: Any = None,
    # use_mlframe_ensembles default aligned to True to match the public-API default in
    # train_mlframe_models_suite (main.py:96). Prior default was False, which would have
    # flipped behaviour for any caller of setup_configuration that omitted the kwarg.
    # main.py:258 currently always passes the value explicitly so the mismatch was dormant.
    use_mlframe_ensembles: bool = True,
    use_ordinary_models: bool = True,
    # Same silently-dropped-kwarg bug class as ``verbose`` (fixed 7479b54): the public
    # train_mlframe_models_suite accepts both of these but the value never reached ctx,
    # so the per-target reads at _phase_train_one_target.py:1051 and :1061 always saw
    # the dataclass default ``None``. Threaded through here and assigned on ctx below.
    linear_model_config: Any = None,
    multilabel_dispatch_config: Any = None,
) -> TrainingContext:
    """Convert and validate all configs, return processed state dict."""
    if verbose:
        log_phase(f"Starting mlframe training suite: {model_name}")

    # Setup-timing scaffolding. The user's observed 5-13 minute zero-CPU gap
    # between the suite-start log and the cal-plot short-circuit log lives
    # entirely between this point and the early-INFO block below. The likely
    # candidates (matplotlib font cache rebuild, plotly lazy template fetch,
    # ``mlframe.training.evaluation`` first-import cascade, AV scan on each
    # .pyc) are indistinguishable from outside; one timing log per major step
    # turns the black box into a labelled punch-list.
    #
    # Opt-out: MLFRAME_SETUP_TIMING=0 (default ON; cost ~5 us per checkpoint
    # via time.perf_counter, fully amortised at the first slow step).
    import time as _time
    _setup_timing_on = _os.environ.get("MLFRAME_SETUP_TIMING", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )
    _setup_t_prev = _time.perf_counter()
    _setup_t_start = _setup_t_prev

    def _step_done(label: str) -> None:
        """Log the elapsed time since the previous checkpoint under ``label``, promoting slow (>=2s) steps to INFO so they surface in prod logs."""
        nonlocal _setup_t_prev
        if not _setup_timing_on or not verbose:
            return
        _now = _time.perf_counter()
        _delta = _now - _setup_t_prev
        # Promote to INFO when a step is genuinely slow (>= 2s) so the loud
        # checkpoints surface in the prod log; quick steps stay at DEBUG to
        # keep the suite-start banner uncluttered on fast hosts.
        _lvl = logging.INFO if _delta >= 2.0 else logging.DEBUG
        logger.log(
            _lvl,
            "[setup-timing] %s in %.2fs (cumulative %.2fs)",
            label, _delta, _now - _setup_t_start,
        )
        _setup_t_prev = _now

    if feature_handling_config is not None:
        try:
            from mlframe.training.feature_handling import FeatureHandlingConfig
            if isinstance(feature_handling_config, FeatureHandlingConfig):
                if mlframe_models:
                    feature_handling_config.validate_against_models(list(mlframe_models))
                if verbose:
                    logger.info(
                        "[fhc] FeatureHandlingConfig active; resolved plan: %s",
                        feature_handling_config.describe(short=True),
                    )
        except ImportError:  # pragma: no cover
            pass
    _step_done("feature_handling_config validate")

    preprocessing_config = _ensure_config(preprocessing_config, PreprocessingConfig, {})
    pipeline_config = _ensure_config(pipeline_config, PreprocessingBackendConfig, {})
    feature_types_config = _ensure_config(feature_types_config, FeatureTypesConfig, {})
    split_config = _ensure_config(split_config, TrainingSplitConfig, {})
    hyperparams_config = _ensure_config(hyperparams_config, ModelHyperparamsConfig, {})
    behavior_config = _ensure_config(behavior_config, TrainingBehaviorConfig, {})
    reporting_config = _ensure_config(reporting_config, ReportingConfig, {})
    _step_done("_ensure_config x7 (preprocessing..reporting)")

    # Publish the PipelineCache RAM-budget fraction to the env the cache
    # reads (both PipelineCache.__init__ and the eviction re-check resolve
    # from it, so one source keeps them consistent). An explicit operator env
    # wins over the config default.
    _cache_frac = getattr(behavior_config, "pipeline_cache_ram_budget_fraction", None)
    if _cache_frac is not None and not _os.environ.get("MLFRAME_PIPELINE_CACHE_RAM_FRACTION") and not _os.environ.get("MLFRAME_PIPELINE_CACHE_BYTES_LIMIT"):
        _os.environ["MLFRAME_PIPELINE_CACHE_RAM_FRACTION"] = str(float(_cache_frac))

    # Module-level overrides for residual_audit + inline_display.
    # Pre-fix the leading comment promised "restored after the suite finishes" but no restore
    # call site existed anywhere -- the flag stayed flipped for the rest of the process and
    # leaked into the next suite call that didn't override it. Snapshot prior values onto
    # transient locals, stash on ctx via artifacts (set below) so _phase_finalize can restore.
    from ..evaluation import (
        _set_residual_audit_enabled as _set_resid_audit,
        _get_residual_audit_enabled as _get_resid_audit,
    )
    _step_done("import mlframe.training.evaluation")
    _residual_audit_prior = _get_resid_audit()
    _set_resid_audit(getattr(behavior_config, "report_residual_audit", True))

    # None = clear override (auto-detect via __IPYTHON__ / sys.ps1); True/False = explicit.
    # Only import the renderers.save module when the caller actually set a non-None value.
    # The import triggers the mlframe.reporting -> renderers chain (~12ms on cold-start)
    # which is pure overhead on suites that never touch charts (plot_outputs='matplotlib[png]'
    # + save_charts=False).
    _inline_display = getattr(reporting_config, "plot_inline_display", None)
    _inline_display_prior_set = False  # whether we successfully captured a prior to restore
    _inline_display_prior = None
    if _inline_display is not None:
        try:
            from mlframe.reporting.renderers.save import (
                set_inline_display_mode as _set_idm,
                get_inline_display_mode as _get_idm,
            )
            try:
                _inline_display_prior = _get_idm()
                _inline_display_prior_set = True
            except (AttributeError, NameError):
                # Older renderers.save without get_inline_display_mode: skip restore (best-effort).
                pass
            _set_idm(_inline_display)
        except ImportError:
            pass
    # Same set-and-restore shape as the inline-display override above, and the same reason for the thread-local:
    # two suites running concurrently must not flip each other's output layout mid-run.
    _subfolders = getattr(reporting_config, "plot_format_subfolders", None)
    _subfolders_prior_set = False
    _subfolders_prior = None
    if _subfolders is not None:
        try:
            from mlframe.reporting.renderers.save import (
                get_format_subfolders as _get_fsf,
                set_format_subfolders as _set_fsf,
            )
            _subfolders_prior = _get_fsf()
            _subfolders_prior_set = True
            _set_fsf(_subfolders)
        except (ImportError, AttributeError):
            pass
    # Same set-and-restore shape again, for the reliability diagram's colour scale.
    _calib_cmap = getattr(reporting_config, "calibration_colormap", None)
    _calib_cmap_prior_set = False
    _calib_cmap_prior = None
    if _calib_cmap is not None:
        try:
            from mlframe.reporting.colors import (
                get_calibration_cmap_override as _get_ccm,
                set_calibration_cmap as _set_ccm,
            )
            _calib_cmap_prior = _get_ccm()
            _calib_cmap_prior_set = True
            _set_ccm(_calib_cmap)
        except (ImportError, AttributeError):
            pass
    # One run's chart-timing table must describe one run's charts, and the registry behind it is process-wide.
    # Reset here rather than at the facade so a caller that builds its own context still gets a clean table.
    try:
        from mlframe.reporting.renderers import reset_chart_timings

        from .._dataset_build_stats import reset_dataset_build_stats

        reset_chart_timings()
        reset_dataset_build_stats()
    except (ImportError, AttributeError):
        pass
    _step_done("inline_display setup (import mlframe.reporting.renderers.save)")

    # Process-wide; None keeps the user's pre-suite matplotlib/plotly settings intact.
    _apply_plot_style_overrides(
        matplotlib_style=getattr(reporting_config, "matplotlib_style", None),
        matplotlib_rcparams=getattr(reporting_config, "matplotlib_rcparams", None),
        plotly_template=getattr(reporting_config, "plotly_template", None),
        verbose=bool(verbose),
    )
    _step_done("_apply_plot_style_overrides (matplotlib + plotly first import / fontcache)")

    output_config = _ensure_config(output_config, OutputConfig, {})
    outlier_detection_config = _ensure_config(outlier_detection_config, OutlierDetectionConfig, {})
    feature_selection_config = _ensure_config(feature_selection_config, FeatureSelectionConfig, {})
    confidence_analysis_config = _ensure_config(confidence_analysis_config, ConfidenceAnalysisConfig, {})
    baseline_diagnostics_config = _ensure_config(baseline_diagnostics_config, BaselineDiagnosticsConfig, {})
    dummy_baselines_config = _ensure_config(dummy_baselines_config, DummyBaselinesConfig, {})
    quantile_regression_config = _ensure_config(quantile_regression_config, QuantileRegressionConfig, {})
    conformal_config = _ensure_config(conformal_config, ConformalConfig, {})
    regression_calibration_config = _ensure_config(regression_calibration_config, RegressionCalibrationConfig, {})
    _step_done("_ensure_config x9 (output..regression_calibration)")

    # Pre-warm numba kernels so the first call does not pay the JIT cold-start.
    #
    # The feature-selection kernel set is by far the expensive half and is only worth compiling when
    # feature selection will actually run: a CatBoost-only fit with no MRMR and no RFECV never calls one of
    # those kernels, so warming them is pure wall time before any data is read.
    _fs_will_run = bool(getattr(feature_selection_config, "use_mrmr_fs", False)) or bool(getattr(feature_selection_config, "rfecv_models", None))
    # The heavy-lib half (lightning -> torchmetrics -> transformers -> a tensorflow probe, plus shap) is worth
    # importing only when this run will actually reach it. A production fit with mlframe_models=['cb'] spent
    # 382.73s warming a neural stack it never touched. SHAP counts too: the trainer imports it when use_shap.
    _heavy_will_run = _heavy_libs_needed(mlframe_models, recurrent_models, reporting_config)
    if dummy_baselines_config.enabled:
        try:
            from ..baselines import _warmup_numba_kernels
            _warmup_numba_kernels(include_feature_selection=_fs_will_run, include_heavy_libs=_heavy_will_run)
        except Exception as e:  # nosec B110 - optional dependency import guard
            logger.debug("numba kernel warm-up failed, first real call will pay JIT cost: %s", e)
    _step_done("_warmup_numba_kernels (JIT prewarm)")

    composite_target_discovery_config = _ensure_config(composite_target_discovery_config, CompositeTargetDiscoveryConfig, {})
    _step_done("_ensure_config(CompositeTargetDiscoveryConfig)")
    if _os.environ.get("MLFRAME_DISABLE_COMPOSITE", "").lower() in {"1", "true", "yes"}:
        composite_target_discovery_config = _ensure_config({"enabled": False}, CompositeTargetDiscoveryConfig, {})
        logger.info("[CompositeTargetDiscovery] disabled by MLFRAME_DISABLE_COMPOSITE env var.")

    data_dir = output_config.data_dir
    models_dir = output_config.models_dir
    save_charts = output_config.save_charts

    # Render-gating is a behavioral kill-switch, NOT a logging concern: a verbose=0 run with
    # save_charts=False and a stale plot_file would otherwise render 100+ charts nobody sees
    # (~60s wasted on 1M-row multiclass). Compute + mutate unconditionally; only the INFO lines
    # below stay gated on verbose. Cached interactive flag doesn't switch between calls in-process.
    _is_interactive_logp = _MLFRAME_INTERACTIVE
    _short_circuit_active = (not _is_interactive_logp) and not save_charts
    if _short_circuit_active:
        output_config.plot_file = ""

    if verbose:
        # Must match ``_setup_helpers.setup_directories``, which writes charts to
        # ``<data_dir>/charts/<target_name>/<model_name>/<target_type>/<cur_target_name>/`` (every segment
        # slugified). The previous string named ``<data_dir>/<models_dir>/<model_name>`` -- the MODELS
        # directory, unslugified, missing the target_name segment entirely -- so anyone who followed this
        # log line looked in a directory charts are never written to and concluded rendering had failed.
        # target_type / cur_target_name are only known inside the per-target loop, hence the trailing "...".
        from mlframe.training.core._setup_helpers import slugify as _slugify

        _plot_dir = f"{data_dir}/charts/{_slugify(target_name)}/{_slugify(model_name)}/..." if data_dir and save_charts else "(no save)"
        if _short_circuit_active:
            logger.info(
                "[reporting] save_charts=%s, interactive=%s -- " "cal-plot short-circuit ACTIVE: clearing plot_file so " "chart rendering is skipped entirely",
                save_charts,
                _is_interactive_logp,
            )
        else:
            logger.info(
                "[reporting] save_charts=%s, plot_dir=%s, interactive=%s -- "
                "cal-plot short-circuit INACTIVE (charts will render)",
                save_charts, _plot_dir, _is_interactive_logp,
            )
        try:
            _po = getattr(reporting_config, "plot_outputs", "") or ""
        except NameError:
            _po = ""
        # Warn ONLY when the PLOTLY backend itself emits a kaleido raster (png/svg/pdf) -- NOT when the
        # raster comes from matplotlib. The old ``"plotly" in _po and "png" in _po`` test false-fired on
        # the DEFAULT ``plotly[html] + matplotlib[png]`` (html via plotly, png via matplotlib, ZERO
        # kaleido), scaring operators into thinking kaleido dominated wall-time when it never ran.
        _plotly_kaleido = False
        if save_charts and _po:
            try:
                from mlframe.reporting.output import parse_plot_output_dsl as _parse_po
                for _bk, _fmts in _parse_po(_po).backends:
                    if _bk == "plotly" and (set(_fmts) & {"png", "svg", "pdf"}):
                        _plotly_kaleido = True
                        break
            except Exception as e:  # -- parse failure -> fall back to the substring heuristic
                logger.debug("plotly output-format parse failed, using substring heuristic: %s", e)
                _plotly_kaleido = "plotly" in _po and ("png" in _po or "svg" in _po or "pdf" in _po)
        if _plotly_kaleido:
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
    # The suite-level flag turns on MRMR's usability second pass
    # so transform() materialises the UNION of all three selection lists (pure-MI + linear + universal),
    # putting the linearly-usable engineered interaction in every model's input. An explicit mrmr_kwargs
    # entry wins; the default path (flag off) leaves mrmr_kwargs untouched / byte-identical.
    if getattr(feature_selection_config, "mrmr_usability_aware_lists", False):
        mrmr_kwargs = dict(mrmr_kwargs or {})
        mrmr_kwargs.setdefault("usability_aware_lists", True)
    rfecv_models = feature_selection_config.rfecv_models
    custom_pre_pipelines = feature_selection_config.custom_pre_pipelines if feature_selection_config.custom_pre_pipelines else None

    common_params_dict = _build_suite_common_params_dict(
        reporting_config=reporting_config,
        preprocessing_config=preprocessing_config,
        confidence_analysis_config=confidence_analysis_config,
        behavior_config=behavior_config,
    )

    if behavior_config.enable_crash_reporting:
        from mlframe.training.crash_reporting import enable_crash_reporting as _enable_crash_reporting
        _enable_crash_reporting()

    _dataset_reuse_caps = _detect_dataset_reuse_capabilities()
    logger.info("Dataset-reuse capabilities: %s", _dataset_reuse_caps)
    if not _dataset_reuse_caps.get("cb_pool_label_swap"):
        # Name the method that is ACTUALLY missing. The old text said "set_label/set_weight not available"
        # whenever the label-swap capability was off, which reads as both being absent -- and set_weight has
        # shipped and been documented for years, so an operator checking the docs concluded mlframe was wrong.
        # In CatBoost 1.2.10 set_weight exists and set_label does not; only the latter gates the swap.
        _missing = [_name for _name, _key in (("set_label", "cb_pool_set_label"), ("set_weight", "cb_pool_set_weight")) if not _dataset_reuse_caps.get(_key)]
        logger.warning(
            "  CatBoost Pool.%s not available in this build (%s present) -- mlframe will rebuild the Pool "
            "for every weight schema and same-type target instead of swapping in place. Upgrade CatBoost to "
            "pick up the Pool label-swap PR.",
            "/".join(_missing) or "label-swap support",
            "/".join(_n for _n in ("set_label", "set_weight") if _n not in _missing) or "neither",
        )

    # Pool cache is keyed by id(df), and Python recycles object ids across independent suite
    # invocations. Without this clear, suite N would see an id() collision against a Pool from
    # suite N-1, fetch the stale Pool, and feed CatBoost stale binned data + stale labels.
    # The cache is small and rebuilds cheaply, so per-suite reset is the safe default.
    #
    # the train-side _CB_POOL_CACHE lives in mlframe.training._cb_pool, NOT
    # in trainer.py. The pre-fix import resolved trainer._CB_POOL_CACHE (a DEAD stub at
    # trainer.py:217 that nothing else reads or writes), called .clear() on an empty dict,
    # and silently succeeded WITHOUT clearing the live cache. The val-side _CB_VAL_POOL_CACHE
    # is re-exported from _predict_guards via trainer.py:71 and its clear was correct.
    try:
        from mlframe.training.cb import _CB_POOL_CACHE
        from mlframe.training.cb._cb_pool_budget import reset_cache_bytes
        from mlframe.training.trainer import _CB_VAL_POOL_CACHE
        _CB_POOL_CACHE.clear()
        _CB_VAL_POOL_CACHE.clear()
        reset_cache_bytes()
    except (ImportError, AttributeError) as _cache_clear_err:
        # Narrow: only the cases that mean "the cache module isn't importable / the symbol
        # was renamed". Anything else (MemoryError, our own bug) should propagate so the
        # stale-Pool risk doesn't silently re-emerge as before.
        logger.warning(
            "CB Pool cache clear skipped: %s: %s. id()-recycle across suite calls " "may now feed stale binned data to CatBoost; investigate.",
            type(_cache_clear_err).__name__,
            _cache_clear_err,
        )

    _mlframe_models_is_default_allowlist = mlframe_models is None
    if mlframe_models is None:
        mlframe_models = ["cb", "lgb", "xgb", "mlp", "linear"]

    # Strategy + tier-sort are suite-constants (depend only on mlframe_models, which never
    # mutates after setup). Computing here avoids paying O(targets * pre_pipelines * models)
    # get_strategy() calls inside the inner training loop.
    from ..strategies import get_strategy as _get_strategy
    from ..models import is_neural_model as _is_neural_model
    _strategy_by_model = {id(m): _get_strategy(m) for m in mlframe_models}
    _sorted_mlframe_models = sorted(
        mlframe_models,
        key=lambda m: (
            _is_neural_model(m),
            tuple(-int(t) for t in _strategy_by_model[id(m)].feature_tier()),
        ),
    )

    metadata = _create_initial_metadata(
        model_name=model_name,
        target_name=target_name,
        mlframe_models=mlframe_models,
        preprocessing_config=preprocessing_config,
        pipeline_config=pipeline_config,
        split_config=split_config,
    )
    metadata["schema_version"] = 2

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
        conformal_config=conformal_config,
        regression_calibration_config=regression_calibration_config,
        composite_target_discovery_config=composite_target_discovery_config,
        linear_model_config=linear_model_config,
        multilabel_dispatch_config=multilabel_dispatch_config,
        ranking_config=ranking_config,
        # Caller's verbose level. Without this the TrainingContext class default (1) was
        # always used regardless of user-passed value, so every ``if ctx.verbose:`` block
        # across phases fired even on verbose=0 runs (including the _phase_finalize.py:438
        # plotly-import for kaleido telemetry, ~25ms cold-start). Fixed.
        verbose=int(verbose) if verbose is not None else 1,
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
        mlframe_models_is_default_allowlist=_mlframe_models_is_default_allowlist,
        strategy_by_model=_strategy_by_model,
        sorted_mlframe_models=_sorted_mlframe_models,
        use_mlframe_ensembles=use_mlframe_ensembles,
        use_ordinary_models=use_ordinary_models,
        metadata=metadata,
    )
    # TrainingContext is slots=True with no ``feature_handling_config`` slot. Stashing via ``ctx.artifacts`` is the only
    # in-scope path; the FH consumer in ``_phase_train_one_target_helpers.py`` (``_maybe_run_feature_handling_apply``)
    # already falls back to ``ctx.artifacts["feature_handling_config"]`` when ``getattr(ctx, "feature_handling_config",
    # None)`` is unset, so the value stashed here is reachable end-to-end.
    if feature_handling_config is not None:
        ctx.artifacts["feature_handling_config"] = feature_handling_config
    # Stash prior values of the process-wide overrides flipped above so _phase_finalize
    # can restore them. Without this restore step, two back-to-back suite calls with
    # different behavior_config.report_residual_audit values silently inherited the
    # first call's setting (the leading "restored after the suite finishes" comment
    # at the original set call was aspirational; no restore call site existed pre-fix).
    ctx.artifacts["_process_flag_prior_residual_audit"] = _residual_audit_prior
    if _inline_display_prior_set:
        ctx.artifacts["_process_flag_prior_inline_display"] = _inline_display_prior
    if _subfolders_prior_set:
        ctx.artifacts["_process_flag_prior_format_subfolders"] = _subfolders_prior
    if _calib_cmap_prior_set:
        ctx.artifacts["_process_flag_prior_calibration_cmap"] = _calib_cmap_prior
    return ctx
