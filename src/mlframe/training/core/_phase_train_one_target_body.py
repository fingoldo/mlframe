"""``_train_one_target`` body carved out of ``mlframe.training.core._phase_train_one_target``.

Holds only the main per-target training entry point. Helpers used by the
function stay in the parent module and are lazily imported inside the
function body so the parent-bottom re-export doesn't create a hard import
cycle (``test_no_import_cycles`` walks top-level imports only).

Re-imported at the parent's module bottom so historical
``from ._phase_train_one_target import _train_one_target`` resolves
transparently.
"""
from __future__ import annotations

import inspect
import logging
from timeit import default_timer as timer
from typing import Any

import numpy as np

try:
    import psutil as _ps_module  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _ps_module = None  # type: ignore[assignment]

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from sklearn.base import clone

from pyutilz.system import tqdmu_lazy_start

from ..configs import TargetTypes as _TargetTypes
from .._ram_helpers import estimate_df_size_mb, get_process_rss_mb, maybe_clean_ram_and_gpu
from ..phases import phase
from ..models import is_neural_model
from ..strategies import get_strategy, PipelineCache
from ..train_eval import process_model, select_target
from ..utils import compute_model_input_fingerprint, filter_existing, get_pandas_view_of_polars_df, log_ram_usage
from ._misc_helpers import (
    _build_tier_dfs, _compute_neural_max_time, _elapsed_str,
    _filter_polars_cat_features_by_dtype, _maybe_clear_shim_cache,
    _prep_polars_df, _split_preds_probs,
)
from ._phase_diagnostics import run_per_target_diagnostics
from ._phase_dummy_baselines import run_dummy_baselines
from ._phase_temporal_audit import _format_temporal_audit_report, _plot_target_over_time
from ._setup_helpers import (
    _build_common_params_for_target, _build_pre_pipelines,
    _build_process_model_kwargs, _setup_model_directories,
    _should_skip_catboost_metamodel,
)
from ._phase_train_one_target_ensembling import _finalize_per_target_ensembling
from ._phase_train_one_target_polars_fastpath import _prepare_strategy_inputs
from ._phase_train_one_target_pre_screen import _maybe_run_unsupervised_pre_screen

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _train_one_target(ctx, target_type, targets, cur_target_name, cur_target_values):
    """Train all models for one (target_type, target_name) pair."""
    # Lazy import: ``._phase_train_one_target`` re-imports this sibling at
    # its module bottom for re-export. Top-level ``from ._phase_train_one_target
    # import ...`` would create a hard import cycle. Python module cache makes
    # repeat imports cheap; the dict lookup is sub-microsecond per call.
    from ._phase_train_one_target import (
        _apply_loss_recommendation_in_place,
        _build_feature_selection_report,
        _cached_init_params,
        _capture_dataset_reuse_cache,
        _compute_pipeline_cache_key,
        _ensure_feature_side_cache,
        _forward_dataset_reuse_cache,
        _invalidate_polars_feature_side_cache,
        _is_regression_target_type,
        _maybe_run_feature_handling_apply,
        _release_ctx_polars_frames,
        _restore_dataset_reuse_cache,
        _selector_params_hash,
        _unwrap_selector,
    )
    from ._phase_train_one_target import slugify
    _maybe_run_unsupervised_pre_screen(ctx, targets)
    model_name = ctx.model_name
    target_name = ctx.target_name
    split_config = ctx.split_config
    hyperparams_config = ctx.hyperparams_config
    behavior_config = ctx.behavior_config
    reporting_config = ctx.reporting_config
    feature_selection_config = ctx.feature_selection_config
    baseline_diagnostics_config = ctx.baseline_diagnostics_config
    dummy_baselines_config = ctx.dummy_baselines_config
    quantile_regression_config = ctx.quantile_regression_config
    verbose = ctx.verbose
    linear_model_config = ctx.linear_model_config
    data_dir = ctx.data_dir
    models_dir = ctx.models_dir
    save_charts = ctx.save_charts
    outlier_detector = ctx.outlier_detector
    use_mrmr_fs = ctx.use_mrmr_fs
    use_ordinary_models = ctx.use_ordinary_models
    use_mlframe_ensembles = ctx.use_mlframe_ensembles
    mrmr_kwargs = ctx.mrmr_kwargs
    rfecv_models = ctx.rfecv_models
    multilabel_dispatch_config = ctx.multilabel_dispatch_config
    custom_pre_pipelines = ctx.custom_pre_pipelines
    common_params_dict = ctx.common_params_dict
    mlframe_models = ctx.mlframe_models
    metadata = ctx.metadata
    target_by_type = ctx.target_by_type
    group_ids = ctx.group_ids
    timestamps = ctx.timestamps
    sample_weights = ctx.sample_weights
    baseline_rss_mb = ctx.baseline_rss_mb
    df_size_mb = ctx.df_size_mb
    train_idx = ctx.train_idx
    test_idx = ctx.test_idx
    train_details = ctx.train_details
    val_details = ctx.val_details
    test_details = ctx.test_details
    fairness_subgroups = ctx.fairness_subgroups
    pipeline = ctx.pipeline
    polars_pipeline_applied = ctx.polars_pipeline_applied
    cat_features = ctx.cat_features
    text_features = ctx.text_features
    embedding_features = ctx.embedding_features
    _dropped_high_card_data = ctx._dropped_high_card_data
    train_df_pd = ctx.train_df_pd
    val_df_pd = ctx.val_df_pd
    test_df_pd = ctx.test_df_pd
    train_df_polars = ctx.train_df_polars
    val_df_polars = ctx.val_df_polars
    test_df_polars = ctx.test_df_polars
    filtered_train_df = ctx.filtered_train_df
    filtered_val_df = ctx.filtered_val_df
    filtered_train_idx = ctx.filtered_train_idx
    filtered_val_idx = ctx.filtered_val_idx
    train_od_idx = ctx.train_od_idx
    val_od_idx = ctx.val_od_idx
    category_encoder = ctx.category_encoder
    imputer = ctx.imputer
    scaler = ctx.scaler
    trainset_features_stats = ctx.trainset_features_stats
    defer_pandas_conv = ctx.defer_pandas_conv
    train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    _all_target_audits = ctx._all_target_audits
    _non_neural_train_times = ctx._non_neural_train_times
    models = ctx.models
    slug_to_original_target_type = ctx.slug_to_original_target_type
    slug_to_original_target_name = ctx.slug_to_original_target_name
    # Initialised pre-conditional so a later reference doesn't NameError when mlframe_models is empty.
    rfecv_models_params = {}
    if mlframe_models:
        # Identity assignment is intentional: keep the slug key registered even when it equals the original name,
        # so downstream lookups via slug never KeyError on round-trip identity targets.
        # Registered ONLY when at least one model is trained -- otherwise the predict-time loader would resolve
        # this slug to a target name that has no corresponding model on disk.
        slug_to_original_target_name[slugify(cur_target_name)] = cur_target_name
        plot_file, model_file = _setup_model_directories(
            target_name=target_name,
            model_name=model_name,
            target_type=target_type,
            cur_target_name=cur_target_name,
            data_dir=data_dir,
            models_dir=models_dir,
            save_charts=save_charts,
        )

        _train_idx = filtered_train_idx if filtered_train_idx is not None else train_idx
        current_train_target = (
            cur_target_values[_train_idx]
            if isinstance(cur_target_values, (np.ndarray, pl.Series))
            else cur_target_values.iloc[_train_idx]
        )
        current_val_target = None
        if filtered_val_idx is not None:
            current_val_target = (
                cur_target_values[filtered_val_idx]
                if isinstance(cur_target_values, (np.ndarray, pl.Series))
                else cur_target_values.iloc[filtered_val_idx]
            )
        # test_idx is intentionally raw (not OD-filtered) - test must never be filtered by outlier detector.
        current_test_target = None
        if test_idx is not None:
            current_test_target = (
                cur_target_values[test_idx]
                if isinstance(cur_target_values, (np.ndarray, pl.Series))
                else cur_target_values.iloc[test_idx]
            )

        # Feature-handling wire-in: opt-in via ctx.feature_handling_config. Sits after the per-target
        # OD-filtered frames + targets are bound (this is the "post-FS / pre-final-pipeline" seam for
        # the inner pre_pipelines x models loops below) and before per-target diagnostics so any
        # FHC-detected text columns surface in the same log block. No-op when fhc is None, so the
        # default code path is unchanged. polars-fastpath frames are preferred when present; the
        # underlying handlers detect polars vs pandas via _extract_column_values. A blanket
        # polars->pandas conversion here would defeat the suite's polars fastpath -- left to apply.py
        # to keep frame container as-given.
        _fhc_train_df = train_df_polars if train_df_polars is not None else filtered_train_df
        _fhc_val_df = val_df_polars if val_df_polars is not None else filtered_val_df
        _fhc_test_df = test_df_polars if test_df_polars is not None else test_df_pd
        _maybe_run_feature_handling_apply(
            ctx,
            cur_target_name=cur_target_name,
            train_df=_fhc_train_df,
            val_df=_fhc_val_df,
            test_df=_fhc_test_df,
            current_train_target=current_train_target,
            sample_weight=sample_weights,
        )

        metadata = run_per_target_diagnostics(
            target_type=target_type,
            cur_target_name=cur_target_name,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            current_test_target=current_test_target,
            filtered_train_df=filtered_train_df,
            filtered_val_df=_fhc_val_df if "_fhc_val_df" in dir() else None,
            filtered_test_df=_fhc_test_df if "_fhc_test_df" in dir() else None,
            baseline_diagnostics_config=baseline_diagnostics_config,
            cat_features=cat_features,
            metadata=metadata,
        )

        metadata = run_dummy_baselines(
            target_type=target_type,
            cur_target_name=cur_target_name,
            target_name=target_name,
            model_name=model_name,
            current_train_target=current_train_target,
            current_val_target=current_val_target,
            current_test_target=current_test_target,
            filtered_train_df=filtered_train_df,
            filtered_val_df=filtered_val_df,
            test_df_pd=test_df_pd,
            filtered_train_idx=filtered_train_idx,
            filtered_val_idx=filtered_val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            cat_features=cat_features,
            dummy_baselines_config=dummy_baselines_config,
            quantile_regression_config=quantile_regression_config,
            reporting_config=reporting_config,
            _dropped_high_card_data=_dropped_high_card_data,
            train_od_idx=train_od_idx,
            val_od_idx=val_od_idx,
            plot_file=plot_file,
            metadata=metadata,
            target_by_type=target_by_type,
            _split_preds_probs=_split_preds_probs,
            # Propagate ctx.group_ids so LTR-Popularity / per-group dummy baselines fire on
            # LTR suites instead of silently degrading to regression-style dummy + blank
            # baseline table (wave 12 #2).
            group_ids=getattr(ctx, "group_ids", None),
        )

        # Audits are precomputed once for all targets via the batch API; this lookup is the per-target render.
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

        # Feature-drift auto-action: when the per-target
        # ``feature_distribution_drift`` report carries a
        # ``recommend_neural_overrides`` dict (FI-weighted score crossed
        # the target-type-grouped threshold in
        # ``WEIGHTED_DRIFT_NEURAL_OVERRIDE_THRESHOLDS``), translate the
        # sklearn-shape override into the nested mlframe ``mlp_kwargs``
        # shape and apply it as a PER-TARGET override to
        # ``hyperparams_config.mlp_kwargs``. Other targets in the same
        # suite keep the original mlp_kwargs.
        #
        # Per-target-type gating happens INSIDE ``compute_feature_distribution_drift``:
        #   regression  -> threshold=3.0 (precision=1.000 / recall=0.883 vs MLP_excess_R^2_harm>0.1)
        #   classification -> threshold=None (DISABLED -- the paired study
        #     found overall Pearson r=-0.101 with interaction_binary r=-0.227,
        #     i.e. MLP with relu OUTPERFORMS LogReg on interaction-rich
        #     classification under drift; no threshold gives reasonable
        #     precision).
        # The wire-in itself is target-type-agnostic; it just consumes the
        # ``recommend_neural_overrides`` payload the sensor produced.
        _target_hyperparams_config = hyperparams_config
        try:
            _fd_for_target = (
                metadata.get("feature_distribution_drift", {})
                .get(str(target_type), {})
                .get(cur_target_name)
            )
            _sklearn_override = (
                _fd_for_target.get("recommend_neural_overrides")
                if isinstance(_fd_for_target, dict) else None
            )
            if _sklearn_override and "mlp" in mlframe_models:
                from ..feature_drift_report import (
                    translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs,
                )
                _mlframe_override = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs(
                    _sklearn_override,
                )
                _untranslated = _mlframe_override.pop("__untranslated__", None)
                _orig_mlp_kwargs = (
                    getattr(hyperparams_config, "mlp_kwargs", None) or {}
                )
                _merged_mlp_kwargs = dict(_orig_mlp_kwargs)
                for _slot in ("model_params", "network_params"):
                    if _slot in _mlframe_override:
                        _merged_mlp_kwargs.setdefault(_slot, {})
                        _merged_mlp_kwargs[_slot] = dict(
                            {**_merged_mlp_kwargs[_slot], **_mlframe_override[_slot]}
                        )
                try:
                    _target_hyperparams_config = hyperparams_config.model_copy(
                        update={"mlp_kwargs": _merged_mlp_kwargs},
                    )
                except Exception:
                    _target_hyperparams_config = hyperparams_config
                metadata.setdefault("feature_drift_auto_action", {}) \
                    .setdefault(str(target_type), {})[cur_target_name] = {
                        "sklearn_override": dict(_sklearn_override),
                        "mlframe_mlp_kwargs_override": _mlframe_override,
                        "untranslated_keys": _untranslated or [],
                        "weighted_drift_score": _fd_for_target.get("weighted_drift_score"),
                    }
                logger.warning(
                    "[feature-drift-auto-action] target='%s' weighted_drift=%.2f "
                    ">= 3.0 -- applying empirically-grounded MLP HPT override "
                    "(sklearn-shape: %s; mlframe-mlp_kwargs deep-merge: %s%s). "
                    "Grounded by profiling/bench_mlp_robustness_sweep.py "
                    "(2026-05-22 sweep, 1440 trials, baseline harm=6.455 -> "
                    "pick harm=0.0006 at drift_z=10).",
                    cur_target_name,
                    float(_fd_for_target.get("weighted_drift_score") or 0.0),
                    _sklearn_override, _mlframe_override,
                    f" -- untranslated keys: {_untranslated}" if _untranslated else "",
                )
        except Exception as _fd_aa_err:
            logger.warning(
                "feature-drift-auto-action failed for target='%s' (%s); "
                "training continues without per-target MLP override.",
                cur_target_name, _fd_aa_err,
            )

        # Test set is never OD-filtered. train_df_size_bytes_cached is the pre-conversion Polars-side size
        # passed through so configure_training_params can skip a 3-min pandas memory_usage(deep=...) scan
        # on high-cardinality object columns; the OD-shrinkage approximation only feeds a GPU-RAM heuristic.
        common_params, models_params, rfecv_models_params, cpu_configs, gpu_configs = select_target(
            model_name=f"{target_name} {model_name} {cur_target_name}",
            target=cur_target_values,
            target_type=target_type,
            df=None,
            train_df=filtered_train_df,
            val_df=filtered_val_df,
            test_df=test_df_pd,
            train_idx=filtered_train_idx,
            val_idx=filtered_val_idx,
            test_idx=test_idx,
            train_details=train_details,
            val_details=val_details,
            test_details=test_details,
            group_ids=group_ids,
            cat_features=cat_features,
            text_features=text_features,
            embedding_features=embedding_features,
            hyperparams_config=_target_hyperparams_config,
            behavior_config=current_behavior_config,
            common_params=od_common_params,
            mlframe_models=mlframe_models,
            linear_model_config=linear_model_config,
            train_df_size_bytes=train_df_size_bytes_cached,
            val_df_size_bytes=val_df_size_bytes_cached,
            multilabel_dispatch_config=multilabel_dispatch_config,
        )

        if verbose:
            logger.info("  select_target done in %s", _elapsed_str(t0_select_target))
            log_ram_usage()

        # Pack H: auto-pick MAE / Huber loss for heavy-tail regression
        # residuals. ``cur_target_values`` is the raw y for raw-target or
        # the composite residual T for composite-target paths; in both
        # cases the inner boosting fits this distribution directly, so
        # the auto-switch matches the actual signal-vs-noise regime.
        if _is_regression_target_type(target_type):
            _apply_loss_recommendation_in_place(
                models_params=models_params,
                target_values=cur_target_values,
                composite_name=cur_target_name,
                logger_=logger,
                verbose=verbose,
            )

        pre_pipelines, pre_pipeline_names = _build_pre_pipelines(
            use_ordinary_models=use_ordinary_models,
            rfecv_models=rfecv_models,
            rfecv_models_params=rfecv_models_params,
            use_mrmr_fs=use_mrmr_fs,
            mrmr_kwargs=mrmr_kwargs,
            custom_pre_pipelines=custom_pre_pipelines,
            rfecv_leakage_corr_threshold=feature_selection_config.rfecv_leakage_corr_threshold,
            rfecv_mbh_adaptive_threshold=feature_selection_config.rfecv_mbh_adaptive_threshold,
            use_boruta_shap=feature_selection_config.use_boruta_shap,
            boruta_shap_kwargs=feature_selection_config.boruta_shap_kwargs,
            use_sample_weights_in_fs=feature_selection_config.use_sample_weights_in_fs,
            mrmr_identity_cache=(
                ctx._mrmr_identity_cache
                if getattr(feature_selection_config, "mrmr_identity_cache_scope", "ctx") == "ctx"
                else None
            ),
            # Thread target_type so BorutaShap can auto-derive
            # ``classification=False`` for regression targets (otherwise the
            # default RandomForestClassifier crashes on continuous y inside
            # sklearn.multiclass).
            target_type=target_type,
        )
    else:
        # No mlframe_models means the downstream pre_pipeline loop must be a no-op; bind empty sequences
        # so callers that iterate ``zip(pre_pipelines, pre_pipeline_names)`` see zero iterations rather
        # than NameError on the unbound names.
        pre_pipelines, pre_pipeline_names = [], []

    # Custom transformers run AFTER preprocessing, so the preprocessing output is shared across
    # pre_pipelines of the same model-type bucket; one cache instance covers the whole sweep. Hoist
    # to ctx (PIPECACHE-PER-TGT) so multi-target suites share one cache across targets -- selector /
    # encoder fits done for target 1 are reusable for target 2 when the cache_key matches (only
    # changes when the feature set / strategy / kind / pp_name changes).
    if ctx._pipeline_cache is None:
        ctx._pipeline_cache = PipelineCache()
    pipeline_cache = ctx._pipeline_cache

    # Suite-scoped cache observability. ``finalize_suite`` aggregates these into
    # ``metadata["cache_stats"]``. Initialise once per call rather than per pre_pipeline so the inner
    # loop's HIT / MISS bumps accumulate across the whole target's training, and use ``setdefault``
    # at ctx level so cross-target calls (multi-target suites) keep counters monotonic across calls.
    if not hasattr(ctx, "_cache_stats") or ctx._cache_stats is None:
        ctx._cache_stats = {}

    for pre_pipeline, pre_pipeline_name in tqdmu_lazy_start(zip(pre_pipelines, pre_pipeline_names), desc="pre_pipeline", total=len(pre_pipelines)):
        # CatBoost + RFECV metamodel_func combination breaks sklearn.clone().
        if _should_skip_catboost_metamodel(pre_pipeline_name.strip(), target_type, behavior_config):
            continue

        # Skip identity-equivalent pre_pipelines: marker survives across targets, so a selector
        # that was a no-op on a prior target gets skipped here before any model trains.
        # Honour ``feature_selection_config.skip_identity_equivalent_pre_pipelines``: when False
        # the caller asked to retrain even on identity-equivalent pre_pipelines (e.g. for
        # ensembling-diversity-via-RNG-seed scenarios), so this early-exit must not fire.
        _pp_name_stripped = pre_pipeline_name.strip()
        if (
            _pp_name_stripped
            and feature_selection_config.skip_identity_equivalent_pre_pipelines
            and getattr(pre_pipeline, "_mlframe_identity_equivalent", False)
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

        if sample_weights:
            weight_schemas = sample_weights
            # SW-LOG-PER-PP-PER-TGT: emit this banner once per suite, not once per (target x
            # pre_pipeline x weight). The weighting schema is suite-constant; identical lines
            # repeated K_targets x K_pp times bloat the log without adding info.
            if not ctx._sw_log_emitted:
                if "uniform" in sample_weights:
                    logger.info("Using %d weighting schema(s) from extractor: %s", len(weight_schemas), list(weight_schemas.keys()))
                else:
                    logger.info("Using %d weighting schema(s) from extractor: %s. Note: uniform weighting not included.", len(weight_schemas), list(weight_schemas.keys()))
                ctx._sw_log_emitted = True
        else:
            weight_schemas = {"uniform": None}
            if not ctx._sw_log_emitted:
                logger.info("No weighting schemas from extractor, defaulting to uniform weighting.")
                ctx._sw_log_emitted = True

        # Backward val placement + recency weighting cancel each other's drift-proxy intent
        # (val older than train, training biased to newest rows). Warn so the user picks one.
        # VAL-PLACE-WARN-PP: gate behind a per-suite latch so the warning fires once, not per PP.
        _val_placement = getattr(split_config, "val_placement", "forward")
        if _val_placement == "backward" and not ctx._val_placement_warn_emitted:
            _non_uniform = [k for k in weight_schemas.keys() if k != "uniform"]
            if _non_uniform:
                ctx._val_placement_warn_emitted = True
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

        # Models sorted by feature tier (richest first) so text/embedding columns are dropped once per tier.
        # Strategy lookup keyed by id() because estimators / tuples are not hashable, and identity-distinct
        # instances must stay distinct in the map. Pre-computed once per suite by setup_configuration;
        # reading off ctx here avoids the O(targets * pre_pipelines * models) re-evaluation that used to
        # rebuild this map per inner-loop iteration.
        strategy_by_model = ctx.strategy_by_model
        sorted_models = ctx.sorted_mlframe_models
        # Suite-scoped feature-side cache: tier_dfs / pl.Enum map / prepared polars frames carry
        # ACROSS targets (target-independent transforms) so only y / sample_weight differ inside
        # the inner loop. Both inner caches are scoped to the current ``pre_pipeline_name`` since
        # different pre_pipelines may keep different columns (MRMR / RFECV vs ordinary), and the
        # tier-DFs / Enum maps depend on the column set after pre-pipeline column trimming.
        _suite_feature_cache = _ensure_feature_side_cache(ctx)
        _per_pp_cache = _suite_feature_cache.setdefault(pre_pipeline_name, {})
        tier_dfs_cache: dict[tuple, dict[str, Any]] = _per_pp_cache.setdefault("tier_dfs", {})
        # Leak-free pl.Enum map built from train+val UNION only (test EXCLUDED to avoid label-time leakage).
        # Depends only on (feature_tier, strategy class) - target-independent so it carries cross-target.
        tier_enum_map_cache: dict[tuple, dict[str, Any] | None] = _per_pp_cache.setdefault("tier_enum_map", {})
        # Prepared polars frames + xgb_category_map per (tier, supports_polars, strategy_class).
        # Target-independent because _prep_polars_df / build_polars_enum_map do not touch y; the
        # text-features fill_null pass below is also target-independent. Carry cross-target.
        prepared_frames_cache: dict[tuple, dict[str, Any]] = _per_pp_cache.setdefault("prepared_frames", {})
        prev_tier = None

        # Neural max_time defaults to P95 of non-neural train times so MLP can't run 2h while boosters take 5min.
        # CODE-LOW-4: per-target reset is INTENTIONAL -- each target's neural budget is computed only from the
        # same target's non-neural runs, so an unusually fast/slow earlier target cannot widen or starve the
        # current target's neural budget. We rebind both the local AND ctx._non_neural_train_times to the
        # SAME fresh list so the writeback at end-of-function is a no-op (the dict the caller sees is the
        # one we just mutated) and downstream readers of ctx._non_neural_train_times observe the per-target
        # contents in-flight, not the previous target's tail.
        _non_neural_train_times = []
        ctx._non_neural_train_times = _non_neural_train_times

        _total_models_in_run = len(sorted_models)
        _model_idx_in_run = 0
        _break_model_loop = False
        for mlframe_model_name in tqdmu_lazy_start(sorted_models, desc="mlframe model"):
            if _should_skip_catboost_metamodel(mlframe_model_name, target_type, behavior_config):
                continue
            _model_idx_in_run += 1
            if verbose:
                # Per-model RSS sample is intentional: localising OOM-blame to a specific
                # model+target+pre_pipeline tuple in the verbose-suite log saves hours of post-mortem
                # log-correlation. The ~3ms/call Windows cost is dwarfed by per-model fit times.
                # PSUTIL-IMPORT-HOT: ``psutil`` is now imported at module level (``_ps_module``);
                # the prior in-loop import paid ImportError lookup costs on every iter.
                try:
                    _ram_gb_now = (
                        _ps_module.Process().memory_info().rss / (1024 ** 3)
                        if _ps_module is not None else 0.0
                    )
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

            # Cross-target dataset reuse: restore the prior target's _DATASET_REUSE_CACHE_ATTRS
            # snapshot onto the freshly-built model template BEFORE the weight loop's clone()
            # forward-transfer reads them. select_target() rebuilds models_params per target so
            # the cache attributes are absent on a virgin template - without this restore the
            # XGB/LGB shims would rebuild the binned dataset on target 2. The shim's
            # signature_of(X) check then matches against the same ctx-pinned train_df pointer
            # and triggers set_label / set_weight in place rather than a fresh build.
            _restore_dataset_reuse_cache(
                ctx, mlframe_model_name, models_params[mlframe_model_name]["model"],
                pp_name=pre_pipeline_name,
            )

            strategy = strategy_by_model[id(mlframe_model_name)]

            # Drop pre-pipeline Polars originals as soon as we hit the first non-Polars strategy. The
            # post-iteration release fires only on tier transitions, but same-tier siblings (e.g. XGB and
            # LGB share tier=(False,False)) would keep Polars frames alive into a lazy pandas conversion,
            # doubling peak RAM. Releasing upfront halves peak in mixed suites.
            if (
                not strategy.supports_polars
                and train_df_polars is not None
            ):
                # Drop locals AND ctx attributes -- ctx still pins the strong ref to the same frames
                # assigned via ctx.*_df_polars at function entry, so a bare ``del`` of the locals would
                # leave maybe_clean_ram_and_gpu with nothing to reclaim and turn the log line into a lie.
                del train_df_polars, val_df_polars, test_df_polars
                train_df_polars = val_df_polars = test_df_polars = None
                # Drop polars-tier entries only - pandas-tier entries hang on the SAME cache dicts
                # (these locals now reference suite-scoped dicts in _per_pp_cache) and must survive
                # the release. ``_invalidate_polars_feature_side_cache(ctx)`` runs further down the
                # _release_ctx_polars_frames path and does the same for the prepared_frames sub-
                # cache; the tier_dfs / tier_enum_map dicts that pre-date this hoist are scrubbed
                # here so a same-target pandas-tier sibling reads a clean enum-map slot.
                for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                    tier_dfs_cache.pop(_pl_only_key, None)
                tier_enum_map_cache.clear()  # All entries are polars-only (populated only on the polars fastpath).
                baseline_rss_mb = _release_ctx_polars_frames(
                    ctx,
                    baseline_rss_mb,
                    df_size_mb,
                    verbose=verbose,
                    reason="non-polars-native strategy entry",
                )
                if verbose:
                    logger.info(
                        "  Released pre-pipeline Polars originals before %s (non-polars-native strategy).",
                        mlframe_model_name,
                    )

            # Clone the base_pipeline per model so each iteration gets a fresh, un-fitted selector. Sharing a
            # fitted MRMR/RFECV across strategies caused `_is_fitted` to misreport True for a partially-fit
            # pipeline (selector fitted but encoder/imputer/scaler not), tripping imputer.transform on a
            # feature-names mismatch.
            _base_for_strategy = orig_pre_pipeline
            if _base_for_strategy is not None:
                try:
                    _base_for_strategy = clone(_base_for_strategy)
                except Exception as _clone_e:
                    # Custom non-BaseEstimator pipelines can't be sklearn-cloned;
                    # falling back to the original reference is correct IF the
                    # pipeline is genuinely stateless OR if it carries its own
                    # per-call reset. WARN-log so operators see when this
                    # fallback fires -- a partially-fit selector reused across
                    # strategies trips `imputer.transform` on a feature-names
                    # mismatch (the exact bug the docstring 4 lines above
                    # describes).
                    logger.warning(
                        "  sklearn.clone failed for base_pipeline (%s); reusing "
                        "original reference. If %s is a stateful selector with "
                        "no per-call reset, downstream `pre_pipeline.fit` may "
                        "see stale state from a prior model in the suite.",
                        _clone_e, type(_base_for_strategy).__name__,
                    )
            pre_pipeline = strategy.build_pipeline(
                base_pipeline=_base_for_strategy,
                cat_features=cat_features,
                category_encoder=category_encoder if cat_features else None,
                imputer=imputer,
                scaler=scaler,
            )
            # Cache key = strategy.cache_key + pre_pipeline_name + feature_tier + container kind + feature-list digest.
            # feature_tier is required because CB/LGB/XGB all share cache_key="tree" but have different
            # tiers; without it, CB's text/embedding-bearing frame would be served to LGB/XGB.
            # Kind suffix prevents Polars-native (XGB) and pandas-only (LGB) consumers from sharing entries
            # within a tier, which would otherwise undo the lazy pandas conversion downstream.
            # See _compute_pipeline_cache_key for the features-digest contract (frozenset, order-invariant).
            # Pass the polars train frame (if present) so dtype changes between targets / runs
            # invalidate the cache; pandas frames don't reach this branch typed-distinct enough to
            # need the suffix (handled upstream in split_features), so it's safe to skip there.
            _cache_key_train_df = train_df_polars if strategy.supports_polars else None
            # Use a CONTENT-based cache key derived from the preprocessing-
            # requirements tuple instead of strategy.cache_key (name-based).
            # Two strategies that consume IDENTICAL ``imp+scaler`` pipelines
            # MUST hit the same cache slot; name-keyed lookups
            # (LinearStrategy.cache_key="linear" vs NeuralStrategy.cache_key=
            # "neural") miss-on-name and re-do the 17s pre_pipeline transform
            # for the second tier (e.g. MLP after Ridge) on the same 4M rows.
            # The content key folds (requires_imputation, requires_scaling,
            # requires_encoding) into a stable string so any two strategies
            # with matching requirements share the cache slot.
            _content_key = (
                f"imp{int(getattr(strategy, 'requires_imputation', False))}"
                f"_scale{int(getattr(strategy, 'requires_scaling', False))}"
                f"_enc{int(getattr(strategy, 'requires_encoding', False))}"
            )
            cache_key = _compute_pipeline_cache_key(
                _content_key,
                pre_pipeline_name,
                strategy.feature_tier(),
                strategy.supports_polars,
                cat_features,
                text_features,
                embedding_features,
                train_df=_cache_key_train_df,
            )

            # Polars fastpath substitutes original Polars DataFrames for natively-Polars consumers
            # (CatBoost >= 1.2.7, HGB). Polars DFs are prepared once per model (outside the weight loop)
            # because prepare_polars_dataframe() allocates via .with_columns().
            polars_fastpath_active = train_df_polars is not None and strategy.supports_polars

            _prep_out = _prepare_strategy_inputs(
                polars_fastpath_active=polars_fastpath_active,
                mlframe_model_name=mlframe_model_name,
                strategy=strategy,
                cat_features=cat_features,
                text_features=text_features,
                embedding_features=embedding_features,
                train_df_polars=train_df_polars,
                val_df_polars=val_df_polars,
                test_df_polars=test_df_polars,
                prepared_frames_cache=prepared_frames_cache,
                tier_dfs_cache=tier_dfs_cache,
                tier_enum_map_cache=tier_enum_map_cache,
                common_params=common_params,
                pre_pipeline_name=pre_pipeline_name,
                ctx=ctx,
                verbose=verbose,
            )
            prepared_train = _prep_out["prepared_train"]
            prepared_val = _prep_out["prepared_val"]
            prepared_test = _prep_out["prepared_test"]
            _xgb_category_map = _prep_out["xgb_category_map"]
            _cat_features = _prep_out["cat_features"]
            tier_pandas = _prep_out["tier_pandas"]

            # CODE-P1-10: compute input-schema fingerprint ONCE per (model, pre_pipeline) outside the
            # weight loop. The fingerprinted train_df is the same across all weight schemas (only
            # sample_weight changes inside the weight loop), so the previous per-iteration call was
            # pure waste. Cache key is purely feature-side (strategy+tier+kind+pp_name) - dropping
            # ``target_type`` / ``cur_target_name`` from the key was the per-target hoist: the
            # schema hash depends on column names/dtypes, NOT on y, so target N reuses target 1's
            # fingerprint without recomputation. Audit-checked vs compute_model_input_fingerprint:
            # signature takes train_df + cat/text/embedding_features only, no target.
            _fp_train_df_pre = prepared_train if polars_fastpath_active else tier_pandas["train_df"]
            # FP-KEY-OMITS-CONTENT: original key excluded the train_df identity, so when the same
            # strategy / tier / kind / pp_name combination was hit by two different per-target
            # frames (filtered_train_df rebuilt across targets), the cache would return target 1's
            # schema hash for target 2. Fold ``id(train_df)`` (strong-ref-pinned at this point) and
            # the schema column-count to disambiguate; full schema hash isn't needed here because a
            # mismatch in id alone forces recompute.
            _fp_train_df_id = id(_fp_train_df_pre) if _fp_train_df_pre is not None else 0
            _fp_train_df_ncols = (
                len(_fp_train_df_pre.columns)
                if _fp_train_df_pre is not None and hasattr(_fp_train_df_pre, "columns")
                else 0
            )
            _fp_cache_key = (
                id(strategy),
                strategy.feature_tier(),
                strategy.supports_polars,
                pre_pipeline_name,
                _fp_train_df_id,
                _fp_train_df_ncols,
            )
            # Fingerprint cache stats: HIT when the per-(strategy, tier, kind, pp_name) key is already
            # cached, MISS when we have to compute. Same proxy-counter pattern as the pandas-view
            # cache above (the underlying cache is a plain dict on ctx with no counters of its own).
            _cs_fp = ctx._cache_stats.setdefault("fingerprint_cache", {"hits": 0, "misses": 0})
            if _fp_cache_key in ctx._model_input_fingerprint_cache:
                _cs_fp["hits"] += 1
                _schema_hash, _input_schema = ctx._model_input_fingerprint_cache[_fp_cache_key]
            else:
                _cs_fp["misses"] += 1
                _schema_hash, _input_schema = compute_model_input_fingerprint(
                    _fp_train_df_pre,
                    cat_features=cat_features,
                    text_features=text_features,
                    embedding_features=embedding_features,
                )
                ctx._model_input_fingerprint_cache[_fp_cache_key] = (_schema_hash, _input_schema)

            for weight_name, weight_values in tqdmu_lazy_start(weight_schemas.items(), desc="weighting schema"):
                model_name_with_weight = common_params["model_name"]
                model_file_name=f"{mlframe_model_name}"
                if weight_name != "uniform":
                    model_name_with_weight += f" w={weight_name}"
                    model_file_name +=f"_{weight_name}"

                # Isolation copy: per-(model, weight) inner mutations (sample_weight, plot_file
                # decoration, lazy pandas conversion, fastpath frame swap) must not bleed into
                # the outer ``common_params`` template that the next iteration consumes. The
                # 4-deep nesting (target_type x target x pre_pipeline x model x weight) has been
                # verified across the suite -- removing the copy regresses the cross-weight
                # contamination tests. Do NOT inline.
                current_common_params = common_params.copy()
                current_common_params["sample_weight"] = weight_values

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
                if getattr(behavior_config, "model_file_hash_suffix", True):
                    model_file_name += f"__sch_{_schema_hash}"

                if weight_name != "uniform" and current_common_params.get("plot_file"):
                    current_common_params["plot_file"] = current_common_params["plot_file"] + weight_name + "_"

                cached_dfs = pipeline_cache.get(cache_key)

                # INTENTIONAL: clone() lives INSIDE the weight loop. Each weight schema produces a
                # different trained model stored separately in models[type][target]; without per-iteration
                # cloning all in-memory entries would alias to the same last-trained sklearn object and
                # only the .dump snapshots would be correct. Do NOT move clone() outside the loop.
                original_model = models_params[mlframe_model_name]["model"]
                try:
                    cloned_model = clone(original_model)
                except RuntimeError:
                    # CatBoost wraps custom eval_metric objects internally; sklearn's identity check fails.
                    # Direct constructor call with get_params() produces an equivalent unfitted instance.
                    cloned_model = type(original_model)(**original_model.get_params())
                except TypeError:
                    # NGBoost: get_params() exposes attributes the constructor doesn't accept.
                    # SIG-IN-EXCEPT: memoize the inspect.signature lookup so the TypeError branch
                    # isn't paying ~0.5-1ms per hit -- the cache lives at module scope keyed by
                    # ``id(cls)`` because ``cls.__init__`` is class-invariant.
                    _cls = type(original_model)
                    _sig_params = _cached_init_params(_cls)
                    _raw = original_model.get_params(deep=False)
                    cloned_model = _cls(**{k: v for k, v in _raw.items() if k in _sig_params})
                # sklearn.clone() strips non-param attributes; re-assert mlframe sticky flags so the
                # calibration directive and the polars-fastpath-broken marker survive each iteration.
                if getattr(original_model, "_mlframe_posthoc_calibrate", False):
                    try:
                        cloned_model._mlframe_posthoc_calibrate = True
                    except Exception as _attr_err:
                        logger.debug("Could not set _mlframe_posthoc_calibrate on clone: %s", _attr_err)
                if getattr(original_model, "_mlframe_polars_fastpath_broken", False):
                    try:
                        cloned_model._mlframe_polars_fastpath_broken = True
                    except Exception as _attr_err:
                        logger.debug("Could not set _mlframe_polars_fastpath_broken on clone: %s", _attr_err)
                # Hand the XGB DMatrix / LGB Dataset reuse caches forward across clone() so the
                # weight-schema loop (uniform -> recency on the same train_df) reuses the heavy binned
                # dataset in place via set_label / set_weight instead of rebuilding.
                _forward_dataset_reuse_cache(original_model, cloned_model)
                # Isolation copy: each weight iteration installs its own cloned_model and may
                # patch fit_params (CatBoost text/embedding fastpath); without copying we would
                # mutate the suite-level models_params template and the next target would inherit
                # this iteration's overrides.
                current_model_params = models_params[mlframe_model_name].copy()
                current_model_params["model"] = cloned_model

                # CatBoost is the only Polars-native consumer that accepts cat_features / text_features /
                # embedding_features at fit time; XGB and HGB auto-detect via enable_categorical=True.
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
                    # Per-strategy decision on whether preprocessing for this strategy is already done.
                    # Two sufficient conditions:
                    #   (1) the suite-level polars-ds pipeline ran AND this strategy consumes polars natively;
                    #   (2) the polars fastpath is active for this strategy (its frame is the polars native
                    #       one, so sklearn encoder/scaler/imputer would be redundant and crash anyway).
                    # Note: requires_encoding=True is NOT a re-run trigger (HGB declares it for pandas-fallback
                    # only; on the polars fastpath HGB consumes pl.Categorical natively). Only non-Polars
                    # strategies fall through to their own pre_pipeline run in trainer.py.
                    polars_pipeline_applied=(
                        (polars_pipeline_applied and strategy.supports_polars)
                        or polars_fastpath_active
                    ),
                    mlframe_model_name=mlframe_model_name,
                    metadata_columns=metadata.get("columns"),
                )

                _is_neural = is_neural_model(mlframe_model_name)
                _timeout = _compute_neural_max_time(_non_neural_train_times) if _is_neural else None
                if _timeout is not None:
                    _max_time_dict, _p95, _n = _timeout
                    # Reach into Pipeline(StandardScaler, TTR(PytorchLightningRegressor(...))) to find trainer_params.
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
                                    mlframe_model_name,
                                    _max_time_dict["hours"], _max_time_dict["minutes"], _max_time_dict["seconds"],
                                    _n, _p95,
                                )

                t0_model = timer()
                try:
                    with phase("process_model", model=mlframe_model_name, weight=weight_name):
                        trainset_features_stats, pre_pipeline, train_df_transformed, val_df_transformed, test_df_transformed = process_model(
                            **process_model_kwargs
                        )
                except Exception as model_err:
                    # Skip-and-continue is opt-in. KeyboardInterrupt is intentionally not caught here;
                    # native SIGSEGV that kills the process won't be caught either.
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

                # After the first model trains, if the pre_pipeline is identity-equivalent (kept all
                # columns) AND the ordinary branch is in the suite, the remaining models would see
                # identical data - skip them.
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

                # Hand the dataset-reuse cache from cloned_model back to the template so the next
                # weight-schema iteration's clone() carries it forward (symmetric to the forward-transfer
                # block above). Without this the cache would be born and die in a single iteration.
                _forward_dataset_reuse_cache(cloned_model, original_model, skip_none=True)

                # Persist this model's input-schema fingerprint in metadata so load-time can verify it
                # against the serving frame. Multi-output extensions (target_type / n_classes /
                # multilabel_strategy + schema_version) let load_mlframe_suite dispatch correctly;
                # legacy artifacts without these fields fall back to binary inference.
                _record = {
                    "schema_hash": _schema_hash,
                    "input_schema": _input_schema,
                    "mlframe_model": mlframe_model_name,
                    "weight_name": weight_name,
                    "target_type": str(target_type) if target_type is not None else None,
                    "schema_version": 2,  # 1=legacy, 2=multi-output-aware
                }
                train_y = (
                    cur_target_values[_train_idx]
                    if isinstance(cur_target_values, (np.ndarray, pl.Series))
                    else cur_target_values.iloc[_train_idx]
                )
                try:
                    if target_type == _TargetTypes.MULTILABEL_CLASSIFICATION:
                        _record["n_classes"] = (
                            int(train_y.shape[1])
                            if hasattr(train_y, "shape") and train_y.ndim == 2
                            else None
                        )
                        _record["multilabel_strategy"] = "native" if (
                            hasattr(strategy, "supports_native_multilabel") and strategy.supports_native_multilabel
                        ) else "wrapper"
                    elif target_type == _TargetTypes.MULTICLASS_CLASSIFICATION:
                        _record["n_classes"] = (
                            int(len(np.unique(np.asarray(train_y))))
                            if hasattr(train_y, "shape") else None
                        )
                        _record["multilabel_strategy"] = None
                    else:
                        _record["n_classes"] = None
                        _record["multilabel_strategy"] = None
                except Exception as _intro_err:
                    # Never fail the metadata write because of an introspection error on optional fields.
                    # Surface as warning since load_mlframe_suite dispatches on n_classes/multilabel_strategy.
                    logger.warning("n_classes/multilabel_strategy introspection failed for %s: %s", mlframe_model_name, _intro_err)

                # Per-model feature-selection report. ``pre_pipeline`` returned by ``process_model``
                # is the FITTED selector / pipeline (or None for the ordinary branch). ``train_df_
                # transformed.columns`` gives the post-FS surviving features for both pandas and
                # polars frames. The report is always stamped (selector_name=None for ordinary) so
                # downstream consumers can rely on the key existing.
                #
                # Cache the report at (target, pp_name, model_name, selector_params_hash, kept_cols)
                # because the fitted selector + kept columns are weight-invariant. The prior key used
                # id(pre_pipeline) which is Python's memory-address; once an object is GC'd its id can
                # be recycled, so a long-lived ``ctx._fs_report_cache`` could collide on a recycled
                # address across the per-(target, model) inner loops. ``_selector_params_hash`` is
                # content-derived and id-stable across recycling.
                try:
                    _kept_cols = None
                    if train_df_transformed is not None and hasattr(train_df_transformed, "columns"):
                        _kept_cols = list(train_df_transformed.columns)
                    _fsr_key = (
                        cur_target_name,
                        pre_pipeline_name,
                        mlframe_model_name,
                        _selector_params_hash(_unwrap_selector(pre_pipeline)),
                        tuple(_kept_cols) if _kept_cols is not None else None,
                    )
                    _fsr_cached = ctx._fs_report_cache.get(_fsr_key)
                    if _fsr_cached is None:
                        _fsr_cached = _build_feature_selection_report(
                            pre_pipeline=pre_pipeline,
                            pre_pipeline_name=pre_pipeline_name,
                            fitted_columns_in=None,
                            kept_columns=_kept_cols,
                        )
                        ctx._fs_report_cache[_fsr_key] = _fsr_cached
                    _record["feature_selection_report"] = _fsr_cached
                except Exception as _fsr_err:
                    logger.warning("feature_selection_report build failed for %s: %s", mlframe_model_name, _fsr_err)
                    _record["feature_selection_report"] = {
                        "selector_name": None,
                        "selector_params_hash": None,
                        "kept_features": None,
                        "dropped_features": None,
                        "scores": None,
                        "reason_per_feature": None,
                    }

                metadata.setdefault("model_schemas", {})[model_file_name] = _record

                if cached_dfs is None:
                    pipeline_cache.set(cache_key, train_df_transformed, val_df_transformed, test_df_transformed)

            # Preserve a fitted feature-selector across same-bucket tree iterations. Tree strategies return
            # just the base_pipeline from build_pipeline(); non-tree strategies wrap it in a full Pipeline
            # (encoder/imputer/scaler), which we do NOT want to reuse as the base for other model types.
            if cache_key.startswith("tree"):
                orig_pre_pipeline = pre_pipeline

            if _break_model_loop:
                break

            # Release dataset-reuse caches at strategy-iter end. Both shims park the heavy binned dataset
            # on ``_cached_train_*`` / ``_cached_val_*`` as a weight-schema-loop scratchpad; nothing
            # downstream reads them (.predict goes through _Booster, ensemble uses pre-computed probs,
            # save strips via __getstate__). Releasing here frees ~30% of peak RAM between strategies.
            # Capture the binned-dataset references off the template BEFORE clearing so the next
            # target's _restore_dataset_reuse_cache can re-attach them. Without this snapshot the
            # clear below frees the dataset and the cross-target hoist degrades to a no-op (same
            # behaviour as before the hoist). Storing references only - the binned dataset is
            # shared with whatever held it before; the clear merely drops the template's pointer.
            _capture_dataset_reuse_cache(ctx, mlframe_model_name, original_model, pp_name=pre_pipeline_name)
            _maybe_clear_shim_cache(original_model)
            # ens_models snapshots may also hold the cache by reference (forward-transfer at clone() copied
            # the reference rather than moving it); release on each so the binned dataset can be freed.
            if ens_models:
                for _ens_ns in ens_models:
                    _maybe_clear_shim_cache(getattr(_ens_ns, "model", None))

            # On a tier transition into a non-Polars strategy, release the pre-pipeline Polars originals.
            cur_tier = strategy.feature_tier()
            if prev_tier is not None and cur_tier != prev_tier and not strategy.supports_polars:
                if train_df_polars is not None:
                    # Same rationale as the entry-site release: locals AND ctx attributes must both drop their refs.
                    del train_df_polars, val_df_polars, test_df_polars
                    train_df_polars = val_df_polars = test_df_polars = None
                    # Selective drop: see same-shape comment at the non-polars-native entry site
                    # above for rationale. These cache references are suite-scoped now, so a blanket
                    # .clear() would also wipe pandas-tier entries that survived the polars release.
                    for _pl_only_key in [_k for _k in tier_dfs_cache if isinstance(_k, tuple) and len(_k) >= 2 and _k[1] == "pl"]:
                        tier_dfs_cache.pop(_pl_only_key, None)
                    tier_enum_map_cache.clear()
                    baseline_rss_mb = _release_ctx_polars_frames(ctx, baseline_rss_mb, df_size_mb, verbose=verbose, reason="tier transition")
                    if verbose:
                        logger.info("  Released pre-pipeline Polars originals (tier transition)")
            prev_tier = cur_tier

        _finalize_per_target_ensembling(
            ens_models=ens_models,
            train_df_transformed=train_df_transformed,
            behavior_config=behavior_config,
            ctx=ctx,
            cur_target_name=cur_target_name,
            current_common_params=locals().get("current_common_params"),
            common_params=common_params,
            pre_pipeline_name=pre_pipeline_name,
            models=models,
            target_type=target_type,
            metadata=metadata,
            verbose=verbose,
        )

    ctx.models = models
    ctx.metadata = metadata
    ctx.trainset_features_stats = trainset_features_stats
    # Merge ``pipeline_cache`` HIT / MISS counters into the per-suite cache_stats accumulator.
    # PipelineCache itself is local to this function (one instance per pre_pipeline sweep) so the
    # only handoff to finalize is this stash; later targets create fresh PipelineCaches whose hits
    # accumulate via ``+=`` into the suite-wide running totals.
    try:
        _cs_pc = ctx._cache_stats.setdefault("pipeline_cache", {"hits": 0, "misses": 0})
        _cs_pc["hits"] += int(getattr(pipeline_cache, "n_hits", 0))
        _cs_pc["misses"] += int(getattr(pipeline_cache, "n_misses", 0))
    except Exception as _pc_stats_err:
        logger.debug("pipeline_cache stats merge failed: %s", _pc_stats_err)
    # CODE-LOW-2 + CODE-LOW-4: slug_to_original_target_{type,name} and _non_neural_train_times
    # are mutable containers we already rebound on ctx (the slugs are bound by reference at the top
    # of this function and mutated in place; _non_neural_train_times is rebound to a fresh list each
    # target with a matching ``ctx._non_neural_train_times = _non_neural_train_times`` at that point).
    # The earlier writeback of these three was a no-op.
    ctx.train_df_polars = train_df_polars
    ctx.val_df_polars = val_df_polars
    ctx.test_df_polars = test_df_polars
    ctx.train_df_pd = train_df_pd
    ctx.val_df_pd = val_df_pd
    ctx.test_df_pd = test_df_pd
    ctx.filtered_train_df = filtered_train_df
    ctx.filtered_val_df = filtered_val_df
    ctx.pipeline = pipeline
    ctx.defer_pandas_conv = defer_pandas_conv
    ctx.baseline_rss_mb = baseline_rss_mb
    ctx.train_df_size_bytes_cached = train_df_size_bytes_cached
    ctx.val_df_size_bytes_cached = val_df_size_bytes_cached
