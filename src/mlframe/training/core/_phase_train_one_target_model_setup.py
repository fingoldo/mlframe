"""Per-target model-setup phase, carved out of ``_train_one_target``
(in ``_phase_train_one_target_body``).

Runs the ``if mlframe_models:`` branch end-to-end: registers the slug
mapping, sets up plot/model paths, slices per-split targets, runs
feature-handling apply, per-target diagnostics + dummy baselines +
audit render, builds common_params / models_params via select_target,
applies the loss-recommendation and feature-drift auto-action,
and finally builds the pre_pipelines list. When ``mlframe_models``
is empty, the helper falls back to ``([], [])`` for the pre_pipelines
pair (no models = no inner loop iterations).

Re-imported at the parent's module bottom so historical
``from ._phase_train_one_target import _setup_per_target_mlframe_models``
keeps resolving transparently.
"""
from __future__ import annotations

import logging
from timeit import default_timer as timer
from typing import Any, Dict

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

from ..train_eval import select_target
from ..utils import log_ram_usage
from ._misc_helpers import _elapsed_str, _split_preds_probs
from ._phase_diagnostics import run_per_target_diagnostics
from ._phase_dummy_baselines import run_dummy_baselines
from ._phase_temporal_audit import _format_temporal_audit_report, _plot_target_over_time
from ._setup_helpers import (
    _build_common_params_for_target, _build_pre_pipelines,
    _setup_model_directories,
)

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _render_per_target_diagnostics(
    *,
    target_type,
    plot_file,
    save_charts,
    reporting_config,
    current_train_target,
    current_val_target,
    current_test_target,
    train_df,
    test_df,
    timestamps,
    test_idx,
    metadata,
    cur_target_name,
):
    """Render the once-per-target diagnostics (distribution overlay, adversarial validation, PSI drift) near suite start.

    Default-ON when charts are saved (``save_charts`` + ``plot_file`` + a ``plot_outputs`` DSL). These need no trained
    model: the distribution overlay compares per-split y, adversarial validation separates train-vs-test feature frames,
    and PSI uses the test frame + its timestamps. All builders cap their own compute, so 100GB frames stay safe.
    """
    plot_outputs = getattr(reporting_config, "plot_outputs", None)
    if not (save_charts and plot_file and plot_outputs):
        return
    tt = (str(target_type) or "").lower()
    task = "classification" if ("class" in tt or "label" in tt) else "regression"

    def _arr(v):
        if v is None:
            return None
        try:
            return v.to_numpy() if hasattr(v, "to_numpy") else np.asarray(v)
        except Exception:
            return None

    y_by_split = {}
    for name, val in (("train", current_train_target), ("val", current_val_target), ("test", current_test_target)):
        a = _arr(val)
        if a is not None and a.ndim == 1 and a.size > 0:
            y_by_split[name] = a

    charts_acc = metadata.setdefault("charts", {"saved": [], "failed": []})
    from mlframe.reporting.diagnostics_dispatch import (
        render_target_dist_overlay, render_target_drift_diagnostics,
    )

    if y_by_split:
        try:
            render_target_dist_overlay(
                y_true_by_split=y_by_split, task=task,
                plot_outputs=plot_outputs, base_path=f"{plot_file}_target",
                metrics_dict=charts_acc if isinstance(charts_acc, dict) else None,
            )
        except Exception as _e:
            logger.warning("per-target distribution overlay failed for target='%s': %s", cur_target_name, _e)

    _ts_test = None
    if timestamps is not None and test_idx is not None:
        try:
            _ts_test = np.asarray(timestamps)[test_idx]
        except Exception:
            _ts_test = None

    if train_df is not None and test_df is not None:
        try:
            render_target_drift_diagnostics(
                train_frame=train_df, test_frame=test_df,
                timestamps=_ts_test, task=task,
                plot_outputs=plot_outputs, base_path=f"{plot_file}_target",
                metrics_dict={"charts": charts_acc} if isinstance(charts_acc, dict) else None,
            )
        except Exception as _e:
            logger.warning("per-target drift/adversarial diagnostics failed for target='%s': %s", cur_target_name, _e)


def _setup_per_target_mlframe_models(
    *,
    ctx,
    target_type,
    cur_target_name,
    cur_target_values,
    metadata,
    slug_to_original_target_name,
) -> Dict[str, Any]:
    """Run the ``if mlframe_models:`` setup phase for one target.

    Returns a dict with the downstream-needed bindings. Caller unpacks
    via dict-access so missing keys raise KeyError on contract drift.

    Output keys:
        plot_file, model_file (paths or None)
        current_train_target, current_val_target, current_test_target
        metadata (mutated dict; same reference returned for symmetry)
        common_params, models_params, rfecv_models_params,
            cpu_configs, gpu_configs (None when mlframe_models is empty)
        pre_pipelines, pre_pipeline_names (always lists)
    """
    # Lazy import: parent re-exports these at its module bottom; a top-level
    # import would form a hard cycle.
    from ._phase_train_one_target import (
        _apply_loss_recommendation_in_place,
        _is_regression_target_type,
        _maybe_run_feature_handling_apply,
    )
    from ._phase_train_one_target import slugify

    mlframe_models = ctx.mlframe_models
    # ``rfecv_models_params`` is initialised here so a downstream reference
    # doesn't NameError when mlframe_models is empty (matches the pre-split
    # parent's initialiser at the same point in the function body).
    rfecv_models_params: Dict[str, Any] = {}

    if not mlframe_models:
        # _train_idx still derived even when no models train: downstream code
        # (e.g. metadata schema_version record) reads ``cur_target_values[_train_idx]``
        # to compute train_y for n_classes / multilabel_strategy introspection.
        _train_idx_fallback = (
            ctx.filtered_train_idx
            if ctx.filtered_train_idx is not None
            else ctx.train_idx
        )
        return {
            "plot_file": None,
            "model_file": None,
            "_train_idx": _train_idx_fallback,
            "current_train_target": None,
            "current_val_target": None,
            "current_test_target": None,
            "metadata": metadata,
            "common_params": None,
            "models_params": None,
            "rfecv_models_params": rfecv_models_params,
            "cpu_configs": None,
            "gpu_configs": None,
            "pre_pipelines": [],
            "pre_pipeline_names": [],
        }

    # ----- ctx unpack (only what's needed below) ------------------------------
    target_name = ctx.target_name
    model_name = ctx.model_name
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
    mrmr_kwargs = ctx.mrmr_kwargs
    rfecv_models = ctx.rfecv_models
    multilabel_dispatch_config = ctx.multilabel_dispatch_config
    custom_pre_pipelines = ctx.custom_pre_pipelines
    common_params_dict = ctx.common_params_dict
    group_ids = ctx.group_ids
    timestamps = ctx.timestamps
    sample_weights = ctx.sample_weights
    train_idx = ctx.train_idx
    test_idx = ctx.test_idx
    train_details = ctx.train_details
    val_details = ctx.val_details
    test_details = ctx.test_details
    fairness_subgroups = ctx.fairness_subgroups
    cat_features = ctx.cat_features
    text_features = ctx.text_features
    embedding_features = ctx.embedding_features
    _dropped_high_card_data = ctx._dropped_high_card_data
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
    trainset_features_stats = ctx.trainset_features_stats
    train_df_size_bytes_cached = ctx.train_df_size_bytes_cached
    val_df_size_bytes_cached = ctx.val_df_size_bytes_cached
    _all_target_audits = ctx._all_target_audits
    target_by_type = ctx.target_by_type

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

    # Calib slice (calib_size>0): raw calib rows + aligned target for the trainer's post-hoc-calibration predict.
    # calib_idx indexes the original full-df rows (disjoint from train/val/test by the splitter's asserts), so the
    # target slice mirrors current_test_target. None when no calib slice was carved.
    _calib_idx = getattr(ctx, "calib_idx", None)
    _calib_df = getattr(ctx, "calib_df", None)
    current_calib_target = None
    if _calib_df is not None and _calib_idx is not None and len(_calib_idx) > 0:
        current_calib_target = (
            cur_target_values[_calib_idx]
            if isinstance(cur_target_values, (np.ndarray, pl.Series))
            else cur_target_values.iloc[_calib_idx]
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
        filtered_val_df=_fhc_val_df,
        filtered_test_df=_fhc_test_df,
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
                # Route through the multi-backend DSL when reporting_config exposes plot_outputs
                # (e.g. "plotly[html]+matplotlib[png]") so the temporal-audit chart obeys the
                # same backend selection as every other suite plot. Falls back to matplotlib-only
                # PNG when plot_outputs is absent (legacy default).
                _plot_outputs = getattr(reporting_config, "plot_outputs", None)
                if _plot_outputs:
                    _plot_target_over_time(
                        _audit,
                        plot_outputs=_plot_outputs,
                        base_path=f"{plot_file}_target_temporal_audit",
                    )
                else:
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

    _render_per_target_diagnostics(
        target_type=target_type,
        plot_file=plot_file,
        save_charts=save_charts,
        reporting_config=reporting_config,
        current_train_target=current_train_target,
        current_val_target=current_val_target,
        current_test_target=current_test_target,
        train_df=filtered_train_df,
        test_df=test_df_pd,
        timestamps=timestamps,
        test_idx=test_idx,
        metadata=metadata,
        cur_target_name=cur_target_name,
    )

    if verbose:
        logger.info("select_target...")

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

    # Feature-drift auto-action layer.
    #
    # Behaviour gates from outermost to innermost:
    #   1. ``behavior_config.feature_drift_auto_apply_neural_overrides``
    #      (default False). When OFF, we still surface a WARN line with
    #      the empirically-grounded recommendation but DO NOT mutate the
    #      user's MLP config. Operators opt in explicitly after reading
    #      the docs; everyone else gets the MLP they configured.
    #
    #   2. ``recommend_neural_overrides`` payload on the per-target drift
    #      report. The sensor's per-target-type threshold table gates
    #      this: regression threshold=3.0 (grounded), classification
    #      threshold=None (paired study showed weak / heterogeneous
    #      correlation across DGPs).
    #
    #   3. ``"mlp" in mlframe_models``. If the model set doesn't include
    #      MLP there's nothing to override.
    #
    # When all three are satisfied: translate the sklearn-shape override
    # into the nested ``mlp_kwargs`` shape, deep-merge into a per-target
    # ``hyperparams_config.model_copy`` (other targets in the same suite
    # keep their original mlp_kwargs), and stamp
    # ``metadata["feature_drift_auto_action"]`` for observability.
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
        _auto_apply_enabled = bool(getattr(
            behavior_config, "feature_drift_auto_apply_neural_overrides", False,
        ))
        if _sklearn_override and "mlp" in mlframe_models and not _auto_apply_enabled:
            # WARN-only path. Surface the recommendation loudly so the
            # operator can copy-paste into their config if desired, but
            # the MLP keeps the user-configured topology.
            logger.warning(
                "[feature-drift-auto-action] target='%s' weighted_drift=%.2f "
                "above empirical threshold -- recommended MLP HPT override "
                "(sklearn-shape) = %s. AUTO-APPLY IS OFF "
                "(behavior_config.feature_drift_auto_apply_neural_overrides "
                "= False). MLP will train with the user-supplied "
                "mlp_kwargs unchanged. To enable per-target auto-apply, "
                "set feature_drift_auto_apply_neural_overrides=True. "
                "Grounded by profiling/bench_mlp_robustness_sweep_nonlinear.py "
                "(regression, 3520 trials).",
                cur_target_name,
                float(_fd_for_target.get("weighted_drift_score") or 0.0),
                _sklearn_override,
            )
            # Stamp the recommendation in metadata for downstream tooling
            # (post-mortem / dashboards) so the same payload that would
            # have been applied is still visible.
            metadata.setdefault("feature_drift_auto_action_skipped", {}) \
                .setdefault(str(target_type), {})[cur_target_name] = {
                    "reason": "auto_apply_disabled",
                    "sklearn_override_recommended": dict(_sklearn_override),
                    "weighted_drift_score": _fd_for_target.get("weighted_drift_score"),
                }
        elif _sklearn_override and "mlp" in mlframe_models and _auto_apply_enabled:
            from ..feature_drift_report import (
                translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs,
            )
            _orig_mlp_kwargs = (
                getattr(hyperparams_config, "mlp_kwargs", None) or {}
            )
            # Pass existing mlp_kwargs so the translator can preserve a
            # caller-pinned optimizer (e.g. MuonAdamWHybrid). Without it
            # the translator hardcoded ``optimizer=torch.optim.AdamW``
            # whenever ``alpha`` was present in the sklearn-shape override,
            # and the deep-merge below silently overwrote a user's Muon
            # choice.
            _mlframe_override = translate_sklearn_mlp_overrides_to_mlframe_mlp_kwargs(
                _sklearn_override,
                existing_mlp_kwargs=_orig_mlp_kwargs,
            )
            _untranslated = _mlframe_override.pop("__untranslated__", None)
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
                ">= per-type threshold -- applying empirically-grounded MLP "
                "HPT override (sklearn-shape: %s; mlframe-mlp_kwargs "
                "deep-merge: %s%s). Grounded by "
                "profiling/bench_mlp_robustness_sweep_nonlinear.py.",
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
        use_shap_proxied_fs=feature_selection_config.use_shap_proxied_fs,
        shap_proxied_fs_kwargs=feature_selection_config.shap_proxied_fs_kwargs,
        use_ace_fs=feature_selection_config.use_ace_fs,
        ace_kwargs=feature_selection_config.ace_kwargs,
        rfecv_cluster_reduce=feature_selection_config.rfecv_cluster_reduce,
        rfecv_cluster_corr_threshold=feature_selection_config.rfecv_cluster_corr_threshold,
        rfecv_cluster_min_reduction=feature_selection_config.rfecv_cluster_min_reduction,
        rfecv_cluster_corr_method=feature_selection_config.rfecv_cluster_corr_method,
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
        # Default FS selector seeds from the split seed for whole-pipeline reproducibility when the
        # operator did not pin a selector seed explicitly.
        fs_random_seed=getattr(getattr(ctx, "split_config", None), "random_seed", None),
        # Whether the split is group-aware. MRMR's MI is group-naive; it WARNS (not crashes) when groups are threaded.
        fs_use_groups=bool(getattr(getattr(ctx, "split_config", None), "use_groups", False)),
    )

    # Thread the raw calib frame + aligned target into common_params so they reach DataConfig via
    # _build_configs_from_params. The trainer transforms calib_df through the same fitted pre_pipeline as test and
    # stamps (calib_probs, calib_target) for finalize's auto-calibration. No-op when no calib slice was carved.
    if _calib_df is not None and current_calib_target is not None:
        common_params["calib_df"] = _calib_df
        common_params["calib_target"] = current_calib_target
        common_params["calib_idx"] = _calib_idx

    # Full-length row timestamps reach DataConfig.timestamps so the per-split reporter can slice them and render the
    # residual-vs-time / metric-over-time temporal-drift panels under the same FTE-timestamp gate as the target audit.
    if timestamps is not None:
        common_params["timestamps"] = timestamps

    return {
        "plot_file": plot_file,
        "model_file": model_file,
        "_train_idx": _train_idx,
        "current_train_target": current_train_target,
        "current_val_target": current_val_target,
        "current_test_target": current_test_target,
        "metadata": metadata,
        "common_params": common_params,
        "models_params": models_params,
        "rfecv_models_params": rfecv_models_params,
        "cpu_configs": cpu_configs,
        "gpu_configs": gpu_configs,
        "pre_pipelines": pre_pipelines,
        "pre_pipeline_names": pre_pipeline_names,
    }
