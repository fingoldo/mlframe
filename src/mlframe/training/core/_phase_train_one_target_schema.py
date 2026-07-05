"""Model-schema record build + persist for ``_train_one_target``.

Lifted out of the deepest weight loop body in ``_phase_train_one_target_body.py`` to shrink the parent toward the <700 LOC budget. The helper preserves the original control flow + side effects (stamps into ``metadata["model_schemas"][model_file_name]``, stamps into ``ctx._fs_report_cache``) bit-for-bit.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

from ..configs import TargetTypes as _TargetTypes

logger = logging.getLogger("mlframe.training.core._phase_train_one_target")


def _resolve_weight_schemas_and_warn_val_placement(
    sample_weights: Any,
    split_config: Any,
    ctx: Any,
) -> dict:
    """Resolve ``weight_schemas`` from caller-supplied sample_weights (or default uniform) and emit the per-suite WARN banners.

    SW-LOG-PER-PP-PER-TGT: emit the weighting-schema banner once per suite, not once per (target x pre_pipeline x weight). The weighting schema is suite-constant; identical lines repeated K_targets x K_pp times bloat the log without adding info.

    Backward val placement + recency weighting cancel each other's drift-proxy intent (val older than train, training biased to newest rows). Warn so the user picks one. VAL-PLACE-WARN-PP: gate behind a per-suite latch so the warning fires once, not per PP.
    """
    if sample_weights:
        weight_schemas = sample_weights
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
    return weight_schemas


def _clone_model_with_sticky_flags(
    original_model: Any,
    _cached_init_params: Any,
    _ngb_fallback_snapshot: Any,
    _forward_dataset_reuse_cache: Any,
    logger_obj: Any,
) -> tuple:
    """Clone ``original_model`` via sklearn.clone with NGBoost / CatBoost fallbacks and re-assert mlframe sticky flags.

    INTENTIONAL: clone() lives INSIDE the weight loop. Each weight schema produces a different trained model stored separately in models[type][target]; without per-iteration cloning all in-memory entries would alias to the same last-trained sklearn object and only the .dump snapshots would be correct.

    CatBoost wraps custom eval_metric objects internally; sklearn's identity check fails -> direct constructor call with get_params() produces an equivalent unfitted instance. NGBoost: get_params() exposes attributes the constructor doesn't accept -> ``_cached_init_params`` memoizes the inspect.signature lookup so the TypeError branch isn't paying ~0.5-1ms per hit; the cache lives at module scope keyed by ``id(cls)``. The constructor-kwargs dict itself (``{k:v for k in sig}``) is invariant across the weight loop, so cache the filtered dict on first hit and re-splat per iteration.

    sklearn.clone() strips non-param attributes; re-assert mlframe sticky flags so the calibration directive and the polars-fastpath-broken marker survive each iteration.

    Hand the XGB DMatrix / LGB Dataset reuse caches forward across clone() so the weight-schema loop (uniform -> recency on the same train_df) reuses the heavy binned dataset in place via set_label / set_weight instead of rebuilding.

    Returns the ``(cloned_model, _ngb_fallback_snapshot)`` tuple so the caller can keep the lazily-built snapshot across weight iterations.
    """
    try:
        from sklearn.base import clone as _sklearn_clone
        cloned_model = _sklearn_clone(original_model)
    except RuntimeError:
        cloned_model = type(original_model)(**original_model.get_params())
    except TypeError:
        _cls = type(original_model)
        _sig_params = _cached_init_params(_cls)
        if _ngb_fallback_snapshot is None:
            _raw = original_model.get_params(deep=False)
            _ngb_fallback_snapshot = {k: v for k, v in _raw.items() if k in _sig_params}
        cloned_model = _cls(**_ngb_fallback_snapshot)
    if getattr(original_model, "_mlframe_posthoc_calibrate", False):
        try:
            cloned_model._mlframe_posthoc_calibrate = True
        except Exception as _attr_err:
            logger_obj.debug("Could not set _mlframe_posthoc_calibrate on clone: %s", _attr_err)
    if getattr(original_model, "_mlframe_polars_fastpath_broken", False):
        try:
            cloned_model._mlframe_polars_fastpath_broken = True
        except Exception as _attr_err:
            logger_obj.debug("Could not set _mlframe_polars_fastpath_broken on clone: %s", _attr_err)
    _forward_dataset_reuse_cache(original_model, cloned_model)
    return cloned_model, _ngb_fallback_snapshot


def _maybe_render_friend_graph(ctx, pre_pipeline, _unwrap_selector, cur_target_name, pre_pipeline_name) -> None:
    """Render the MRMR friend graph once per (target, pre_pipeline) when charts are enabled.

    Routes through the reporting DSL (``ctx.reporting_config.plot_outputs``) so the same graph
    is written as interactive plotly HTML and a static image with no bespoke plotting. Guarded
    end-to-end -- a render failure must never affect training or the metadata write. The
    per-(target, pp) latch in ``ctx.artifacts`` avoids re-rendering the weight-invariant graph
    once per (model, weight) inner-loop iteration.
    """
    try:
        selector = _unwrap_selector(pre_pipeline)
        graph = getattr(selector, "friend_graph_", None)
        if graph is None or not getattr(graph, "nodes", None):
            return
        reporting_config = getattr(ctx, "reporting_config", None)
        output_config = getattr(ctx, "output_config", None)
        plot_outputs = getattr(reporting_config, "plot_outputs", None) if reporting_config else None
        plot_file = getattr(output_config, "plot_file", None) if output_config else None
        if not plot_outputs or not plot_file:
            return
        artifacts = getattr(ctx, "artifacts", None)
        rendered = artifacts.setdefault("_friend_graph_rendered", set()) if isinstance(artifacts, dict) else None
        latch_key = (cur_target_name, pre_pipeline_name)
        if rendered is not None and latch_key in rendered:
            return
        import re

        from mlframe.feature_selection.filters.friend_graph import plot_friend_graph
        _safe = re.sub(r"[^0-9A-Za-z._-]+", "_", f"{cur_target_name}_{pre_pipeline_name}".strip())
        plot_friend_graph(
            graph, plot_outputs=plot_outputs, base_path=f"{plot_file}_friend_graph_{_safe}",
            title=f"Feature friend graph - {cur_target_name} ({str(pre_pipeline_name).strip()})",
        )
        if rendered is not None:
            rendered.add(latch_key)
    except Exception:
        logger.warning("friend-graph render skipped for %s/%s", cur_target_name, pre_pipeline_name, exc_info=True)


def _build_and_record_model_schema(
    ctx: Any,
    metadata: dict,
    model_file_name: str,
    mlframe_model_name: str,
    weight_name: str,
    target_type: Any,
    strategy: Any,
    cur_target_name: str,
    cur_target_values: Any,
    _train_idx: Any,
    pre_pipeline: Any,
    pre_pipeline_name: str,
    train_df_transformed: Any,
    _schema_hash: Any,
    _input_schema: Any,
    _build_feature_selection_report: Any,
    _selector_params_hash: Any,
    _unwrap_selector: Any,
) -> None:
    """Persist this model's input-schema fingerprint in metadata so load-time can verify it against the serving frame.

    Multi-output extensions (target_type / n_classes / multilabel_strategy + schema_version) let load_mlframe_suite dispatch correctly; legacy artifacts without these fields fall back to binary inference.

    Per-model feature-selection report. ``pre_pipeline`` returned by ``process_model`` is the FITTED selector / pipeline (or None for the ordinary branch). ``train_df_transformed.columns`` gives the post-FS surviving features for both pandas and polars frames. The report is always stamped (selector_name=None for ordinary) so downstream consumers can rely on the key existing.

    Cache the report at (target, pp_name, model_name, selector_params_hash, kept_cols) because the fitted selector + kept columns are weight-invariant. The prior key used id(pre_pipeline) which is Python's memory-address; once an object is GC'd its id can be recycled, so a long-lived ``ctx._fs_report_cache`` could collide on a recycled address across the per-(target, model) inner loops. ``_selector_params_hash`` is content-derived and id-stable across recycling.
    """
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
        if isinstance(cur_target_values, (np.ndarray, pl.Series if pl is not None else ()))
        else cur_target_values.iloc[_train_idx]
    )
    try:
        if target_type == _TargetTypes.MULTILABEL_CLASSIFICATION:
            _record["n_classes"] = int(train_y.shape[1]) if hasattr(train_y, "shape") and train_y.ndim == 2 else None
            _record["multilabel_strategy"] = (
                "native" if (hasattr(strategy, "supports_native_multilabel") and strategy.supports_native_multilabel) else "wrapper"
            )
        elif target_type == _TargetTypes.MULTICLASS_CLASSIFICATION:
            _record["n_classes"] = int(len(np.unique(np.asarray(train_y)))) if hasattr(train_y, "shape") else None
            _record["multilabel_strategy"] = None
        else:
            _record["n_classes"] = None
            _record["multilabel_strategy"] = None
    except Exception as _intro_err:
        # Never fail the metadata write because of an introspection error on optional fields.
        # Surface as warning since load_mlframe_suite dispatches on n_classes/multilabel_strategy.
        logger.warning("n_classes/multilabel_strategy introspection failed for %s: %s", mlframe_model_name, _intro_err)

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

    _maybe_render_friend_graph(ctx, pre_pipeline, _unwrap_selector, cur_target_name, pre_pipeline_name)

    metadata.setdefault("model_schemas", {})[model_file_name] = _record
