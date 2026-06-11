"""``_run_composite_target_wrapping`` -- composite-target ensemble wrapping.

Wave 100 (2026-05-21): split out from
``training/core/_phase_composite_post.py`` to keep that file below the
1k-line monolith threshold. Behaviour preserved bit-for-bit; the symbol
is re-exported from ``_phase_composite_post`` so existing imports
continue to work.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple  # noqa: F401

import numpy as np
import pandas as pd

# Wave 100: dependencies needed by the moved _run_composite_target_wrapping.
# _ADDITIVE_TRANSFORMS is defined inside the function body itself, not at
# module scope of any sibling -- the static analyzer flagged it as a missing
# reference but Python resolves it from function-local scope.
from ..composite import CompositeTargetEstimator, get_transform, _extract_base_matrix
from .._format import format_metric as _fmt, strip_shim_suffix as _strip

logger = logging.getLogger(__name__)

# Mirror parent's watchdog threshold so the moved function sees the same
# value it did pre-split. Kept in sync with _phase_composite_post.py:48.
_WATCHDOG_RELATIVE_THRESHOLD = 0.01


def _watchdog_base_columns(spec: dict) -> tuple[str, ...]:
    """Full ordered base-column tuple for a spec, mirroring the wrapper's ``_resolve_base_columns``.

    Multi-base specs (``linear_residual_multi`` and any future multi-base transform) carry the secondary bases in ``extra_base_columns``; the
    watchdog must feed the transform's ``forward``/``inverse`` the same ``(n, K)`` base matrix the wrapper uses, otherwise those calls raise
    "base has 1 columns but fitted alphas has K entries" and the watchdog silently swallows the error -- leaving the multi-base family (exactly
    the one most prone to wrapper-math bugs) with zero coverage. Single-base specs return a 1-tuple, so the 1-D fast path below is unchanged.
    """
    _bc = spec.get("base_column") if isinstance(spec, dict) else None
    if not _bc:
        return ()
    _extra = tuple(spec.get("extra_base_columns") or ()) if isinstance(spec, dict) else ()
    return (_bc, *_extra)


def _watchdog_extract_base(split_df: Any, base_columns: tuple[str, ...]) -> np.ndarray:
    """Extract the watchdog base array for ``base_columns`` from ``split_df``.

    For a single base column returns a 1-D float64 array (bit-identical to the prior ``np.asarray(split_df[col]).astype(np.float64)`` pull);
    for K>=2 columns returns the canonical ``(n, K)`` matrix via the same ``_extract_base_matrix`` helper the wrapper uses, so the transform's
    ``forward``/``inverse`` see all K bases. Format-native (no whole-frame copy): single-col path stays a narrow column pull, multi-col path
    routes through the polars ``.select`` / pandas ``.loc`` single-buffer extractor.
    """
    if len(base_columns) == 1:
        return np.asarray(split_df[base_columns[0]]).astype(np.float64)
    return _extract_base_matrix(split_df, base_columns)


def _emit_yscale_composite_chart(
    *,
    y_target: np.ndarray,
    y_pred: np.ndarray,
    inner_entry: Any,
    composite_name: str,
    orig_tname: str,
    target_name: str,
    plot_file: str | None,
    reporting_config: Any,
    rmse_y: float,
    mae_y: float,
    r2_y: float,
) -> None:
    """Emit a y-scale chart for a composite-target model.

    Called from the wrap pass on the TEST split. The chart's
    ``targets`` and ``preds`` are both already on the raw y scale
    (wrapper.predict returns y-scale; y_target is the original raw
    target sliced to test rows). Model name format mirrors raw-target
    reports (``MTTR/MTTS``) so the chart sits alongside them.
    """
    try:
        from ..evaluation import report_regression_model_perf
    except Exception:
        return
    if y_target.size == 0 or y_pred.size == 0:
        return
    # Derive a chart-friendly model_name. The original entry's
    # ``model_name`` attribute (if present) carries the inner backend
    # name; fall back to the inner class name.
    _outer = getattr(inner_entry, "model", None) or inner_entry
    _inner_class = type(getattr(_outer, "estimator_", _outer)).__name__
    # Header stats are the TEST-split mean/std (y_target is the test slice), distinct from the suite's MTTR (TRAIN-split mean) -- label them as such to avoid cross-reading drift.
    _mttr = float(np.mean(y_target))
    _mtts = float(np.std(y_target))
    chart_model_name = (
        f"{_inner_class} {target_name} {composite_name} "
        f"[y-scale wrap-pass] test_mean/test_std={_mttr:.2f}/{_mtts:.2f}"
    )
    # Per-composite plot file (so the chart doesn't overwrite the
    # raw-target one and shows up as a sibling file alongside it).
    if plot_file:
        if "." in plot_file:
            _stem, _ext = plot_file.rsplit(".", 1)
            _plot_path = f"{_stem}_yscale_{composite_name}.{_ext}"
        else:
            _plot_path = f"{plot_file}_yscale_{composite_name}"
    else:
        _plot_path = ""
    _plot_outputs = getattr(reporting_config, "plot_outputs", None) if reporting_config else None
    _plot_dpi = getattr(reporting_config, "plot_dpi", None) if reporting_config else None
    report_regression_model_perf(
        targets=y_target,
        preds=y_pred,
        columns=(),
        model_name=chart_model_name,
        model=None,
        report_title="TEST",
        print_report=True,
        show_perf_chart=True,
        plot_file=_plot_path,
        plot_outputs=_plot_outputs,
        plot_dpi=_plot_dpi,
    )
    logger.info(
        "[CompositeTargetEstimator] y-scale chart emitted for "
        "composite='%s' inner=%s (MAE=%.4g RMSE=%.4g R2=%.4f, "
        "n_test=%d)",
        composite_name, _inner_class, mae_y, rmse_y, r2_y, int(y_target.size),
    )


def emit_per_model_composite_y_scale_test(
    *,
    entry: Any,
    composite_spec: dict,
    orig_target_name: str,
    composite_name: str,
    target_name: str,
    y_full: np.ndarray,
    test_idx,
    test_df_pd,
    train_idx=None,
    plot_file: str | None = None,
    reporting_config: Any = None,
) -> None:
    """Wrap a freshly-fit composite-target inner model in
    CompositeTargetEstimator (IDEMPOTENT -- safe to call again at end-of-target)
    and emit a TEST-split y-scale chart + log a TEST-split y-scale metric line
    immediately, so composite targets get per-model feedback in the ORIGINAL
    scale right after each fit instead of only at end-of-target.

    Mutates ``entry.model`` (or ``entry`` itself) to the wrapper, matching
    what the end-of-target ``_run_composite_target_wrapping`` does. The
    downstream wrap-pass re-checks idempotency and skips already-wrapped
    entries for the wrap step; its multi-split metric block still runs for
    the comprehensive train/val/test table + watchdog. Never raises -- any
    failure is logged at WARNING and swallowed (training must not crash on
    a reporting hook).
    """
    try:
        if test_idx is None or test_df_pd is None or y_full is None:
            return
        _inner = getattr(entry, "model", None) or entry
        if _inner is None or not hasattr(_inner, "predict"):
            return
        # Idempotent: if already wrapped (e.g. recover_composite_y_scale_metrics
        # re-entry, or the per-model hook ran already), reuse the wrapper.
        if isinstance(_inner, CompositeTargetEstimator):
            _wrapper = _inner
        else:
            _extra = tuple(composite_spec.get("extra_base_columns") or ())
            _base_columns = (
                (composite_spec["base_column"], *_extra) if _extra else None
            )
            _y_full_arr = np.asarray(y_full)
            # y-clip envelope MUST be train-only: this wrapper persists (the
            # end-of-target pass skips already-wrapped entries for idempotency),
            # so a full-y envelope here would leak val/test range into the
            # post-inverse clip and flatter the reported TEST metrics. Slice to
            # the train rows when available; fall back to full y only when the
            # caller cannot supply train_idx (still no leak into the inner --
            # just a wider, conservative envelope).
            if train_idx is not None:
                try:
                    _y_train_arr = _y_full_arr[train_idx]
                except Exception:
                    _y_train_arr = _y_full_arr
            else:
                _y_train_arr = _y_full_arr
            _wrapper = CompositeTargetEstimator.from_fitted_inner(
                fitted_inner=_inner,
                transform_name=composite_spec["transform_name"],
                base_column=composite_spec["base_column"],
                base_columns=_base_columns,
                transform_fitted_params=composite_spec["fitted_params"],
                y_train=_y_train_arr,
            )
            # Mutate the entry so downstream callers (and the end-of-target
            # wrap-pass idempotency check) see the wrapped form.
            if hasattr(entry, "model"):
                try:
                    entry.model = _wrapper
                except Exception:
                    # Read-only attribute -- skip the in-place mutation; the
                    # end-of-target pass will rebuild the wrapper.
                    pass
        _y_arr = np.asarray(y_full)
        _y_test = _y_arr[test_idx]
        _y_pred = np.asarray(
            _wrapper.predict(test_df_pd), dtype=np.float64,
        ).reshape(-1)
        _finite = np.isfinite(_y_pred) & np.isfinite(_y_test)
        if int(_finite.sum()) == 0:
            return
        _yt = _y_test.astype(np.float64)[_finite]
        _yp = _y_pred[_finite]
        _diff = _yp - _yt
        _rmse = float(np.sqrt(np.mean(_diff * _diff)))
        _mae = float(np.mean(np.abs(_diff)))
        _ss_tot = float(np.sum((_yt - _yt.mean()) ** 2))
        _r2 = (1.0 - float(np.sum(_diff * _diff)) / _ss_tot) if _ss_tot > 0 else float("nan")
        # Inner class name for the log line, matching raw-target reports.
        _inner_for_label = (
            _wrapper.estimator_ if hasattr(_wrapper, "estimator_") else _inner
        )
        _inner_cls = type(_inner_for_label).__name__
        logger.info(
            "TEST %s %s %s [y-scale, per-model immediate] "
            "MAE=%.4f RMSE=%.4f R2=%.4f n=%d",
            _inner_cls, target_name, composite_name,
            _mae, _rmse, _r2, int(_finite.sum()),
        )
        _emit_yscale_composite_chart(
            y_target=_yt, y_pred=_yp,
            inner_entry=entry,
            composite_name=composite_name,
            orig_tname=orig_target_name,
            target_name=target_name,
            plot_file=plot_file,
            reporting_config=reporting_config,
            rmse_y=_rmse, mae_y=_mae, r2_y=_r2,
        )
        # Mark the entry so the end-of-target wrap pass skips re-emitting the identical test-split chart (same _yscale_{composite} path -> overwrite + duplicate predict).
        try:
            entry._yscale_chart_emitted = True
        except Exception:
            pass
    except Exception as _err:
        logger.warning(
            "[CompositeTargetEstimator] per-model y-scale emit failed for "
            "composite='%s' (non-fatal): %s",
            composite_name, _err,
        )


def _run_composite_target_wrapping(
    *,
    models: dict,
    metadata: dict,
    target_by_type: dict,
    composite_specs_by_target_type: dict,
    filtered_train_idx,
    filtered_train_df,
    filtered_val_idx,
    filtered_val_df,
    test_idx,
    test_df_pd,
    skip_predict: bool = False,
    enable_watchdog: bool = True,
    target_name: str | None = None,
    plot_file: str | None = None,
    reporting_config: Any = None,
) -> dict[tuple, np.ndarray]:
    """Wrap T-scale inner models in CompositeTargetEstimator so predict() returns y-scale; record y-scale RMSE/MAE/R2 per split.

    Mutates ``models`` in-place (replaces each composite-target inner with its wrapper) and writes ``metadata["composite_target_y_scale_metrics"]``.
    Returns the train-prediction cache (keyed by ``(id(wrapper), id(filtered_train_df), shape)``) so the downstream cross-target ensemble block can reuse the predictions
    without re-calling ``.predict`` on the wrapped models. Folding the frame identity into the key defends against ``id()`` recycling across GC cycles when wrappers
    or frames get freed between the wrap pass and the ensemble pass on long-lived suites.

    ``skip_predict=True`` (Pack 2026-05-18): skip the 3-split predict() calls used to compute y-scale RMSE/MAE/R2 metrics. The wrap step (replacing each entry's inner with ``CompositeTargetEstimator``) still runs so downstream predict-path consumers see y-scale predictions; only the metric computation block is bypassed. Pack G watchdog (additive transforms: T-MAE == y-MAE) already covers the correctness check, so the y-scale metrics are redundant when watchdog is on -- skipping them saves up to ~30 predict() calls on multi-million-row frames per composite target.
    """
    _train_pred_cache: dict[tuple, np.ndarray] = {}
    _train_frame_key = (id(filtered_train_df), getattr(filtered_train_df, "shape", None))
    for _tt_w, _by_name in (models or {}).items():
        if not isinstance(_by_name, dict):
            continue
        _tt_specs = composite_specs_by_target_type.get(str(_tt_w), {})
        if not _tt_specs:
            continue
        _name_to_spec: dict[str, tuple[str, dict[str, Any]]] = {}
        for _orig_tname, _spec_list in _tt_specs.items():
            for _spec in _spec_list:
                _name_to_spec[_spec["name"]] = (_orig_tname, _spec)
        for _composite_name, _entries in list(_by_name.items()):
            if _composite_name not in _name_to_spec:
                continue
            _orig_tname, _spec = _name_to_spec[_composite_name]
            # y_train for wrapping is the ORIGINAL y (not T) at the train rows the wrapper saw at fit time.
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
            _n_wrapped = 0
            for _i, _entry in enumerate(_entries):
                _inner = getattr(_entry, "model", None) or _entry
                if not hasattr(_inner, "predict"):
                    continue
                # Idempotency: if the entry is ALREADY a CompositeTargetEstimator (re-entry via recover_composite_y_scale_metrics), skip wrap.
                # Double-wrap would treat y-scale predict output as if it were T-scale and invert the transform a second time, producing garbage.
                if isinstance(_inner, CompositeTargetEstimator):
                    continue
                try:
                    # Multi-base specs (linear_residual_multi / future
                    # multi-base transforms) carry extra base columns
                    # alongside the primary. Build the full base_columns
                    # tuple so the wrapper's predict() reconstructs the
                    # (n, K) base matrix to match the K alphas saved in
                    # fitted_params. Without this the wrapper defaults to
                    # base_columns=None and _resolve_base_columns()
                    # falls back to (base_column,) -> predict raises
                    # "base has 1 columns but fitted alphas has K entries".
                    _extra = tuple(_spec.get("extra_base_columns") or ())
                    _base_columns = (
                        (_spec["base_column"], *_extra) if _extra else None
                    )
                    _wrapper = CompositeTargetEstimator.from_fitted_inner(
                        fitted_inner=_inner,
                        transform_name=_spec["transform_name"],
                        base_column=_spec["base_column"],
                        base_columns=_base_columns,
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
                # Preserve auxiliary metadata (columns, model_name, metrics) by replacing inner on entry.
                if hasattr(_entry, "model"):
                    try:
                        _entry.model = _wrapper
                    except Exception:
                        # Read-only attribute: replace the entry itself.
                        _entries[_i] = _wrapper
                        _n_wrapped += 1
                    else:
                        _n_wrapped += 1
                else:
                    _entries[_i] = _wrapper
                    _n_wrapped += 1
            logger.info(
                "[CompositeTargetEstimator] wrapped %d model(s) for composite "
                "target '%s'; predictions now y-scale.",
                _n_wrapped, _composite_name,
            )
            # Compute y-scale RMSE/MAE/R2 per split so composite is comparable to raw (per-target metrics were T-scale).
            # ``skip_predict``: bypass the per-split predict + metric block; wrap step above already ran so downstream
            # predict-path callers see y-scale predictions. Pack G watchdog on additive transforms (T-MAE == y-MAE) is
            # the correctness gate; the y-scale numbers here would just restate what the T-scale metrics already say.
            if skip_predict:
                logger.info(
                    "[CompositeTargetEstimator] composite='%s': wrap done, "
                    "y-scale metric block SKIPPED (skip_wrap_pass_predict=True). "
                    "T-scale metrics already in the per-target training log; "
                    "Pack G watchdog covers correctness for additive transforms.",
                    _composite_name,
                )
                # Even when the heavy multi-split metric block is skipped,
                # emit a SINGLE test-split y-scale chart per composite entry
                # so the operator gets the chart the user asked for in the
                # 2026-05-27 bug report. Cost: one wrapper.predict(test_df)
                # per entry (~0.1s booster, ~5s MLP). Cheap relative to the
                # full 3-split metric block (~5-15 min).
                if target_name is not None and test_idx is not None \
                        and test_df_pd is not None:
                    _y_full_chart = target_by_type.get(_tt_w, {}).get(_orig_tname)
                    if _y_full_chart is not None:
                        _y_arr_chart = np.asarray(_y_full_chart)
                        for _entry in _entries:
                            # The per-model hook already emitted this entry's test chart; skip to avoid a duplicate predict + overwrite of the same _yscale_{composite} file.
                            if getattr(_entry, "_yscale_chart_emitted", False):
                                continue
                            try:
                                _wrap_chart = getattr(_entry, "model", None) or _entry
                                _y_split_chart = _y_arr_chart[test_idx]
                                _y_pred_chart = np.asarray(
                                    _wrap_chart.predict(test_df_pd),
                                    dtype=np.float64,
                                ).reshape(-1)
                                _finite_chart = (
                                    np.isfinite(_y_pred_chart)
                                    & np.isfinite(_y_split_chart)
                                )
                                if _finite_chart.sum() == 0:
                                    continue
                                _y_t = _y_split_chart[_finite_chart]
                                _y_p = _y_pred_chart[_finite_chart]
                                _diff = _y_p - _y_t.astype(np.float64)
                                _rmse_c = float(np.sqrt(np.mean(_diff * _diff)))
                                _mae_c = float(np.mean(np.abs(_diff)))
                                _ss_tot_c = float(np.sum(
                                    (_y_t - _y_t.mean()) ** 2
                                ))
                                _r2_c = (
                                    1.0 - float(np.sum(_diff * _diff)) / _ss_tot_c
                                ) if _ss_tot_c > 0 else float("nan")
                                _emit_yscale_composite_chart(
                                    y_target=_y_t,
                                    y_pred=_y_p,
                                    inner_entry=_entry,
                                    composite_name=_composite_name,
                                    orig_tname=_orig_tname,
                                    target_name=target_name,
                                    plot_file=plot_file,
                                    reporting_config=reporting_config,
                                    rmse_y=_rmse_c, mae_y=_mae_c, r2_y=_r2_c,
                                )
                            except Exception as _chart_err:
                                logger.warning(
                                    "[CompositeTargetEstimator] y-scale chart "
                                    "emit failed for composite='%s' (non-fatal): %s",
                                    _composite_name, _chart_err,
                                )
                continue
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
                _entry_y_scores: dict[str, dict[str, float]] = {}
                for _split_name, _split_idx, _split_df in (
                    ("train", filtered_train_idx, filtered_train_df),
                    ("val", filtered_val_idx, filtered_val_df),
                    ("test", test_idx, test_df_pd),
                ):
                    if _split_idx is None or _split_df is None:
                        continue
                    try:
                        _y_split = _y_arr_metric[_split_idx]
                        # Wrapped (post-clip) prediction = today's headline value. Train RMSE here is optimistic by construction:
                        # the clip is [y_train_min, y_train_max], train rows are in-envelope, clip is a no-op. Val / test rows
                        # may drift outside; the clip then narrows the headline RMSE. To make that contribution explicit we ALSO
                        # capture the raw (pre-clip) prediction via ``predict_pre_clip`` and emit a parallel metric block.
                        _y_pred_wrapped = np.asarray(
                            _wrapper_for_score.predict(_split_df),
                            dtype=np.float64,
                        ).reshape(-1)
                        if hasattr(_wrapper_for_score, "predict_pre_clip"):
                            _y_pred_raw = np.asarray(
                                _wrapper_for_score.predict_pre_clip(_split_df),
                                dtype=np.float64,
                            ).reshape(-1)
                        else:
                            # Inner is not a CompositeTargetEstimator (raw / passthrough); raw == wrapped is the honest answer.
                            _y_pred_raw = _y_pred_wrapped
                        # Use wrapped predictions for sample-log, cache, and the headline metric block (back-compat).
                        _y_pred = _y_pred_wrapped
                        # Sample-log the first 3 (y_pred, y_true) pairs per split as a leakage / contract sanity check.
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
                        if _split_name == "train":
                            _train_pred_cache[(id(_wrapper_for_score),) + _train_frame_key] = _y_pred
                            # Inner-model key too: composite_post.py reads via ``getattr(comp, 'model', comp)`` which unwraps one level.
                            _inner_for_write = getattr(_wrapper_for_score, "model", None)
                            if _inner_for_write is not None and _inner_for_write is not _wrapper_for_score:
                                _train_pred_cache[(id(_inner_for_write),) + _train_frame_key] = _y_pred
                        _diff = _y_pred - _y_split.astype(np.float64)
                        _finite = np.isfinite(_diff)
                        if _finite.sum() == 0:
                            continue
                        # Zero-variance y => R2 undefined; emit NaN rather than 0.0 to mark the degenerate case.
                        _y_finite = _y_split.astype(np.float64)[_finite]
                        _ss_tot = float(np.sum(
                            (_y_finite - _y_finite.mean()) ** 2
                        ))
                        _ss_res = float(np.sum(
                            _diff[_finite] * _diff[_finite]
                        ))
                        _r2 = (1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")
                        _rmse_wrapped = float(np.sqrt(np.mean(_diff[_finite] * _diff[_finite])))
                        _mae_wrapped = float(np.mean(np.abs(_diff[_finite])))
                        # Raw (pre-clip) RMSE / MAE: align finite mask to raw predictions so any wrapped-only NaN doesn't
                        # bias the comparison. On in-envelope splits (train) raw and wrapped agree exactly.
                        _diff_raw = _y_pred_raw - _y_split.astype(np.float64)
                        _finite_raw = np.isfinite(_diff_raw)
                        if int(_finite_raw.sum()) > 0:
                            _rmse_raw = float(np.sqrt(np.mean(_diff_raw[_finite_raw] * _diff_raw[_finite_raw])))
                            _mae_raw = float(np.mean(np.abs(_diff_raw[_finite_raw])))
                        else:
                            _rmse_raw = float("nan")
                            _mae_raw = float("nan")
                        _entry_y_scores[_split_name] = {
                            "RMSE": _rmse_wrapped,
                            "MAE": _mae_wrapped,
                            "R2": _r2,
                            "n_rows_finite": int(_finite.sum()),
                            "RMSE_raw": _rmse_raw,
                            "RMSE_wrapped": _rmse_wrapped,
                            "MAE_raw": _mae_raw,
                            "MAE_wrapped": _mae_wrapped,
                        }
                        # 2026-05-27 user requirement: emit a Y-SCALE
                        # chart for composite models on the TEST split
                        # so it is directly comparable to raw-target
                        # charts (same MTTR/MTTS units, same scatter
                        # axes). The T-scale residual chart in
                        # ``_reporting_regression`` is skipped exactly
                        # to make room for this y-scale chart.
                        if _split_name == "test" and target_name is not None:
                            try:
                                _emit_yscale_composite_chart(
                                    y_target=_y_split.astype(np.float64)[_finite],
                                    y_pred=_y_pred[_finite],
                                    inner_entry=_entry,
                                    composite_name=_composite_name,
                                    orig_tname=_orig_tname,
                                    target_name=target_name,
                                    plot_file=plot_file,
                                    reporting_config=reporting_config,
                                    rmse_y=_rmse_wrapped, mae_y=_mae_wrapped, r2_y=_r2,
                                )
                            except Exception as _chart_err:
                                logger.warning(
                                    "[CompositeTargetEstimator] y-scale chart "
                                    "emit failed for composite='%s' (non-fatal): %s",
                                    _composite_name, _chart_err,
                                )
                        # HIGH#4 2026-05-18: watchdog short-circuit when caller disabled it.
                        # The check below does an extra wrapper.predict + inner.predict
                        # per (entry, split) so a wide model zoo can pay 10s+ on 4M-row
                        # frames. Caller passes ``enable_watchdog=False`` to skip.
                        if not enable_watchdog:
                            continue
                        # Pack G runtime watchdog: detects when wrapper math
                        # silently breaks. Pre-fix this covered only additive
                        # transforms via the ``error_T == error_y`` invariant.
                        # The extended check below works for ALL transforms
                        # via the fundamental contract: ``wrapper.predict(X)``
                        # must equal ``transform.inverse(inner.predict(X),
                        # base, params)`` modulo y-clip. If they diverge, the
                        # wrapper's predict path is corrupted (entry-mutation
                        # cache stale, inner double-scaled via TTR state loss,
                        # base column mismatch).
                        #
                        # Additive transforms ALSO get the original
                        # ``error_T == error_y`` check (legacy) since that's
                        # a stricter assertion (the MAE numbers must agree,
                        # not just per-row predictions).
                        try:
                            _spec_t_name = _spec.get("transform_name") if isinstance(_spec, dict) else None
                            _ADDITIVE_TRANSFORMS = {
                                "linear_residual", "linear_residual_robust",
                                "linear_residual_multi", "linear_residual_grouped",
                                "diff", "monotonic_residual", "quantile_residual",
                                "ewma_residual",
                            }
                            # Universal predict-vs-inverse-of-inner check (works for ALL transforms including multiplicative).
                            # If wrapper.predict diverges from manually-reconstructed inverse(inner.predict, base, params), the wrapper math is broken.
                            try:
                                _wi_uni = getattr(_wrapper_for_score, "estimator_", None)
                                # Resolve the FULL base set (primary + extra_base_columns) so multi-base specs feed the transform's
                                # inverse the same (n, K) matrix the wrapper uses; a 1-D pull would raise the alphas-width mismatch and
                                # the watchdog would swallow it at DEBUG, leaving linear_residual_multi entirely uncovered.
                                _bcs_uni = _watchdog_base_columns(_spec)
                                if (_wi_uni is not None and _spec_t_name
                                        and _bcs_uni
                                        and all(_c in _split_df for _c in _bcs_uni)):
                                    _bivar_uni = get_transform(_spec_t_name)
                                    _base_uni = _watchdog_extract_base(_split_df, _bcs_uni)
                                    _t_pred_uni = np.asarray(
                                        _wi_uni.predict(_split_df), dtype=np.float64,
                                    ).reshape(-1)
                                    _y_reconstructed = _bivar_uni.inverse(
                                        _t_pred_uni, _base_uni,
                                        _spec.get("fitted_params", {}),
                                    )
                                    _ru = _y_pred - _y_reconstructed
                                    _fu = np.isfinite(_ru)
                                    if int(_fu.sum()) > 0:
                                        _max_dev = float(np.max(np.abs(_ru[_fu])))
                                        _y_scale = float(np.std(_y_split[np.isfinite(_y_split)])) or 1.0
                                        _rel = _max_dev / _y_scale
                                        # ``_WATCHDOG_RELATIVE_THRESHOLD`` (module-level) carries the rationale; tune there.
                                        if _rel > _WATCHDOG_RELATIVE_THRESHOLD:
                                            logger.warning(
                                                "[CompositeTargetEstimator.watchdog.universal] "
                                                "composite='%s' split='%s' transform=%s inner=%s: "
                                                "wrapper.predict diverges from "
                                                "transform.inverse(inner.predict, base, params) "
                                                "by max abs=%.4f (%.2f%% of y_std). Wrapper "
                                                "math broken. (additive-error invariant check "
                                                "may also fire below for additive transforms.)",
                                                _composite_name, _split_name, _spec_t_name,
                                                type(_wi_uni).__name__, _max_dev, _rel * 100.0,
                                            )
                            except Exception as _uni_err:
                                logger.debug(
                                    "[CompositeTargetEstimator.watchdog.universal] check "
                                    "failed for composite='%s' split='%s': %s",
                                    _composite_name, _split_name, _uni_err,
                                )
                            if _spec_t_name in _ADDITIVE_TRANSFORMS:
                                _wi = getattr(_wrapper_for_score, "estimator_", None)
                                # Full base set including extra_base_columns: linear_residual_multi is additive but its forward needs the
                                # (n, K) matrix. Building a 1-D base here is what previously raised the alphas-width ValueError and was
                                # swallowed at DEBUG below, leaving this family with no T-MAE invariant coverage.
                                _bcs_add = _watchdog_base_columns(_spec)
                                if _wi is not None and _bcs_add and all(_c in _split_df for _c in _bcs_add):
                                    _bivar = get_transform(_spec_t_name)
                                    _base_arr = _watchdog_extract_base(_split_df, _bcs_add)
                                    _t_true = _bivar.forward(
                                        _y_split.astype(np.float64),
                                        _base_arr,
                                        _spec.get("fitted_params", {}),
                                    )
                                    _t_pred = np.asarray(
                                        _wi.predict(_split_df), dtype=np.float64,
                                    ).reshape(-1)
                                    _dt = _t_pred - _t_true
                                    _ft = np.isfinite(_dt)
                                    if int(_ft.sum()) > 0:
                                        _mae_t = float(np.mean(np.abs(_dt[_ft])))
                                        _drel = abs(_mae_t - _mae_wrapped) / max(_mae_t, 1e-9)
                                        if _drel > 0.01:
                                            # Pack #9 diagnostic dump: when the watchdog fires we want enough info in the log to ROOT-CAUSE the divergence on the next production run without re-running. Surface first 5 (y_true, y_pred, T_true, T_pred) tuples + per-inner statistics so the operator can see WHERE the divergence enters: the wrapper math (T_pred vs T_true), the inverse path (T_pred -> y_pred), or the post-clip step.
                                            _n_dbg = min(5, int(_ft.sum()))
                                            _dbg_rows = []
                                            _ft_idx = np.flatnonzero(_ft)[:_n_dbg]
                                            _base_is_multi = _base_arr.ndim > 1
                                            for _i in _ft_idx:
                                                # Multi-base specs carry a (n, K) base matrix; render the row as a bracketed K-vector,
                                                # so the dump stays readable instead of crashing on ``%.4f`` of an array.
                                                _base_repr = (
                                                    "[" + ", ".join(f"{_v:.4f}" for _v in _base_arr[_i]) + "]"
                                                    if _base_is_multi else f"{_base_arr[_i]:.4f}"
                                                )
                                                _dbg_rows.append(
                                                    f"y={_y_split[_i]:.4f}, "
                                                    f"y_hat={_y_pred[_i]:.4f}, "
                                                    f"T={_t_true[_i]:.4f}, "
                                                    f"T_hat={_t_pred[_i]:.4f}, "
                                                    f"base={_base_repr}"
                                                )
                                            _y_resid_sample = _y_pred[_ft_idx] - _y_split[_ft_idx]
                                            _t_resid_sample = _dt[_ft_idx]
                                            logger.warning(
                                                "[CompositeTargetEstimator.watchdog] "
                                                "composite='%s' split='%s' inner=%s: "
                                                "y-MAE=%.4f diverges from T-MAE=%.4f "
                                                "by %.1f%% (>1%%). Additive-invertible "
                                                "transform should give identical errors. "
                                                "Probable causes: (1) inner.predict NOT "
                                                "returning T-scale (TTR transformer_ "
                                                "state lost via clone/pickle), (2) "
                                                "wrapper.predict double-applies inverse, "
                                                "(3) base column at predict differs from "
                                                "fit. Diagnostic sample of first %d rows: "
                                                "%s. y-residuals first %d: %s; T-residuals "
                                                "first %d: %s.",
                                                _composite_name, _split_name,
                                                type(_wi).__name__,
                                                _mae_wrapped, _mae_t, _drel * 100.0,
                                                _n_dbg, " | ".join(_dbg_rows),
                                                _n_dbg,
                                                ", ".join(f"{v:.4f}" for v in _y_resid_sample.tolist()),
                                                _n_dbg,
                                                ", ".join(f"{v:.4f}" for v in _t_resid_sample.tolist()),
                                            )
                        except Exception as _watchdog_err:
                            logger.debug(
                                "[CompositeTargetEstimator.watchdog] check failed for "
                                "composite='%s' split='%s': %s",
                                _composite_name, _split_name, _watchdog_err,
                            )
                    except Exception as _split_err:
                        # Per-split metric block can fail on shape / predict
                        # mismatch (especially during composite-target rerank
                        # where wrapper.predict may raise on edge geometries).
                        # Log at DEBUG so per-target failures are visible in
                        # verbose logs without spamming WARN -- caller still
                        # gets the model in metadata; just missing this entry's
                        # split metrics.
                        logger.debug(
                            "[composite y-scale metrics] split='%s' composite='%s' skipped: %s",
                            _split_name, _composite_name, _split_err,
                        )
                        continue
                _metrics_dict.append({
                    "model_name": getattr(_entry, "model_name", None),
                    "metrics": _entry_y_scores,
                })
                # Log y-scale summary so composite numbers are comparable to raw-target models in script output.
                if _entry_y_scores:
                    _y_summary_parts: list[str] = []
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
                        # After wrapping _entry.model IS the CompositeTargetEstimator; drill into base_estimator for the actual inner type name.
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
    return _train_pred_cache
