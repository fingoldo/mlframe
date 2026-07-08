"""Suite wiring for the error-analysis + drift diagnostic charts.

The chart builders in ``charts.error_analysis`` / ``charts.drift`` are task-agnostic and take explicit data; this module is
the glue that the training suite calls from its per-split / per-target hot path. It selects the right builders for the
data on hand, renders them through the active backend(s), and records every rendered grid in a ``charts`` accounting
dict (``{"saved": [...], "failed": [...]}``) so a run can assert chart presence while keeping the no-crash contract.

RAM safety is the governing constraint: the suite runs on 100GB+ frames. Every entry point pulls column views (never a
whole-frame copy), and the feature-frame-consuming builders (weak-segment tree, error-bias tagging, worst-K) are fed a
bounded row subsample that preserves the largest-error rows so the weak region is never sampled away. The drift /
adversarial builders already cap their own work (per-feature O(n) histograms, 200k/side adversarial fit).

cProfile (n=1.5M, matplotlib backend): split-error path ~2.9s render + ~2.7s weak-segment (tree fit already capped at
50k by the builder); drift path's compute floors live in the builders (adversarial 200k/side LightGBM fit is the lever,
PSI/residual are O(n) bincount/histogram). The bulk of a cold-process drift profile is the one-time import of
``training.evaluation`` (pulled by ``metric_over_time`` -> ``compute_ml_perf_by_time``), which is already loaded inside a
real suite run -- no actionable speedup in this wiring layer; the orchestration itself is O(builders).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Row cap for the feature-frame-consuming error-analysis builders. ``_resolve_feature_matrix`` densifies the pulled
# columns into one float64 matrix, so an unbounded frame would materialise a full dense copy; the tree only needs
# enough rows to RANK split features and the error-bias quantiles converge well below this. Worst-error rows are
# preserved in the subsample so the localisation verdict is unchanged.
DIAG_ROW_CAP: int = 100_000
# Hard ceiling on feature columns handed to the dense-matrix builders; a several-hundred-column frame at the row cap is
# still bounded, but a pathological thousands-of-columns engineered frame would blow the dense matrix up.
DIAG_MAX_FEATURES: int = 200


def _record(charts: Optional[dict], name: str, ok: bool) -> None:
    """Append ``name`` to ``charts["saved"]`` or ``charts["failed"]`` depending on ``ok``; no-op when ``charts`` isn't a dict (diagnostics run with tracking disabled)."""
    if not isinstance(charts, dict):
        return
    bucket = charts.setdefault("saved" if ok else "failed", [])
    bucket.append(name)


def _save_spec(spec, plot_outputs: str, base_path: str) -> bool:
    """Render + save a FigureSpec via the DSL backends. Returns True on success, False (logged) on any failure."""
    try:
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save

        render_and_save(spec, parse_plot_output_dsl(plot_outputs), base_path)
        return True
    except Exception:
        logger.exception("diagnostics_dispatch: rendering %s failed; continuing.", base_path)
        return False


def _save_figure(fig, plot_outputs: str, base_path: str) -> bool:
    """Save a raw matplotlib Figure (builders that emit a Figure, not a FigureSpec) to ``base_path.png`` when png is requested.

    Mirrors the matplotlib renderer's on-disk name so ``build_combined_html_report`` can stitch it. Returns True on success.
    """
    # ``plt.close`` MUST run on every exit (early non-png return, savefig failure, success); otherwise a builder that
    # hands us a Figure on a non-png run -- or whose savefig raises -- leaks it into matplotlib's global registry,
    # which grows unbounded across the per-split/per-target hot path. Hence the close lives in ``finally``.
    try:
        if "png" not in (plot_outputs or "").lower():
            return False
        try:
            fig.savefig(base_path + ".png", bbox_inches="tight")
            return True
        except Exception:
            logger.exception("diagnostics_dispatch: saving figure %s failed; continuing.", base_path)
            return False
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close(fig)
        except Exception:  # nosec B110 - optional dependency import guard
            pass


def _column_names(frame: Any) -> Optional[List[str]]:
    """Stringified column names of ``frame`` (pandas/polars), or None if it has no ``.columns`` attribute."""
    cols = getattr(frame, "columns", None)
    if cols is None:
        return None
    return [str(c) for c in list(cols)]


def _row_count(frame: Any) -> int:
    """Row count of ``frame``, format-agnostic: prefers ``.shape[0]``, falls back to ``len()``, else 0."""
    shape = getattr(frame, "shape", None)
    if shape is not None:
        return int(shape[0])
    try:
        return len(frame)
    except TypeError:
        return 0


def _subset_rows(frame: Any, idx: np.ndarray) -> Any:
    """Return a row view of ``frame`` at integer positions ``idx`` (pandas .iloc / polars [idx] / ndarray fancy index)."""
    if frame is None:
        return None
    if hasattr(frame, "iloc"):
        return frame.iloc[idx]
    try:
        return frame[idx]
    except Exception:
        return np.asarray(frame)[idx]


def _select_feature_columns(frame: Any, feature_names: Optional[Sequence[str]], cap: int):
    """Cap the feature set to ``cap`` columns (column views, no copy). Returns ``(frame_or_view, names)``."""
    names = list(feature_names) if feature_names is not None else _column_names(frame)
    if names is None:
        return frame, None
    if len(names) <= cap:
        return frame, names
    keep = names[:cap]
    if hasattr(frame, "loc") and not isinstance(frame, np.ndarray):
        try:
            return frame[keep], keep
        except Exception:
            return frame, names
    return frame, names


def _bounded_sample_idx(n: int, loss: np.ndarray, seed: int = 0) -> np.ndarray:
    """Indices of a <=DIAG_ROW_CAP subsample preserving the largest-|loss| rows (so the weak region survives)."""
    if n <= DIAG_ROW_CAP:
        return np.arange(n, dtype=np.int64)
    from mlframe.reporting.charts import subsample_preserving_extremes

    return subsample_preserving_extremes(
        loss, sample_size=DIAG_ROW_CAP, extreme_values=loss,
        k_extremes=min(DIAG_ROW_CAP // 10, n), rng=seed,
    )


def render_split_error_diagnostics(
    *,
    df: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    feature_names: Optional[Sequence[str]] = None,
    feature_importances: Optional[Sequence[float]] = None,
    subgroups: Optional[Dict[str, np.ndarray]] = None,
    timestamps: Optional[np.ndarray] = None,
    worst_k: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    """Render the per-split error-analysis diagnostics (weak segments, error bias, worst-K) and account each grid.

    Default-ON when ``df`` + ``plot_outputs`` + ``base_path`` are present. ``task`` is ``"regression"`` or
    ``"classification"``. Returns ``{"worst_k_indices": ndarray, "worst_k_table": DataFrame|None}`` so the caller can
    surface the worst-K table in its metrics dict and (where the scatter supports it) highlight those points.

    Large-n: the feature-frame-consuming builders are fed a bounded, worst-error-preserving row subsample; worst-K is
    computed on the FULL arrays (O(n) argpartition) so the returned highlight indices map onto the caller's full data.
    """
    out: Dict[str, Any] = {"worst_k_indices": np.empty(0, dtype=np.int64), "worst_k_table": None}
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if df is None or not plot_outputs or not base_path:
        return out

    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    n = min(len(yt), _row_count(df))
    if n == 0:
        return out
    yt, yp = yt[:n], yp[:n]

    from mlframe.reporting.charts.error_analysis import (
        _per_row_error,
        error_bias_per_feature,
        segments_bar,
        weak_segment_heatmap,
        worst_k_table,
    )

    loss = _per_row_error(yt, yp, task=task)
    loss_finite = np.where(np.isfinite(loss), loss, -np.inf)

    # Worst-K table on the FULL arrays so the highlight indices map onto the caller's data (no frame densify needed --
    # only the K worst rows pull feature values). Surface the table in the caller's metrics dict.
    sample_idx = _bounded_sample_idx(n, loss_finite, seed=seed)
    sub_df, names = _select_feature_columns(_subset_rows(df, sample_idx), feature_names, DIAG_MAX_FEATURES)
    try:
        wk = worst_k_table(
            _select_feature_columns(df, feature_names, DIAG_MAX_FEATURES)[0] if n <= DIAG_ROW_CAP else sub_df,
            yt if n <= DIAG_ROW_CAP else yt[sample_idx],
            yp if n <= DIAG_ROW_CAP else yp[sample_idx],
            task=task, k=worst_k, feature_names=names, feature_importances=feature_importances,
            timestamps=(list(timestamps[:n]) if timestamps is not None and n <= DIAG_ROW_CAP else None),
        )
        out["worst_k_table"] = wk.table
        # Map subsample-local indices back to original positions when subsampled.
        out["worst_k_indices"] = (wk.indices if n <= DIAG_ROW_CAP else sample_idx[wk.indices]).astype(np.int64)
    except Exception:
        logger.exception("diagnostics_dispatch: worst_k_table failed; continuing.")

    yt_s = yt if n <= DIAG_ROW_CAP else yt[sample_idx]
    yp_s = yp if n <= DIAG_ROW_CAP else yp[sample_idx]

    try:
        res = weak_segment_heatmap(sub_df, yt_s, yp_s, task=task, feature_names=names, seed=seed)
        ok = _save_spec(res.figure, plot_outputs, base_path + "_weak_segments")
        _record(charts, "weak_segments", ok)
    except Exception:
        logger.exception("diagnostics_dispatch: weak_segment_heatmap failed; continuing.")
        _record(charts, "weak_segments", False)

    try:
        eb_res = error_bias_per_feature(sub_df, yt_s, yp_s, feature_names=names)
        ok = _save_spec(eb_res.figure, plot_outputs, base_path + "_error_bias")
        _record(charts, "error_bias", ok)
    except Exception:
        logger.exception("diagnostics_dispatch: error_bias_per_feature failed; continuing.")
        _record(charts, "error_bias", False)

    # Per-subgroup metric bars: only when the caller supplies subgroup masks (fairness frame). Built from a small
    # per-group error table so a 100GB frame never densifies here.
    if subgroups:
        try:
            import pandas as pd

            rows = []
            for name, mask in subgroups.items():
                m = np.asarray(mask).ravel()[:n].astype(bool)
                if not m.any():
                    continue
                rows.append({"group": str(name), "metric": float(np.nanmean(loss[m])), "count": int(m.sum())})
            if rows:
                global_metric = float(np.nanmean(loss))
                spec = segments_bar(
                    pd.DataFrame(rows), group_col="group", metric_col="metric",
                    global_value=global_metric, metric_name=("|resid|" if task == "regression" else "loss"),
                    higher_is_worse=True,
                )
                ok = _save_spec(spec, plot_outputs, base_path + "_segments")
                _record(charts, "segments", ok)
        except Exception:
            logger.exception("diagnostics_dispatch: segments_bar failed; continuing.")
            _record(charts, "segments", False)

    return out


def render_target_drift_diagnostics(
    *,
    train_frame: Any,
    test_frame: Any,
    val_frame: Any = None,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    task: str = "regression",
    plot_outputs: str = "",
    base_path: str = "",
    metrics_dict: Optional[dict] = None,
    feature_names: Optional[Sequence[str]] = None,
    metric: str = "roc_auc",
    seed: int = 0,
    calibration_drift: bool = True,
    target_acf: bool = True,
    cusum_drift: bool = True,
) -> None:
    """Render the per-target temporal-drift + adversarial-validation diagnostics, each accounted.

    ``psi_heatmap`` + ``residual_vs_time`` + ``metric_over_time`` fire when ``timestamps`` cover the split (same gate as
    the temporal target audit); ``adversarial_validation`` fires when train + test (or train + val) feature frames are
    available. When timestamps cover the split, ``calibration_drift`` (classification) + ``target_acf`` also emit
    default-on (both cheap: O(n) warmed njit / FFT-capped). All builders cap their own compute, so 100GB frames stay
    safe (column-view histograms, 200k/side fit).
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path:
        return

    from mlframe.reporting.charts.drift import (
        adversarial_validation,
        metric_over_time,
        psi_heatmap,
        residual_vs_time,
    )

    has_time = timestamps is not None and len(np.asarray(timestamps)) > 0

    if has_time and test_frame is not None:
        ts = np.asarray(timestamps)
        try:
            spec = psi_heatmap(test_frame, ts[: _row_count(test_frame)], feature_names=feature_names)
            ok = _save_spec(spec, plot_outputs, base_path + "_psi")
            _record(charts, "psi_heatmap", ok)
        except Exception:
            logger.exception("diagnostics_dispatch: psi_heatmap failed; continuing.")
            _record(charts, "psi_heatmap", False)

    if has_time and y_true is not None and y_pred is not None:
        ts = np.asarray(timestamps)
        yt = np.asarray(y_true).ravel()
        yp = np.asarray(y_pred).ravel()
        m = min(len(yt), len(yp), len(ts))
        if m > 0:
            if task == "regression":
                try:
                    spec = residual_vs_time(yt[:m], yp[:m], ts[:m])
                    ok = _save_spec(spec, plot_outputs, base_path + "_residual_vs_time")
                    _record(charts, "residual_vs_time", ok)
                except Exception:
                    logger.exception("diagnostics_dispatch: residual_vs_time failed; continuing.")
                    _record(charts, "residual_vs_time", False)
                # CUSUM change-point catches a SUSTAINED residual mean shift that per-bucket residual_vs_time misses.
                if cusum_drift:
                    try:
                        from mlframe.reporting.charts.drift import cusum_residual_drift

                        spec = cusum_residual_drift(yt[:m], yp[:m], ts[:m])
                        ok = _save_spec(spec, plot_outputs, base_path + "_cusum_drift")
                        _record(charts, "cusum_drift", ok)
                        if ok:
                            _record_path(charts, base_path + "_cusum_drift")
                    except Exception:
                        logger.exception("diagnostics_dispatch: cusum_drift failed; continuing.")
                        _record(charts, "cusum_drift", False)
            try:
                # Direction from the canonical metric-direction table, not a 2-item
                # allowlist that mislabeled rmse/mae/mape/log_loss/ice/ece/pinball as
                # higher-is-better and inverted the "over time" trend annotation.
                from mlframe.training.metrics_registry import metric_name_higher_is_better
                _dir = metric_name_higher_is_better(metric)
                higher_is_better = True if _dir is None else _dir
                spec = metric_over_time(yt[:m], yp[:m], ts[:m], metric=metric, higher_is_better=higher_is_better)
                ok = _save_spec(spec, plot_outputs, base_path + "_metric_over_time")
                _record(charts, "metric_over_time", ok)
            except Exception:
                logger.exception("diagnostics_dispatch: metric_over_time failed; continuing.")
                _record(charts, "metric_over_time", False)

            # Calibration drift over time -- classification only (y_pred is the positive-class probability here).
            if calibration_drift and task != "regression":
                render_calibration_drift_diagnostic(
                    y_true=yt[:m], y_score=yp[:m], timestamps=ts[:m],
                    plot_outputs=plot_outputs, base_path=base_path, metrics_dict=metrics_dict,
                )
            # Target serial-dependence ACF/PACF on the time-ordered target.
            if target_acf:
                render_target_acf_diagnostic(
                    y_true=yt[:m], timestamps=ts[:m],
                    plot_outputs=plot_outputs, base_path=base_path, metrics_dict=metrics_dict,
                )

    if train_frame is not None and (test_frame is not None or val_frame is not None):
        try:
            spec = adversarial_validation(
                train_frame, test_frame if test_frame is not None else val_frame,
                val_frame=val_frame if test_frame is not None else None,
                feature_names=feature_names, seed=seed,
            )
            ok = _save_spec(spec, plot_outputs, base_path + "_adversarial")
            _record(charts, "adversarial", ok)
        except Exception:
            logger.exception("diagnostics_dispatch: adversarial_validation failed; continuing.")
            _record(charts, "adversarial", False)


def _record_path(charts: Optional[dict], path: str) -> None:
    """Append a rendered-artifact base path to the charts accounting so the combined report can stitch it."""
    if isinstance(charts, dict) and path:
        charts.setdefault("paths", []).append(path)


def render_pdp_ice_diagnostic(
    *,
    model: Any,
    df: Any,
    feature_names: Optional[Sequence[str]],
    feature_importances: Optional[Sequence[float]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    top_features: int = 4,
    sample: int = 2_000,
    grid: int = 20,
    seed: int = 0,
) -> bool:
    """PDP/ICE for the top feature-importance features. Default-ON when a fitted model + feature frame are present.

    The composer subsamples rows to ``sample`` before any predict, so cost is ``grid`` predicts independent of n
    (RAM-safe on 100GB frames -- the carrier frame is never copied, only a row view is sampled inside the composer).
    Skips cheaply when the model cannot predict, the frame is empty, or no features can be ranked.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or not plot_outputs or not base_path:
        return False
    if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    if not names:
        return False
    # Rank by importance when available, else take the first columns; cap to the top-N legible features.
    if feature_importances is not None and len(feature_importances) == len(names):
        order = np.argsort(np.asarray(feature_importances, dtype=np.float64))[::-1]
        ranked = [names[i] for i in order]
    else:
        ranked = names
    top = ranked[: max(1, int(top_features))]
    interaction = (top[0], top[1]) if len(top) >= 2 else None
    try:
        from mlframe.reporting.charts.pdp_ice import compose_pdp_figure

        spec = compose_pdp_figure(
            model, df, top, grid=grid, sample=sample, interaction_pair=interaction, seed=seed,
        )
        ok = _save_spec(spec, plot_outputs, base_path + "_pdp_ice")
        _record(charts, "pdp_ice", ok)
        if ok:
            _record_path(charts, base_path + "_pdp_ice")
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: pdp_ice failed; continuing.")
        _record(charts, "pdp_ice", False)
        return False


def render_pdp_2d_diagnostic(
    *,
    model: Any,
    df: Any,
    feature_names: Optional[Sequence[str]],
    feature_importances: Optional[Sequence[float]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    sample: int = 2_000,
    grid: int = 20,
    seed: int = 0,
) -> bool:
    """2-D PDP surface for the top interacting feature pair (opt-in). The composer picks the pair (top SHAP-interaction
    pair when available, else top-2 importances) and caps sample_rows + grid internally, so cost is ``grid`` predicts
    independent of n. Best-effort: any failure is logged and swallowed so the report never aborts.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or not plot_outputs or not base_path:
        return False
    if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    if not names or len(names) < 2:
        return False
    # Rank by importance when available so the SHAP-less fallback pair is the top-2 most important, mirroring pdp_ice.
    feat_x = feat_y = None
    if feature_importances is not None and len(feature_importances) == len(names):
        order = np.argsort(np.asarray(feature_importances, dtype=np.float64))[::-1]
        feat_x, feat_y = names[int(order[0])], names[int(order[1])]
    try:
        from mlframe.reporting.charts.pdp_2d import compose_pdp_2d_figure

        fig = compose_pdp_2d_figure(model, df, feat_x, feat_y, grid=grid, sample_rows=sample, seed=seed)
        ok = _save_figure(fig, plot_outputs, base_path + "_pdp_2d")
        _record(charts, "pdp_2d", ok)
        if ok:
            _record_path(charts, base_path + "_pdp_2d")
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: pdp_2d failed; continuing.")
        _record(charts, "pdp_2d", False)
        return False


def _interaction_cost_within_budget(model: Any, df: Any, k: int, grid: int, sample: int, max_seconds: float) -> bool:
    """Project the H-statistic 2-D PDP cost from one timed predict and compare to ``max_seconds``.

    The dominant work is ``C(k,2)`` 2-D PDP surfaces at ``grid^2`` predict-batches of ``sample`` rows each, plus ``k``
    1-D PDPs at ``grid`` batches. One predict on the sample gives the per-batch latency; the projection is model-agnostic
    (a slow deep net self-skips, a fast tree runs). On any probe failure we allow the render (fail-open) -- the render
    itself is best-effort and swallows errors.
    """
    import time as _time

    try:
        n = _row_count(df)
        if n == 0 or k < 2:
            return True
        probe = _subset_rows(df, np.arange(min(sample, n), dtype=np.int64))
        fn = getattr(model, "predict_proba", None) or getattr(model, "predict", None)
        if fn is None:
            return True
        t0 = _time.perf_counter()
        fn(probe)
        t_batch = _time.perf_counter() - t0
        n_pairs = k * (k - 1) // 2
        projected = (n_pairs * grid * grid + k * grid) * t_batch
        return projected <= float(max_seconds)
    except Exception:
        logger.debug("interaction_strength: cost probe failed; allowing render.", exc_info=True)
        return True


def render_interaction_strength_diagnostic(
    *,
    model: Any,
    df: Any,
    feature_names: Optional[Sequence[str]],
    feature_importances: Optional[Sequence[float]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    max_features: int = 8,
    sample: int = 2_000,
    grid: int = 20,
    max_seconds: float = 20.0,
    seed: int = 0,
) -> bool:
    """Friedman-Popescu H-statistic heatmap over the top feature-importance features. Reuses the PDP machinery.

    Cost is C(k,2) 2-D PDP surfaces (grid^2 predicts each), so total time scales with the model's predict latency. A
    per-predict TIME probe projects the full cost and skips (logged) when it exceeds ``max_seconds`` -- so a fast/small
    model runs it and a slow one self-skips. Best-effort: any failure is logged and swallowed so the report never aborts.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or not plot_outputs or not base_path:
        return False
    if not (hasattr(model, "predict") or hasattr(model, "predict_proba")):
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    if not names or len(names) < 2:
        return False
    if feature_importances is not None and len(feature_importances) == len(names):
        order = np.argsort(np.asarray(feature_importances, dtype=np.float64))[::-1]
        ranked = [names[int(i)] for i in order]
    else:
        ranked = names
    top = ranked[: max(2, int(max_features))]
    if max_seconds and max_seconds > 0 and not _interaction_cost_within_budget(model, df, len(top), grid, sample, max_seconds):
        logger.info(
            "[diagnostics] interaction_strength: projected 2-D PDP cost over budget (%d features, grid=%d, "
            "%.0fs cap) -- skipping; raise ReportingConfig.interaction_strength_max_seconds to force.",
            len(top), grid, float(max_seconds),
        )
        return False
    try:
        from mlframe.reporting.charts.interaction_strength import compose_interaction_strength_figure

        spec = compose_interaction_strength_figure(
            model, df, top, max_features=max_features, grid=grid, sample=sample, seed=seed,
        )
        ok = _save_spec(spec, plot_outputs, base_path + "_interaction_strength")
        _record(charts, "interaction_strength", ok)
        if ok:
            _record_path(charts, base_path + "_interaction_strength")
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: interaction_strength failed; continuing.")
        _record(charts, "interaction_strength", False)
        return False


def render_slice_finder_diagnostic(
    *,
    df: Any,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    feature_names: Optional[Sequence[str]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    seed: int = 0,
) -> bool:
    """Multi-dim weak-slice search on the precomputed per-row error. Default-ON (no model calls).

    Feeds a bounded, worst-error-preserving row subsample + capped feature set (column views) so a 100GB frame
    never densifies. Surfaces the worst-slice table in ``metrics_dict["weak_slices"]`` alongside the bar figure.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if df is None or not plot_outputs or not base_path:
        return False
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    n = min(len(yt), len(yp), _row_count(df))
    if n == 0:
        return False
    yt, yp = yt[:n], yp[:n]

    from mlframe.reporting.charts.error_analysis import _per_row_error
    from mlframe.reporting.charts.slice_finder import find_weak_slices

    loss = _per_row_error(yt, yp, task=task)
    loss_finite = np.where(np.isfinite(loss), loss, -np.inf)
    idx = _bounded_sample_idx(n, loss_finite, seed=seed)
    sub_df, names = _select_feature_columns(_subset_rows(df, idx), feature_names, DIAG_MAX_FEATURES)
    try:
        res = find_weak_slices(
            sub_df, yt[idx], yp[idx], task=task, feature_names=names, seed=seed,
        )
        ok = _save_spec(res.figure, plot_outputs, base_path + "_weak_slices")
        _record(charts, "weak_slices", ok)
        if ok:
            _record_path(charts, base_path + "_weak_slices")
        if isinstance(metrics_dict, dict):
            metrics_dict["weak_slices"] = res.table
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: slice_finder failed; continuing.")
        _record(charts, "weak_slices", False)
        return False


def render_decision_curve_diagnostic(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    model_label: str = "model",
) -> bool:
    """Binary decision-curve (net-benefit) analysis. Default-ON for binary targets; needs only y_true + score."""
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path:
        return False
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    m = min(len(yt), len(ys))
    if m == 0:
        return False
    try:
        from mlframe.reporting.charts.decision_curve import build_decision_curve_spec

        res = build_decision_curve_spec(yt[:m], ys[:m], model_label=model_label)
        ok = _save_spec(res.figure, plot_outputs, base_path + "_decision_curve")
        _record(charts, "decision_curve", ok)
        if ok:
            _record_path(charts, base_path + "_decision_curve")
        if isinstance(metrics_dict, dict):
            metrics_dict["decision_curve_useful"] = bool(getattr(res, "useful", False))
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: decision_curve failed; continuing.")
        _record(charts, "decision_curve", False)
        return False


def render_calibration_drift_diagnostic(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    timestamps: np.ndarray,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    n_windows: int = 10,
    n_bins: int = 10,
) -> bool:
    """Binary calibration drift over equal-population time windows. Default-ON when a split timestamp is present.

    O(n) (one argsort + warmed njit ECE per window); skips cheaply when timestamps are absent or all-equal.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or timestamps is None:
        return False
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    ts = np.asarray(timestamps).ravel()
    m = min(len(yt), len(ys), len(ts))
    if m < n_windows * 2:
        return False
    try:
        from mlframe.reporting.charts.calibration_drift import build_calibration_drift_spec, calibration_drift

        res = calibration_drift(yt[:m], ys[:m], ts[:m], n_windows=n_windows, n_bins=n_bins)
        spec = build_calibration_drift_spec(res)
        ok = _save_spec(spec, plot_outputs, base_path + "_calibration_drift")
        _record(charts, "calibration_drift", ok)
        if ok:
            _record_path(charts, base_path + "_calibration_drift")
        if isinstance(metrics_dict, dict):
            metrics_dict["calibration_drift_trend"] = float(getattr(res, "ece_trend", float("nan")))
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: calibration_drift failed; continuing.")
        _record(charts, "calibration_drift", False)
        return False


def render_target_acf_diagnostic(
    *,
    y_true: np.ndarray,
    timestamps: np.ndarray,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
) -> bool:
    """Target ACF/PACF (serial dependence) when the split carries timestamps. Default-ON; the series is tail-capped."""
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or timestamps is None:
        return False
    yt = np.asarray(y_true).ravel()
    ts = np.asarray(timestamps)
    if yt.size < 8 or len(ts) < yt.size:
        return False
    try:
        from mlframe.reporting.charts.temporal import compose_target_acf_figure

        order = np.argsort(ts[: yt.size], kind="stable")
        spec = compose_target_acf_figure(yt[order], suptitle="Target ACF / PACF")
        ok = _save_spec(spec, plot_outputs, base_path + "_target_acf")
        _record(charts, "target_acf", ok)
        if ok:
            _record_path(charts, base_path + "_target_acf")
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: target_acf failed; continuing.")
        _record(charts, "target_acf", False)
        return False


def render_shap_diagnostic(
    *,
    model: Any,
    df: Any,
    feature_names: Optional[Sequence[str]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    max_rows: int = 20_000,
    top_k: int = 6,
    allow_kernel: bool = False,
    seed: int = 0,
) -> bool:
    """SHAP beeswarm + top-K dependence. Default-ON for TREE models; non-tree uses the slow KernelExplainer only when
    ``allow_kernel`` is set. Uses shap's matplotlib-savefig path (not render_and_save -- these are figures, not specs).

    Rows are stratified-subsampled to ``max_rows`` inside the builder (high-|score| tail kept) before any SHAP work,
    so cost scales with the explained-row cap, not n (RAM-safe: the carrier frame is row-viewed, never copied).
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or not plot_outputs or not base_path:
        return False
    try:
        from mlframe.reporting.charts.shap_panels import is_tree_model, shap_summary_and_dependence
    except Exception:
        logger.debug("diagnostics_dispatch: shap unavailable; skipping shap panels.", exc_info=True)
        return False
    if not is_tree_model(model) and not allow_kernel:
        return False
    try:
        res = shap_summary_and_dependence(
            model, df, feature_names=list(feature_names) if feature_names else None,
            max_rows=max_rows, top_k=top_k, plot_file=base_path + "_shap.png",
            plot_outputs=plot_outputs, allow_kernel=allow_kernel, seed=seed,
        )
        ok = bool(res.paths) and res.skipped is None
        _record(charts, "shap_panels", ok)
        for p in res.paths:
            _record_path(charts, os.path.splitext(p)[0])
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: shap panels failed; continuing.")
        _record(charts, "shap_panels", False)
        return False


def render_shap_interactions_diagnostic(
    *,
    model: Any,
    df: Any,
    feature_names: Optional[Sequence[str]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    max_rows: int = 2_000,
    top_pairs: int = 10,
    seed: int = 0,
) -> bool:
    """Top feature-PAIR SHAP interactions (opt-in). Tree models only -- interaction values are O(F^2) per row, so
    the builder caps rows hard. Best-effort: any failure is logged and never aborts the report."""
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or not plot_outputs or not base_path:
        return False
    try:
        from mlframe.reporting.charts.shap_interactions import shap_interaction_summary
        from mlframe.reporting.charts.shap_panels import is_tree_model
    except Exception:
        logger.debug("diagnostics_dispatch: shap unavailable; skipping shap interactions.", exc_info=True)
        return False
    if not is_tree_model(model):
        return False
    try:
        res = shap_interaction_summary(
            model, df, feature_names=list(feature_names) if feature_names else None,
            max_rows=max_rows, top_pairs=top_pairs, plot_file=base_path + "_shap_interactions.png",
            plot_outputs=plot_outputs, seed=seed,
        )
        ok = bool(res.paths) and res.skipped is None
        _record(charts, "shap_interactions", ok)
        for p in res.paths:
            _record_path(charts, os.path.splitext(p)[0])
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: shap interactions failed; continuing.")
        _record(charts, "shap_interactions", False)
        return False


def render_shap_per_instance_diagnostic(
    *,
    model: Any,
    df: Any,
    y_true: Any,
    y_score: Any,
    feature_names: Optional[Sequence[str]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    k: int = 4,
    max_explain_rows: int = 2_000,
    seed: int = 0,
) -> bool:
    """Per-instance SHAP attribution for the top-K most-confident-wrong predictions (opt-in). Tree models only;
    needs a 1-D ``y_score`` (binary positive-class prob or regression prediction). Best-effort: failures logged, never abort."""
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if model is None or df is None or y_true is None or y_score is None or not plot_outputs or not base_path:
        return False
    try:
        from mlframe.reporting.charts.shap_panels import is_tree_model
        from mlframe.reporting.charts.shap_per_instance import shap_worst_errors_explanation
    except Exception:
        logger.debug("diagnostics_dispatch: shap unavailable; skipping shap per-instance.", exc_info=True)
        return False
    if not is_tree_model(model):
        return False
    try:
        res = shap_worst_errors_explanation(
            model, df, y_true, y_score, feature_names=list(feature_names) if feature_names else None,
            k=k, max_explain_rows=max_explain_rows, plot_file=base_path + "_shap_per_instance.png",
            plot_outputs=plot_outputs, seed=seed,
        )
        ok = bool(res.paths) and res.skipped is None
        _record(charts, "shap_per_instance", ok)
        for p in res.paths:
            _record_path(charts, os.path.splitext(p)[0])
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: shap per-instance failed; continuing.")
        _record(charts, "shap_per_instance", False)
        return False


# comparison/decile/model-card/stability/split render helpers carved to _diagnostics_dispatch_extra.py (1k-LOC ceiling).
from ._diagnostics_dispatch_extra import (
    _entry_score,
    _flat_scalar_metrics,
    _split_entry_arrays,
    build_combined_html_report,
    render_category_discriminability_diagnostic,
    render_class_structure_diagnostic,
    render_decile_table_diagnostic,
    render_engineered_separability_diagnostic,
    render_model_card_diagnostic,
    render_model_comparison_diagnostic,
    render_model_comparison_from_suite,
    render_prediction_stability_diagnostic,
    render_split_comparison_from_suite,
    render_target_dist_overlay,
)
