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
from typing import Any, Dict, Optional, Sequence

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



# The four record/save helpers live in the parent ``diagnostics_dispatch``, which re-exports THIS module's
# render_* + _entry_score at its own bottom -- a mutual cycle. A top-level ``from .diagnostics_dispatch
# import ...`` was "cycle-safe" only when the parent imported FIRST; when a sibling (e.g. discover_tuners
# during ``refresh-all``) imports THIS module first, the top-level import re-enters the half-initialised
# parent, whose bottom then fails to find ``_entry_score`` (defined further down here). Delegate lazily
# instead: importing this module no longer triggers the parent at import time, so either load order works
# and the helpers resolve (from the module cache) on first actual call.
def _record(*args, **kwargs):
    from .diagnostics_dispatch import _record as _f
    return _f(*args, **kwargs)


def _record_path(*args, **kwargs):
    from .diagnostics_dispatch import _record_path as _f
    return _f(*args, **kwargs)


def _save_figure(*args, **kwargs):
    from .diagnostics_dispatch import _save_figure as _f
    return _f(*args, **kwargs)


def _save_spec(*args, **kwargs):
    from .diagnostics_dispatch import _save_spec as _f
    return _f(*args, **kwargs)


def render_model_comparison_diagnostic(
    *,
    per_model: Dict[str, Dict[str, Any]],
    task_type: str,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    metric: Optional[str] = None,
    seed: int = 0,
) -> bool:
    """Multi-model leaderboard. Default-ON when >=2 models were trained on the same task (single-model skips cheaply).

    ``per_model`` maps ``name -> {"y_true", "y_score"/"y_pred", "metrics"}``; the composer subsamples internally for
    the correlation heatmap, so the assembly is bounded regardless of n.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or not per_model or len(per_model) < 2:
        return False
    try:
        from mlframe.reporting.charts.model_comparison import compose_model_comparison_figure

        spec = compose_model_comparison_figure(per_model, task_type, metric=metric, seed=seed)
        ok = _save_spec(spec, plot_outputs, base_path + "_model_comparison")
        _record(charts, "model_comparison", ok)
        if ok:
            _record_path(charts, base_path + "_model_comparison")
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: model_comparison failed; continuing.")
        _record(charts, "model_comparison", False)
        return False


def _entry_score(entry: Any) -> Optional[np.ndarray]:
    """Per-row scalar test-split score from a suite model entry: positive-class proba, else point prediction."""
    probs = getattr(entry, "test_probs", None)
    if probs is not None:
        arr = np.asarray(probs)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 1].astype(np.float64)
        if arr.ndim == 1:
            return arr.astype(np.float64)
    preds = getattr(entry, "test_preds", None)
    if preds is not None:
        p = np.asarray(preds)
        if p.ndim == 1:
            return p.astype(np.float64)
    return None


def _flat_scalar_metrics(metrics: Any) -> Dict[str, float]:
    """Best-effort flat ``{name: float}`` from a (possibly nested) per-model test-metrics dict for the leaderboard."""
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = float(v)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                    out.setdefault(str(k2), float(v2))
    return out


def render_model_comparison_from_suite(
    *,
    model_entries: Sequence[Any],
    target_type: str,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    metric: Optional[str] = None,
    seed: int = 0,
) -> bool:
    """Assemble a per-target leaderboard from the suite's returned per-model entries and render it.

    ``model_entries`` are the ``SimpleNamespace`` records the suite returns under ``models[target_type][name]``
    (each carries ``test_target`` / ``test_probs`` / ``test_preds`` + ``metrics``). Default-ON contract: renders only
    when >=2 entries carry a usable test score on the same task. The composer subsamples internally, so assembly is
    bounded regardless of n. This is the post-all-models hook the suite finalize calls once per target.
    """
    per_model: Dict[str, Dict[str, Any]] = {}
    for i, e in enumerate(model_entries or []):
        yt = getattr(e, "test_target", None)
        ys = _entry_score(e)
        if yt is None or ys is None:
            continue
        yt = np.asarray(yt).ravel()
        m = min(len(yt), len(ys))
        if m == 0:
            continue
        name = str(getattr(e, "model_name", None) or type(getattr(e, "model", None)).__name__ or f"model_{i}")
        if name in per_model:
            name = f"{name}_{i}"
        per_model[name] = {
            "y_true": yt[:m], "y_score": ys[:m],
            "metrics": _flat_scalar_metrics(getattr(e, "metrics", {}).get("test") if isinstance(getattr(e, "metrics", None), dict) else None),
        }
    tt = (target_type or "").lower()
    task = "binary" if tt == "binary_classification" else ("regression" if "regress" in tt else tt)
    return render_model_comparison_diagnostic(
        per_model=per_model, task_type=task, plot_outputs=plot_outputs, base_path=base_path,
        metrics_dict=metrics_dict, metric=metric, seed=seed,
    )


def build_combined_html_report(
    *,
    base_path: str,
    chart_paths: Sequence[str],
    plot_outputs: str,
    title: str = "Model report",
    metrics_dict: Optional[dict] = None,
) -> Optional[str]:
    """Stitch the rendered per-(model, split) chart PNGs into one navigable HTML index. Assembly-only (no re-render).

    Looks for a ``<base>.png`` next to each recorded chart base path (the matplotlib renderer's output); missing
    artifacts are noted inline by the builder, never crash. Records the combined path in ``metrics_dict["charts"]``.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not base_path or not chart_paths or "png" not in (plot_outputs or "").lower():
        return None
    try:
        from mlframe.reporting.report_html import build_combined_report

        # Display worst feature-value slices (``_weak_slices``) before the per-split weak-segment heatmaps
        # (``_weak_segments``): the once-on-test slice ranking is the headline; the per-split heatmaps drill in after.
        ordered = list(chart_paths)
        slice_pos = [i for i, p in enumerate(ordered) if p and p.endswith("_weak_slices")]
        segs = [p for p in ordered if p and p.endswith("_weak_segments")]
        if slice_pos and segs:
            segset = set(segs)
            rest = [p for p in ordered if p not in segset]
            anchor = ordered[max(slice_pos)]
            ordered = []
            for p in rest:
                ordered.append(p)
                if p == anchor:
                    ordered.extend(segs)

        entries = []
        seen = set()
        for p in ordered:
            if not p or p in seen:
                continue
            seen.add(p)
            label = os.path.basename(p)
            png = p if p.lower().endswith(".png") else p + ".png"
            if not os.path.exists(png):
                # matplotlib renderer may suffix the backend (e.g. ``_pdp_ice.matplotlib.png``).
                alt = p + ".matplotlib.png"
                png = alt if os.path.exists(alt) else png
            entries.append(("charts", label, png))
        if not entries:
            return None
        out_path = base_path + "_report.html"
        build_combined_report(entries, title=title, out_path=out_path)
        _record(charts, "combined_html", True)
        if isinstance(metrics_dict, dict):
            charts.setdefault("combined_report", out_path)
        return out_path
    except Exception:
        logger.exception("diagnostics_dispatch: combined HTML report failed; continuing.")
        _record(charts, "combined_html", False)
        return None


def render_decile_table_diagnostic(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    n_deciles: int = 10,
) -> bool:
    """Binary decile gain/lift/KS table figure (the tabular complement to the GAIN curve). Default-ON for binary targets.

    A single O(n log n) score sort inside the builder; skips cheaply on a single-class target or absent score.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path:
        return False
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    m = min(len(yt), len(ys))
    if m == 0:
        return False
    try:
        from mlframe.reporting.charts.binary import binary_decile_table_figure

        fig = binary_decile_table_figure(yt[:m], ys[:m], n_deciles=n_deciles)
        out = base_path + "_decile_table"
        ok = _save_figure(fig, plot_outputs, out)
        _record(charts, "decile_table", ok)
        if ok:
            _record_path(charts, out)
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: decile_table failed; continuing.")
        _record(charts, "decile_table", False)
        return False


def render_model_card_diagnostic(
    *,
    task: str,
    y_true: np.ndarray,
    y_score: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    model_name: str = "model",
    split: str = "test",
) -> bool:
    """One-glance per-(model, split) model card. Default-ON when charts are saved; reuses the split's y_true + scores/preds.

    ``task`` is ``"binary"``/``"classification"`` (needs ``y_score``) or ``"regression"`` (needs ``y_pred``).
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or y_true is None:
        return False
    yt = np.asarray(y_true).ravel()
    if yt.size == 0:
        return False
    try:
        from mlframe.reporting.charts.model_card import compose_model_card_figure

        spec = compose_model_card_figure(
            task=task, y_true=yt,
            y_score=None if y_score is None else np.asarray(y_score, dtype=np.float64).ravel(),
            y_pred=None if y_pred is None else np.asarray(y_pred).ravel(),
            model_name=model_name, split=split,
        )
        out = base_path + "_model_card"
        ok = _save_spec(spec, plot_outputs, out)
        _record(charts, "model_card", ok)
        if ok:
            _record_path(charts, out)
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: model_card failed; continuing.")
        _record(charts, "model_card", False)
        return False


def render_prediction_stability_diagnostic(
    *,
    member_preds: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    seed: int = 0,
) -> bool:
    """Ensemble member-disagreement panels. Default-ON when an ``(n_rows, n_members)`` matrix with >=2 members is present.

    The composer subsamples its scatter internally; skips cheaply when fewer than 2 members are supplied.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or member_preds is None:
        return False
    mp = np.asarray(member_preds, dtype=np.float64)
    if mp.ndim != 2 or mp.shape[1] < 2:
        return False
    try:
        from mlframe.reporting.charts.prediction_stability import compose_prediction_stability_figure

        yt = None if y_true is None else np.asarray(y_true, dtype=np.float64).ravel()[: mp.shape[0]]
        spec = compose_prediction_stability_figure(mp, y_true=yt, seed=seed)
        out = base_path + "_prediction_stability"
        ok = _save_spec(spec, plot_outputs, out)
        _record(charts, "prediction_stability", ok)
        if ok:
            _record_path(charts, out)
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: prediction_stability failed; continuing.")
        _record(charts, "prediction_stability", False)
        return False


def _split_entry_arrays(entry: Any, split: str, task: str) -> Optional[Dict[str, np.ndarray]]:
    """Pull ``{y_true, y_score|y_pred}`` for one split from a suite model entry, or None when that split is absent."""
    yt = getattr(entry, f"{split}_target", None)
    if yt is None:
        return None
    yt = np.asarray(yt).ravel()
    if yt.size == 0:
        return None
    if task == "regression":
        preds = getattr(entry, f"{split}_preds", None)
        if preds is None:
            return None
        yp = np.asarray(preds).ravel()
        m = min(len(yt), len(yp))
        return {"y_true": yt[:m], "y_pred": yp[:m]} if m else None
    probs = getattr(entry, f"{split}_probs", None)
    ys: Optional[np.ndarray] = None
    if probs is not None:
        arr = np.asarray(probs)
        if arr.ndim == 2 and arr.shape[1] == 2:
            ys = arr[:, 1].astype(np.float64)
        elif arr.ndim == 1:
            ys = arr.astype(np.float64)
    if ys is None:
        preds = getattr(entry, f"{split}_preds", None)
        if preds is not None and np.asarray(preds).ndim == 1:
            ys = np.asarray(preds).astype(np.float64)
    if ys is None:
        return None
    m = min(len(yt), len(ys))
    return {"y_true": yt[:m], "y_score": ys[:m]} if m else None


def render_split_comparison_from_suite(
    *,
    entry: Any,
    target_type: str,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    model_name: str = "model",
    seed: int = 0,
) -> bool:
    """Cross-split overfit panel for ONE model, assembled from the entry's per-split arrays. Default-ON when >=2 usable splits.

    ``entry`` is the suite ``SimpleNamespace`` record carrying ``{train,val,test}_{target,probs,preds}``.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or entry is None:
        return False
    tt = (target_type or "").lower()
    task = "regression" if "regress" in tt else ("binary" if tt == "binary_classification" else "classification")
    per_split: Dict[str, Any] = {}
    for split in ("train", "val", "test"):
        arrs = _split_entry_arrays(entry, split, task)
        if arrs is not None:
            per_split[split] = arrs
    if len(per_split) < 2:
        return False
    try:
        from mlframe.reporting.charts.split_comparison import compose_split_comparison_figure

        spec = compose_split_comparison_figure(per_split, task, model_name=model_name, seed=seed)
        out = base_path + "_split_comparison"
        ok = _save_spec(spec, plot_outputs, out)
        _record(charts, "split_comparison", ok)
        if ok:
            _record_path(charts, out)
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: split_comparison failed; continuing.")
        _record(charts, "split_comparison", False)
        return False


def render_target_dist_overlay(
    *,
    y_true_by_split: Dict[str, np.ndarray],
    pred_by_split: Optional[Dict[str, np.ndarray]] = None,
    task: str,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
) -> bool:
    """Render the per-target y / prediction distribution overlay (R-3 / INV-11) once per target. Returns success."""
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or not y_true_by_split:
        return False
    from mlframe.reporting.charts.error_analysis import target_dist_overlay

    overlay_task = "classification" if task == "classification" else "regression"
    try:
        spec = target_dist_overlay(y_true_by_split, pred_by_split=pred_by_split, task=overlay_task)
        ok = _save_spec(spec, plot_outputs, base_path + "_target_dist")
        _record(charts, "target_dist", ok)
        return ok
    except Exception:
        logger.exception("diagnostics_dispatch: target_dist_overlay failed; continuing.")
        _record(charts, "target_dist", False)
        return False


__all__ = [
    # render_* diagnostics that live in the parent ``diagnostics_dispatch`` (split/target-drift/pdp/slice/
    # decision-curve/calibration-drift/target-acf/shap) are intentionally NOT re-exported here; they are not
    # defined in this carved-out module. Only the names actually defined below are listed.
    "render_target_dist_overlay",
    "render_model_comparison_diagnostic",
    "render_model_comparison_from_suite",
    "render_decile_table_diagnostic",
    "render_model_card_diagnostic",
    "render_prediction_stability_diagnostic",
    "render_split_comparison_from_suite",
    "build_combined_html_report",
    "DIAG_ROW_CAP",
    "DIAG_MAX_FEATURES",
]
