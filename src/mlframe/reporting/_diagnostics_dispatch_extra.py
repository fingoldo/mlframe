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
import numbers
import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# DIAG_ROW_CAP / DIAG_MAX_FEATURES are NOT redefined here -- the actual consumer (and single source of
# truth) is diagnostics_dispatch.py:34,37; this module never uses either constant itself, only re-exports
# them via __all__ below for callers that import from this submodule directly. A top-level `from
# .diagnostics_dispatch import DIAG_ROW_CAP, DIAG_MAX_FEATURES` would reintroduce the exact half-initialised-
# parent hazard the lazy _record/_record_path/_save_figure delegates above already document (a sibling
# importing THIS module first would trigger diagnostics_dispatch's bottom import of this module while it's
# still partially initialised). Resolve lazily via module __getattr__ instead, deferring the cross-import
# past both modules' load time.
def __getattr__(name: str):
    """Lazily resolve DIAG_ROW_CAP/DIAG_MAX_FEATURES from diagnostics_dispatch.py (the single source of
    truth) on first access, avoiding a duplicate top-level definition that could drift from the original."""
    if name in ("DIAG_ROW_CAP", "DIAG_MAX_FEATURES"):
        from . import diagnostics_dispatch

        return getattr(diagnostics_dispatch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The four record/save helpers live in the parent ``diagnostics_dispatch``, which re-exports THIS module's
# render_* + _entry_score at its own bottom -- a mutual cycle. A top-level ``from .diagnostics_dispatch
# import ...`` was "cycle-safe" only when the parent imported FIRST; when a sibling (e.g. discover_tuners
# during ``refresh-all``) imports THIS module first, the top-level import re-enters the half-initialised
# parent, whose bottom then fails to find ``_entry_score`` (defined further down here). Delegate lazily
# instead: importing this module no longer triggers the parent at import time, so either load order works
# and the helpers resolve (from the module cache) on first actual call.
def _record(*args, **kwargs):
    """Lazy delegate to the parent module's chart-outcome recorder (see the module docstring for the
    mutual-import reason this isn't a top-level import); appends a saved/failed entry to the ``charts`` dict."""
    from .diagnostics_dispatch import _record as _f
    return _f(*args, **kwargs)


def _record_path(*args, **kwargs):
    """Lazy delegate to the parent module's helper that records a successfully rendered chart's output path."""
    from .diagnostics_dispatch import _record_path as _f
    return _f(*args, **kwargs)


def _save_figure(*args, **kwargs):
    """Lazy delegate to the parent module's matplotlib-figure saver (writes per ``plot_outputs``, returns success)."""
    from .diagnostics_dispatch import _save_figure as _f
    return _f(*args, **kwargs)


def _save_spec(*args, **kwargs):
    """Lazy delegate to the parent module's chart-spec saver (renders a composed spec through the active
    backend(s) and writes it per ``plot_outputs``, returns success)."""
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
        return bool(ok)
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


def _is_real_scalar(v) -> bool:
    """True for a real numeric scalar, including numpy's -- and excluding bools.

    ``isinstance(v, (int, float))`` misses every numpy scalar except ``np.float64``: ``np.float32`` does not
    inherit from ``float`` and ``np.int64`` does not inherit from ``int``. LightGBM and XGBoost routinely
    hand back ``float32`` metrics, so that test silently DROPPED those models' AUC/logloss from the
    leaderboard -- an omission with no warning, indistinguishable from "the model did not report it".
    """
    return isinstance(v, (numbers.Real, np.number)) and not isinstance(v, (bool, np.bool_))


def _flat_scalar_metrics(metrics: Any) -> Dict[str, float]:
    """Best-effort flat ``{name: float}`` from a (possibly nested) per-model test-metrics dict for the leaderboard.

    Merge precedence: a top-level scalar key ALWAYS wins over a same-named key from any nested sub-dict. Among
    nested sub-dicts themselves, the LAST one (in ``metrics`` iteration order) wins on a name collision --
    consistent with a plain dict's own last-write-wins semantics.
    """
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    top_level_keys: set = set()
    for k, v in metrics.items():
        if _is_real_scalar(v):
            out[str(k)] = float(v)
            top_level_keys.add(str(k))
    for v in metrics.values():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if str(k2) in top_level_keys:
                    continue  # a top-level scalar of the same name always wins.
                if _is_real_scalar(v2):
                    out[str(k2)] = float(v2)  # last nested sub-dict wins on a nested-vs-nested collision.
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


# Chart-name -> (report section, human label). The combined report used to file EVERY chart under one
# section literally named "charts", labelled by raw filename basename, so a 40-chart report was one
# undifferentiated list of strings like "MRMR LGBMClassifier_val_weak_segments". Grouping by what the chart
# answers, and naming it in words, is the difference between an index and a directory listing.
_CHART_SECTIONS: tuple = (
    ("Calibration", ("calib", "reliability", "decile_table", "fairness")),
    ("Discrimination", ("binary_panels", "multiclass", "multilabel", "roc", "model_card", "model_comparison")),
    ("Errors and weak segments", ("weak_seg", "weak_slices", "error_bias", "segments", "worst_k")),
    ("Explainability", ("shap", "pdp", "interaction", "feature")),
    ("Drift and stability", ("psi", "drift", "cusum", "over_time", "acf", "stability", "adversarial")),
    ("Decision quality", ("decision_curve", "risk_coverage", "gain", "threshold")),
    ("Training", ("training_curve", "learning_curve")),
)


def _classify_chart(basename: str) -> tuple:
    """Map a chart's file basename to ``(section, human label)`` for the combined report's navigation.

    Unrecognised names fall into "Other" and keep their basename, so a new chart is never dropped or
    mislabelled -- it just does not get a curated home until it is added above.
    """
    low = basename.lower()
    section = "Other"
    for name, keys in _CHART_SECTIONS:
        if any(k in low for k in keys):
            section = name
            break
    # The suite prefixes every artifact with "<model> <split>_", so the chart's own name is whatever follows
    # the LAST split marker. Splitting on the final underscore instead would keep only the last word and turn
    # "weak_segments" into "segments" and "decision_curve" into "curve".
    label = basename
    for marker in ("_val_", "_test_", "_train_", "_oof_", "_calib_"):
        pos = low.rfind(marker)
        if pos != -1:
            label = basename[pos + len(marker) :]
            break
    return section, label.replace("_", " ").strip() or basename


def _candidate_paths(base: str, fmt: str, backends) -> list:
    """Every place ``render_and_save`` could have written ``base`` in ``fmt``, most likely first.

    The report builder LOOKS UP files it did not write, so it has to know both layouts: the per-format subfolder
    (the default) and the flat one, each with and without the backend infix a multi-output run adds. Reconstructing
    only one of them silently drops the image from the report rather than failing.
    """
    out: list = []
    for stem in (f"{base}.{fmt}", *(f"{base}.{b}.{fmt}" for b in backends)):
        directory, name = os.path.split(stem)
        out.append(os.path.join(directory, fmt, name))
        out.append(stem)
    return out


def _find_html_fragment(base: str) -> Optional[str]:
    """Return an interactive plotly fragment for ``base`` when one was written, else ``None``.

    Lets a ``plotly[html]``-only run still produce a combined index: the report builder embeds a fragment
    exactly as happily as a PNG, so there is no reason for the html-only configuration to get no report.
    """
    from mlframe.reporting.output import BACKEND_FORMATS

    for cand in _candidate_paths(base, "html", BACKEND_FORMATS):
        if os.path.exists(cand):
            try:
                with open(cand, encoding="utf-8") as fh:
                    return fh.read()
            except OSError:
                return None
    return None


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
    # The report builder embeds a plotly HTML fragment just as happily as a PNG, so gating the WHOLE report on
    # "png in plot_outputs" meant a `plotly[html]`-only run -- the interactive-first configuration -- produced
    # no combined index at all. Any renderable output is enough; the per-entry lookup below picks whichever
    # artifact actually exists on disk.
    _outputs = (plot_outputs or "").lower()
    if not base_path or not chart_paths or not _outputs:
        return None
    try:
        from mlframe.reporting.output import BACKEND_FORMATS
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

        # Heterogeneous by design: a PNG entry is (section, label, png) and an interactive one is
        # (section, label, None, fragment); build_combined_report accepts both tuple arities.
        entries: list = []
        seen = set()
        for p in ordered:
            if not p or p in seen:
                continue
            seen.add(p)
            label = os.path.basename(p)
            png = p if p.lower().endswith(".png") else ""
            if not png:
                png = next((c for c in _candidate_paths(p, "png", BACKEND_FORMATS) if os.path.exists(c)), p + ".png")
            if not os.path.exists(png):
                # No PNG (e.g. a plotly[html]-only run): fall back to the interactive fragment so the entry
                # still appears in the index instead of being dropped.
                _frag = _find_html_fragment(p)
                if _frag is not None:
                    section, nice = _classify_chart(os.path.basename(p))
                    entries.append((section, nice, None, _frag))
                    continue
            section, nice = _classify_chart(label)
            entries.append((section, nice, png))
        if not entries:
            return None
        out_path = base_path + "_report.html"
        build_combined_report(entries, title=title, out_path=out_path)
        _record(charts, "combined_html", True)
        if isinstance(metrics_dict, dict) and charts is not None:
            # Assign, do not setdefault: `setdefault` kept the FIRST path, so rebuilding a report (a
            # re-render into a new directory, or a second call in the same run) left the metrics dict
            # pointing at the previous, now-stale document.
            charts["combined_report"] = out_path
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
        if ok is None:
            return False  # png not requested; nothing rendered, nothing to record either way
        _record(charts, "decile_table", ok)
        if ok:
            _record_path(charts, out)
        return bool(ok)
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
        return bool(ok)
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

        yt = None if y_true is None else np.asarray(y_true, dtype=np.float64).ravel()
        # member_test_preds and test_target can come from different upstream slices (e.g. a coarse ensemble
        # re-scoring pass over more rows than the target was subsampled to), and EITHER can be the shorter one,
        # so both are aligned to the shorter length before ``abs_error`` broadcasts them together.
        if yt is not None and yt.shape[0] != mp.shape[0]:
            n = min(yt.shape[0], mp.shape[0])
            yt = yt[:n]
            mp = mp[:n]
        spec = compose_prediction_stability_figure(mp, y_true=yt, seed=seed)
        out = base_path + "_prediction_stability"
        ok = _save_spec(spec, plot_outputs, out)
        _record(charts, "prediction_stability", ok)
        if ok:
            _record_path(charts, out)
        return bool(ok)
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
        return bool(ok)
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
    """Render the per-target y / prediction distribution overlay once per target. Returns success."""
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if not plot_outputs or not base_path or not y_true_by_split:
        return False
    from mlframe.reporting.charts.error_analysis import target_dist_overlay

    overlay_task = "classification" if task == "classification" else "regression"
    try:
        spec = target_dist_overlay(y_true_by_split, pred_by_split=pred_by_split, task=overlay_task)
        ok = _save_spec(spec, plot_outputs, base_path + "_target_dist")
        _record(charts, "target_dist", ok)
        if ok:
            _record_path(charts, base_path + "_target_dist")
        return bool(ok)
    except Exception:
        logger.exception("diagnostics_dispatch: target_dist_overlay failed; continuing.")
        _record(charts, "target_dist", False)
        return False


def _column_names(*args, **kwargs):
    """Lazy delegate to the parent module's frame-agnostic (pandas/polars) column-name lister."""
    from .diagnostics_dispatch import _column_names as _f
    return _f(*args, **kwargs)


def _ranked_top_features(names: Sequence[str], feature_importances: Optional[Sequence[float]], k: int) -> list:
    """Top-``k`` feature names ranked by importance when available, else the first ``k`` names (mirrors pdp_ice)."""
    if feature_importances is not None and len(feature_importances) == len(names):
        importances = np.asarray(feature_importances, dtype=np.float64)
        # np.argsort sorts NaN LAST (ascending); the prior `[::-1]` reversal then put NaN-importance
        # features FIRST -- picked as top-ranked instead of excluded. Sort ascending by a NaN-safe
        # descending key (-x for finite values, so ascending order of the key is descending order of
        # importance; NaN mapped to +inf so it sinks to the very end regardless).
        sort_key = np.where(np.isnan(importances), np.inf, -importances)
        order = np.argsort(sort_key)
        return [names[int(i)] for i in order][:k]
    return list(names)[:k]


def _first_group_column(df: Any, names: Optional[Sequence[str]], max_card: int = 50) -> Optional[str]:
    """First bounded-cardinality categorical column usable as the class-structure ``group`` axis, else None.

    Prefers pandas ``category`` dtype (cardinality is the cheap ``.cat.categories`` length); falls back to an ``object``
    column whose sampled cardinality is in [2, max_card]. The chart caps to its own ``max_groups`` anyway, so a slightly
    higher-cardinality object column is still acceptable -- this only skips free-text / id-like columns.
    """
    for c in names or []:
        try:
            col = df[c]
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            continue
        dt = getattr(col, "dtype", None)
        dt_str = str(dt)
        # pandas category dtype OR polars Categorical/Enum (str(dtype) is "Categorical(...)"/"Enum(...)").
        if dt_str == "category" or dt_str.startswith("Categorical") or dt_str.startswith("Enum"):
            try:
                if hasattr(col, "cat"):
                    card = len(col.cat.categories)  # pandas
                elif hasattr(col, "n_unique"):
                    card = col.n_unique()  # polars
                else:
                    continue
                if 2 <= card <= max_card:
                    return c
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed: %s", e)
                continue
        # `dt is object` compares the dtype INSTANCE by identity against the Python builtin `object`
        # TYPE, which a pandas object-dtype (numpy.dtype('O')) never satisfies -- this branch was
        # unreachable for object-dtype columns. Use equality (numpy defines dtype == object
        # meaningfully) plus explicit string-dtype coverage for pandas' "string"/polars' "String"/"Utf8"/
        # pandas>=3's PDEP-14 default string dtype, whose str(dtype) is the bare "str" (not "string").
        elif dt == object or dt_str.startswith("string") or dt_str in ("Utf8", "String", "str"):  # noqa: E721 -- `is` genuinely does not work here (numpy.dtype('O') is object is False); that was the bug just fixed above.
            try:
                head = col.head(20_000) if hasattr(col, "head") else col
                if hasattr(head, "nunique"):
                    nun = int(head.nunique(dropna=True))  # pandas
                elif hasattr(head, "n_unique"):
                    nun = int(head.n_unique())  # polars
                else:
                    nun = 0
                if 2 <= nun <= max_card:
                    return c
            except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed: %s", e)
                continue
    return None


def render_engineered_separability_diagnostic(
    *,
    df: Any,
    y_true: Any,
    feature_names: Optional[Sequence[str]],
    feature_importances: Optional[Sequence[float]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    sample: int = 5000,
    seed: int = 0,
) -> bool:
    """2-D scatter of the top-2 importance features colored by target, annotated with the Fisher separability score.

    Default-ON when a feature frame + targets are present. Cost is one bounded seeded row subsample + an O(n) Fisher
    ratio (njit); RAM-safe (only the two feature columns are pulled as views, never a frame copy).
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if df is None or y_true is None or not plot_outputs or not base_path:
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    if not names or len(names) < 2:
        return False
    top2 = _ranked_top_features(names, feature_importances, 2)
    if len(top2) < 2:
        return False
    try:
        from mlframe.reporting.charts.engineered_separability import compose_separability_figure

        spec = compose_separability_figure(df, np.asarray(y_true).ravel(), features=top2, sample=sample, seed=seed)
        ok = _save_spec(spec, plot_outputs, base_path + "_separability")
        _record(charts, "engineered_separability", ok)
        if ok:
            _record_path(charts, base_path + "_separability")
        return bool(ok)
    except Exception:
        logger.exception("diagnostics_dispatch: engineered_separability failed; continuing.")
        _record(charts, "engineered_separability", False)
        return False


def render_category_discriminability_diagnostic(
    *,
    df: Any,
    y_true: Any,
    feature_names: Optional[Sequence[str]],
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    top_k: int = 15,
    min_support: int = 30,
    max_columns: int = 40,
    seed: int = 0,
) -> bool:
    """Per-category-level Weight-of-Evidence bar (case_sdsj discriminability) for a BINARY target.

    Default-ON for binary classification with at least one categorical column; skips cheaply otherwise. RAM-safe: the
    builder pulls one categorical column at a time as codes and bounds the count pass to a 200k row subsample; the
    number of columns scanned is capped at ``max_columns`` (importance order, since ``feature_names`` arrives ranked).
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if df is None or y_true is None or not plot_outputs or not base_path:
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    if names and max_columns and len(names) > max_columns:
        names = names[:max_columns]
    try:
        from mlframe.reporting.charts.category_discriminability import compose_category_discriminability_figure

        spec = compose_category_discriminability_figure(
            df, np.asarray(y_true).ravel(), features=names, top_k=top_k, min_support=min_support, seed=seed,
        )
        if spec is None:
            return False
        ok = _save_spec(spec, plot_outputs, base_path + "_category_discriminability")
        _record(charts, "category_discriminability", ok)
        if ok:
            _record_path(charts, base_path + "_category_discriminability")
        return bool(ok)
    except Exception:
        logger.exception("diagnostics_dispatch: category_discriminability failed; continuing.")
        _record(charts, "category_discriminability", False)
        return False


def render_class_structure_diagnostic(
    *,
    df: Any,
    y_true: Any,
    feature_names: Optional[Sequence[str]],
    timestamps: Optional[Any] = None,
    plot_outputs: str,
    base_path: str,
    metrics_dict: Optional[dict] = None,
    max_groups: int = 30,
    n_time_bins: int = 20,
    seed: int = 0,
) -> bool:
    """Group x time-bin class-rate heatmap (case_visual leakage/structure diagnostic).

    Default-ON when the frame carries a bounded-cardinality categorical column to use as the group axis. Time bins come
    from ``timestamps`` when present, else row order (still exposes group-band structure). RAM-safe: the kernel is a
    single njit O(n) 2-D accumulate over the group + time codes; only the group column is pulled as a view.
    """
    charts = metrics_dict.setdefault("charts", {"saved": [], "failed": []}) if isinstance(metrics_dict, dict) else None
    if df is None or y_true is None or not plot_outputs or not base_path:
        return False
    names = list(feature_names) if feature_names else _column_names(df)
    group = _first_group_column(df, names, max_card=max(2, int(max_groups)) * 4)
    if group is None:
        return False
    try:
        from mlframe.reporting.charts.class_structure_heatmap import compose_class_structure_figure

        spec = compose_class_structure_figure(
            df, np.asarray(y_true).ravel(), group=group, timestamps=timestamps,
            max_groups=max_groups, n_time_bins=n_time_bins,
        )
        ok = _save_spec(spec, plot_outputs, base_path + "_class_structure")
        _record(charts, "class_structure", ok)
        if ok:
            _record_path(charts, base_path + "_class_structure")
        return bool(ok)
    except Exception:
        logger.exception("diagnostics_dispatch: class_structure failed; continuing.")
        _record(charts, "class_structure", False)
        return False


__all__ = [
    # render_* diagnostics that live in the parent ``diagnostics_dispatch`` (split/target-drift/pdp/slice/
    # decision-curve/calibration-drift/target-acf/shap) are intentionally NOT re-exported here; they are not
    # defined in this carved-out module. Only the names actually defined below are listed.
    "render_target_dist_overlay",
    "render_engineered_separability_diagnostic",
    "render_category_discriminability_diagnostic",
    "render_class_structure_diagnostic",
    "render_model_comparison_diagnostic",
    "render_model_comparison_from_suite",
    "render_decile_table_diagnostic",
    "render_model_card_diagnostic",
    "render_prediction_stability_diagnostic",
    "render_split_comparison_from_suite",
    "build_combined_html_report",
    "DIAG_ROW_CAP",  # noqa: F822 -- resolved lazily via module __getattr__ above, not a top-level binding
    "DIAG_MAX_FEATURES",  # noqa: F822 -- resolved lazily via module __getattr__ above, not a top-level binding
]
