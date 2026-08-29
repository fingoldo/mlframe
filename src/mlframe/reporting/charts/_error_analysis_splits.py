"""Per-split target / prediction distribution overlays, carved out of ``error_analysis``.

``target_dist_overlay`` and its helpers answer one question: are the splits exchangeable? Overlaid per-split
densities of the target and of the predictions show whether a holdout metric can be expected to transfer, and
the drift verdict states it in words.

Split from ``error_analysis.py`` to keep that module under the house 1000-LOC limit. This is its most
self-contained group -- it shares only the small array helpers imported below, and no other builder calls into
it. ``error_analysis`` re-exports every public name here, so existing import sites are unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid, pack_panels
from mlframe.reporting.spec import BarPanelSpec, FigureSpec, LinePanelSpec, PanelSpec

# Imported from the parent rather than redefined: one source of truth for the shared array coercion, the
# overlay bin count and the drift z-quantile. A second copy would drift silently.
from ._error_analysis_shared import DEFAULT_OVERLAY_BINS, _DRIFT_Z, _as_float_1d

logger = logging.getLogger(__name__)


def _split_arrays(
    values: np.ndarray,
    split_labels: Sequence[Any],
) -> Dict[str, np.ndarray]:
    """Group a flat value array by its per-row split label, preserving label order of first appearance."""
    vals = np.asarray(values)
    labels = np.asarray(split_labels)
    out: Dict[str, np.ndarray] = {}
    for lab in dict.fromkeys(labels.tolist()):
        out[str(lab)] = vals[labels == lab]
    return out


def _common_edges(groups: Dict[str, np.ndarray], nbins: int) -> Optional[np.ndarray]:
    """Shared histogram edges across all split groups so overlaid densities are comparable. None when no finite data."""
    finite_min = np.inf
    finite_max = -np.inf
    for arr in groups.values():
        a = _as_float_1d(arr)
        a = a[np.isfinite(a)]
        if a.size:
            finite_min = min(finite_min, float(a.min()))
            finite_max = max(finite_max, float(a.max()))
    if not np.isfinite(finite_min) or finite_max <= finite_min:
        if np.isfinite(finite_min):
            return np.linspace(finite_min, finite_min + 1.0, nbins + 1)
        return None
    return np.linspace(finite_min, finite_max, nbins + 1)


def _density_overlay_panel(
    groups: Dict[str, np.ndarray],
    *,
    nbins: int,
    title: str,
    xlabel: str,
    train_key: Optional[str],
) -> PanelSpec:
    """Overlaid per-split density histograms on shared edges, with p01/p99 vlines and train-envelope shading.

    All binning is ``np.histogram`` on shared edges (O(n) per split); the curve is the bin centres so it stays at
    ``nbins`` vertices regardless of row count. cProfile at 2.9M total rows: ~120 ms, all in ``np.histogram`` bin
    search + the single train ``np.percentile`` partition -- no actionable speedup, this is the aggregate floor.
    """
    # Imported function-locally (not at module top) so this chart submodule does not pull in the ``mlframe.reporting``
    # package surface at load time -- that back-edge closes the whole reporting.charts facade into one import SCC.
    from mlframe.reporting import colors as _colors

    edges = _common_edges(groups, nbins)
    if edges is None:
        from mlframe.reporting.spec import AnnotationPanelSpec
        return AnnotationPanelSpec(text=f"{title}\n(no finite data)", title=title)
    centers = (edges[:-1] + edges[1:]) / 2.0
    series: List[np.ndarray] = []
    labels: List[str] = []
    cols: List[str] = []
    for i, (lab, arr) in enumerate(groups.items()):
        a = _as_float_1d(arr)
        a = a[np.isfinite(a)]
        dens, _ = np.histogram(a, bins=edges, density=True) if a.size else (np.zeros(len(centers)), edges)
        series.append(dens)
        labels.append(f"{lab} (mean={a.mean():.3g})" if a.size else f"{lab} (empty)")
        cols.append(_colors.line_color(i))

    vlines = None
    vspans = None
    if train_key is not None and train_key in groups:
        tr = _as_float_1d(groups[train_key])
        tr = tr[np.isfinite(tr)]
        if tr.size:
            p01, p99 = float(np.percentile(tr, 1)), float(np.percentile(tr, 99))
            vlines = ((p01, "gray", "train p01"), (p99, "gray", "train p99"))
            vspans = ((p01, p99, "gray", 0.08),)
    return LinePanelSpec(
        x=centers,
        y=tuple(series),
        series_labels=tuple(labels),
        colors=tuple(cols),
        title=title,
        xlabel=xlabel,
        ylabel="Density",
        vlines=vlines,
        vspans=vspans,
    )


def _classrate_panel(
    groups: Dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
) -> PanelSpec:
    """Per-split class-rate grouped bars: one bar group per class, one bar per split. Aggregate via bincount."""
    classes = np.unique(np.concatenate([np.asarray(a).ravel() for a in groups.values() if len(a)])) if any(len(a) for a in groups.values()) else np.array([0])
    class_index = {c: i for i, c in enumerate(classes)}
    rate_series: List[np.ndarray] = []
    split_labels: List[str] = []
    for lab, arr in groups.items():
        a = np.asarray(arr).ravel()
        rates = np.zeros(len(classes), dtype=np.float64)
        if a.size:
            for c, cnt in zip(*np.unique(a, return_counts=True)):
                rates[class_index[c]] = cnt / a.size
        rate_series.append(rates)
        split_labels.append(str(lab))
    return BarPanelSpec(
        categories=tuple(f"class {c:g}" if np.issubdtype(type(c), np.number) else str(c) for c in classes),
        values=tuple(rate_series),
        series_labels=tuple(split_labels),
        title=title,
        xlabel=xlabel,
        ylabel="Class rate",
    )


def _target_drift_verdict(
    y_true_by_split: Dict[str, np.ndarray],
    *,
    train_key: str,
    task: str,
) -> str:
    """One-line distribution-shift verdict: how far each non-train split's target mean drifts from train.

    The bar is per-split and scales with sample size. For CLASSIFICATION the quantity is a class-1 rate, and the old
    fixed ``0.25 * train_std`` bar hardcoded ``train_std = 1.0``, so a rate had to move 0.25 ABSOLUTE to be flagged:
    a base rate going 0.5% -> 2.0% (four times as many positives, a serious shift) reported "No material drift from
    train". A rate difference is now compared against its two-proportion standard error. For REGRESSION the shift must
    clear both the two-sample standard error of the mean difference (so a 30-row split stops firing on noise) and the
    existing quarter-of-a-standard-deviation effect-size bar (so a huge n stops flagging a shift nobody would act on).
    Returns a reader-facing sentence appended to the figure title so a train/val/test target shift is called out, not
    just drawn.
    """
    if train_key not in y_true_by_split:
        return "Distribution-shift check: no 'train' split provided -> cannot compare drift."
    tr = _as_float_1d(y_true_by_split[train_key])
    tr = tr[np.isfinite(tr)]
    if tr.size == 0:
        return "Distribution-shift check: train split empty -> cannot compare drift."
    tr_mean = float(tr.mean())
    is_clf = task == "classification"
    tr_var = float(tr.var())
    tr_std = float(np.sqrt(tr_var))
    effect_thr = 0.25 * tr_std  # regression-only "materiality" floor, unchanged
    parts: List[str] = []
    flagged: List[str] = []
    excluded: List[str] = []
    for lab, arr in y_true_by_split.items():
        if lab == train_key:
            continue
        a = _as_float_1d(arr)
        a = a[np.isfinite(a)]
        if a.size == 0:
            excluded.append(lab)
            continue
        shift = float(a.mean()) - tr_mean
        parts.append(f"{lab} {shift:+.3g} (n={a.size:,})")
        if is_clf:
            # Two-proportion SE under the pooled rate; a 30-row split needs a far bigger move than a 200k-row one.
            pooled = (tr.sum() + a.sum()) / float(tr.size + a.size)
            se = float(np.sqrt(max(pooled * (1.0 - pooled), 0.0) * (1.0 / tr.size + 1.0 / a.size)))
            if se > 0 and abs(shift) > _DRIFT_Z * se:
                flagged.append(lab)
        else:
            se = float(np.sqrt(tr_var / tr.size + float(a.var()) / a.size))
            if abs(shift) > max(_DRIFT_Z * se, effect_thr) > 0:
                flagged.append(lab)
    excluded_note = f" (split(s) empty/all-NaN, excluded from drift check: {', '.join(excluded)})" if excluded else ""
    if not parts:
        if excluded:
            return f"Distribution-shift check: no usable non-train split{excluded_note} -> cannot compare drift."
        return "Distribution-shift check: only the train split present -> no drift to assess."
    scale = (
        " (class-1 rate shift vs train; flagged when it clears 1.96 two-proportion SE)"
        if is_clf
        else " (vs train, in target units; flagged when it clears both 1.96 SE and 0.25*train_std)"
    )
    head = f"Mean shift{scale}: " + ", ".join(parts) + f"; train mean={tr_mean:.3g}.{excluded_note}"
    if flagged:
        return head + f" WARNING: {', '.join(flagged)} drift materially from train -> distribution-shift risk; holdout metrics may not transfer."
    return head + " No material drift from train."


def target_dist_overlay(
    y_true_by_split: Dict[str, np.ndarray],
    *,
    pred_by_split: Optional[Dict[str, np.ndarray]] = None,
    task: str = "regression",
    nbins: int = DEFAULT_OVERLAY_BINS,
    train_key: str = "train",
    title: str = "Target & prediction distribution by split",
) -> FigureSpec:
    """Overlaid per-split distributions of y AND of predictions.

    ``y_true_by_split`` / ``pred_by_split`` map a split name ("train"/"val"/"test"/"oof") to its value array. For
    regression each panel overlays per-split density histograms with the train p01/p99 vlines + a shaded train
    envelope, so a train/test target shift is visible as separated curves. For classification each panel shows
    per-split class-rate grouped bars. The prediction panel naturally carries the OOF-vs-test prediction overlay
    when both keys are present. All binning is ``np.histogram`` / ``bincount`` (O(n)); curves stay at ``nbins``
    vertices regardless of row count.
    """
    panels: List[PanelSpec] = []
    drift_line = _target_drift_verdict(y_true_by_split, train_key=train_key, task=task)
    if task == "classification":
        panels.append(_classrate_panel(y_true_by_split, title="Target class rate by split", xlabel="class"))
        if pred_by_split:
            panels.append(_classrate_panel(pred_by_split, title="Prediction class rate by split", xlabel="class"))
    else:
        panels.append(_density_overlay_panel(
            y_true_by_split, nbins=nbins, title="Target (y) distribution by split",
            xlabel="y", train_key=train_key,
        ))
        if pred_by_split:
            panels.append(_density_overlay_panel(
                pred_by_split, nbins=nbins, title="Prediction distribution by split (incl. OOF vs test)",
                xlabel="prediction", train_key=train_key if train_key in pred_by_split else None,
            ))
    grid = pack_panels(panels, max_cols=2)
    return FigureSpec(
        suptitle=title,
        panels=grid,
        figsize=figsize_for_grid(1, max(len(panels), 1), cell_width=7.0, cell_height=4.5),
        caption=(
            "Overlaid per-split distributions of the target and of the predictions. Curves that separate mean the "
            "splits are not exchangeable, so a holdout metric may not transfer to the next period. The grey band is "
            "the train p01-p99 envelope: prediction mass outside it is extrapolation, where the model has never "
            f"seen a comparable example. VERDICT: {drift_line}"
        ),
    )


__all__ = [
    "target_dist_overlay",
]
