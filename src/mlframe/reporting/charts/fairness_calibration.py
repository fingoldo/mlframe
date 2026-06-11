"""Per-subgroup reliability small-multiples + per-group ECE -- a calibration-fairness diagnostic.

Equal accuracy across groups does NOT imply equal calibration: a model can score group A's positives correctly
yet systematically over/under-confident on group B, so a single pooled reliability curve hides the disparity.
This composer slices ``(y_true, y_score)`` by a sensitive/group feature, builds a reliability curve + ECE PER group,
and surfaces the MAX-MIN ECE gap across groups as a calibration-disparity metric with a traffic-light.

Layout: one overlay panel with a per-group reliability curve over the shared diagonal, plus a per-group ECE bar
panel sorted worst-first with the disparity gap annotated. Degenerate groups (single-class / too few rows) are
annotated and skipped (no curve, no ECE bar).

Efficiency: O(n) total. Groups are formed by a single ``np.unique(..., return_inverse=True)`` over the group codes;
top-N-by-support are kept and the rest folded into one "other" bucket via a vectorised code remap; each kept group
is binned by the existing ``fast_calibration_binning`` njit path on its contiguous slice. No per-row python loop.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.charts.calibration import standard_ece
from mlframe.reporting.spec import AnnotationPanelSpec, BarPanelSpec, FigureSpec, LinePanelSpec, PanelSpec

# Below this many finite rows OR with a single class present a group's reliability curve / ECE is meaningless noise;
# annotate the group and skip its curve (mirrors the iter-1 degenerate-input guard style).
_MIN_GROUP_ROWS: int = 30
# ECE-gap traffic-light thresholds: gap below green is "calibrated equally", above red is a clear disparity.
_GAP_GREEN: float = 0.05
_GAP_RED: float = 0.10
# Distinct colours cycled across group curves / bars.
_GROUP_COLORS: Tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
_OTHER_LABEL: str = "other"


def _is_single_class(y_true: np.ndarray) -> bool:
    """True iff the binary labels are all-0 or all-1 -- a reliability curve needs both classes. O(n), no sort/unique."""
    s = float(y_true.sum())
    return s == 0.0 or s == float(y_true.size)


def _reliability_points(y_true: np.ndarray, y_score: np.ndarray, n_bins: int):
    """Per-bin (mean-pred, observed-freq, population) + standard ECE for one group, via the shared njit binning.

    Returns ``(freqs_predicted, freqs_true, ece)`` or ``None`` when the group is degenerate (single class /
    all-equal scores / no populated bin). Reuses ``fast_calibration_binning`` so the binning matches the suite's
    reliability diagram exactly.
    """
    from mlframe.metrics.calibration._calibration_plot import fast_calibration_binning

    fp, ft, hits = fast_calibration_binning(y_true, y_score, nbins=n_bins)
    if fp.size == 0:
        return None
    ece = standard_ece(fp, ft, hits)
    if not np.isfinite(ece):
        return None
    return fp, ft, ece


def _gap_traffic_light(gap: float) -> str:
    """green/amber/red verdict on the MAX-MIN ECE gap across groups."""
    if not np.isfinite(gap):
        return "n/a"
    if gap < _GAP_GREEN:
        return "green"
    if gap < _GAP_RED:
        return "amber"
    return "red"


def _prepare_group_codes(subgroups: np.ndarray, max_groups: int):
    """Map raw group labels to top-N-by-support codes + one folded 'other' bucket. Returns (codes, labels, supports).

    ``codes`` is an int array parallel to ``subgroups`` indexing into ``labels``; the rare tail (beyond the top
    ``max_groups``) is remapped to a single trailing 'other' code. Vectorised: one unique + one boolean remap, no loop.
    """
    raw = np.asarray(subgroups).ravel()
    uniq, inv, counts = np.unique(raw, return_inverse=True, return_counts=True)
    order = np.argsort(counts)[::-1]
    if uniq.size <= max_groups:
        labels = [str(uniq[i]) for i in order]
        remap = np.empty(uniq.size, dtype=np.int64)
        remap[order] = np.arange(uniq.size)
        return remap[inv], labels, [int(counts[i]) for i in order]

    keep = order[:max_groups]
    labels = [str(uniq[i]) for i in keep] + [_OTHER_LABEL]
    other_code = max_groups
    remap = np.full(uniq.size, other_code, dtype=np.int64)
    remap[keep] = np.arange(max_groups)
    codes = remap[inv]
    supports = [int(counts[i]) for i in keep] + [int(counts[order[max_groups:]].sum())]
    return codes, labels, supports


def compose_fairness_calibration_figure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    subgroups: np.ndarray | Sequence,
    *,
    max_groups: int = 6,
    n_bins: int = 10,
    title: str = "Calibration fairness by subgroup",
    figsize: Optional[Tuple[float, float]] = None,
) -> FigureSpec:
    """Per-subgroup reliability overlay + per-group ECE bar, with the MAX-MIN ECE gap as a fairness-disparity metric.

    For each value of ``subgroups`` (a sensitive/group feature, capped to the top ``max_groups`` by support; the rest
    folded into one 'other' bucket) a reliability curve and a standard ECE are computed over that group's rows. The
    overlay panel shows every group's curve over the shared perfect-calibration diagonal; the bar panel ranks the
    per-group ECE worst-first and annotates the MAX-MIN gap + a traffic-light (gap < {green} green, < {red} amber, else
    red). A large gap means the model is calibrated UNEQUALLY across groups -- a recognised fairness failure that
    equal-accuracy diagnostics miss. Degenerate groups (single-class / fewer than the row floor) are listed in the
    title and excluded from both panels.

    O(n): groups via one ``np.unique`` + a vectorised code remap, each kept group binned by the shared njit path.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    grp = np.asarray(subgroups).ravel()
    n = min(yt.size, ys.size, grp.size)
    yt, ys, grp = yt[:n], ys[:n], grp[:n]

    finite = np.isfinite(yt) & np.isfinite(ys)
    yt, ys, grp = yt[finite], ys[finite], grp[finite]
    if yt.size == 0:
        return FigureSpec(
            suptitle="",
            panels=((AnnotationPanelSpec(text="calibration fairness unavailable: no finite (label, score) rows", title=title),),),
            figsize=figsize or (8.0, 5.0),
        )

    codes, labels, supports = _prepare_group_codes(grp, max_groups)

    centers = (np.arange(n_bins) + 0.5) / n_bins  # shared bin-centre grid for the overlay diagonal series
    curve_x: list[np.ndarray] = [centers]
    curve_y: list[np.ndarray] = [centers]
    curve_labels: list[str] = ["perfect"]
    curve_styles: list[str] = [":"]
    curve_colors: list[str] = ["#888888"]

    bar_labels: list[str] = []
    bar_eces: list[float] = []
    bar_colors: list[str] = []
    skipped: list[str] = []

    for gi, label in enumerate(labels):
        mask = codes == gi
        gn = int(mask.sum())
        gy, gs = yt[mask], ys[mask]
        if gn < _MIN_GROUP_ROWS or _is_single_class(gy):
            skipped.append(f"{label} (n={gn})")
            continue
        pts = _reliability_points(gy, gs, n_bins)
        if pts is None:
            skipped.append(f"{label} (degenerate)")
            continue
        fp, ft, ece = pts
        color = _GROUP_COLORS[gi % len(_GROUP_COLORS)]
        curve_x.append(fp)
        curve_y.append(ft)
        curve_labels.append(f"{label} (n={gn:,})")
        curve_styles.append("lines+markers")
        curve_colors.append(color)
        bar_labels.append(label)
        bar_eces.append(float(ece))
        bar_colors.append(color)

    skipped_note = ("  skipped: " + ", ".join(skipped)) if skipped else ""

    if len(bar_eces) < 2:
        # Fewer than two groups carry a usable curve: a disparity gap is undefined. Emit an honest annotation rather
        # than a one-bar "disparity" chart, but still show the single curve if present.
        text = "calibration-fairness disparity needs >=2 non-degenerate groups." + (skipped_note or "")
        if len(bar_eces) == 1:
            text = f"only one non-degenerate group ({bar_labels[0]}, ECE={bar_eces[0]:.3f}); disparity undefined." + skipped_note
        panels: Tuple[Tuple[PanelSpec, ...], ...]
        if len(curve_labels) > 1:
            overlay = LinePanelSpec(
                x=tuple(curve_x), y=tuple(curve_y), series_labels=tuple(curve_labels),
                line_styles=tuple(curve_styles), colors=tuple(curve_colors),
                title=title + skipped_note, xlabel="predicted probability", ylabel="observed frequency",
            )
            panels = ((overlay,), (AnnotationPanelSpec(text=text, title="disparity"),))
        else:
            panels = ((AnnotationPanelSpec(text=text, title=title),),)
        return FigureSpec(suptitle="", panels=panels, figsize=figsize or (8.0, 8.0))

    eces = np.asarray(bar_eces, dtype=np.float64)
    gap = float(eces.max() - eces.min())
    light = _gap_traffic_light(gap)

    sort_idx = np.argsort(eces)[::-1]  # worst-first
    bar_cats = tuple(bar_labels[i] for i in sort_idx)
    bar_vals = eces[sort_idx]
    bar_cols = tuple(bar_colors[i] for i in sort_idx)

    overlay = LinePanelSpec(
        x=tuple(curve_x),
        y=tuple(curve_y),
        series_labels=tuple(curve_labels),
        line_styles=tuple(curve_styles),
        colors=tuple(curve_colors),
        title=f"{title}{skipped_note}",
        xlabel="predicted probability",
        ylabel="observed frequency",
    )
    bar = BarPanelSpec(
        categories=bar_cats,
        values=bar_vals,
        colors=bar_cols,
        title=f"per-subgroup ECE  |  disparity gap (max-min)={gap:.3f}  [{light}]",
        xlabel="subgroup",
        ylabel="ECE (lower = better calibrated)",
        orientation="horizontal",
        hline=(float(eces.min()), "#2ca02c", "best-group ECE"),
    )
    return FigureSpec(
        suptitle="",
        panels=((overlay,), (bar,)),
        figsize=figsize or (8.0, 9.0),
        row_height_ratios=(3.0, 2.0),
    )


def compute_subgroup_ece_disparity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    subgroups: np.ndarray | Sequence,
    *,
    max_groups: int = 6,
    n_bins: int = 10,
) -> Mapping[str, object]:
    """Per-group ECE dict + the MAX-MIN disparity gap + traffic-light, without building a figure.

    Returns ``{"per_group_ece": {label: ece}, "gap": float, "traffic_light": str, "skipped": [...]}``. The figure
    composer is the visual surface; this is the headless metric the biz_value test + the suite metrics dict consume.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    grp = np.asarray(subgroups).ravel()
    n = min(yt.size, ys.size, grp.size)
    yt, ys, grp = yt[:n], ys[:n], grp[:n]
    finite = np.isfinite(yt) & np.isfinite(ys)
    yt, ys, grp = yt[finite], ys[finite], grp[finite]

    per_group: dict[str, float] = {}
    skipped: list[str] = []
    if yt.size == 0:
        return {"per_group_ece": per_group, "gap": float("nan"), "traffic_light": "n/a", "skipped": ["no finite rows"]}

    codes, labels, _ = _prepare_group_codes(grp, max_groups)
    for gi, label in enumerate(labels):
        mask = codes == gi
        gn = int(mask.sum())
        gy, gs = yt[mask], ys[mask]
        if gn < _MIN_GROUP_ROWS or _is_single_class(gy):
            skipped.append(label)
            continue
        pts = _reliability_points(gy, gs, n_bins)
        if pts is None:
            skipped.append(label)
            continue
        per_group[label] = float(pts[2])

    if len(per_group) < 2:
        return {"per_group_ece": per_group, "gap": float("nan"), "traffic_light": "n/a", "skipped": skipped}
    vals = np.asarray(list(per_group.values()), dtype=np.float64)
    gap = float(vals.max() - vals.min())
    return {"per_group_ece": per_group, "gap": gap, "traffic_light": _gap_traffic_light(gap), "skipped": skipped}


__all__ = [
    "compose_fairness_calibration_figure",
    "compute_subgroup_ece_disparity",
]
