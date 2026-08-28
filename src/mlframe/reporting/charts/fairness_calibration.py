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

from mlframe.reporting.charts._calibration_chart_shared import is_single_class, null_ece_scale, reliability_points
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


_is_single_class = is_single_class
_reliability_points = reliability_points


# Re-exported: the ECE noise floor lives in the shared calibration module because every ECE consumer in this
# package needs it. Here it grades the GAP between two groups -- a perfectly calibrated model with identical
# mechanisms in both, 200000 rows in one and 30 in the other, produces ECEs of 0.002 and 0.193, and the fixed
# 0.10 bar called that 0.19 gap a red fairness failure when the only difference was sample size.
_null_ece_scale = null_ece_scale


def _gap_traffic_light(gap: float, noise_floor: float = 0.0) -> str:
    """green/amber/red verdict on the MAX-MIN ECE gap, graded against the larger of a fixed bar and this data's noise.

    ``noise_floor`` is the gap the SMALLEST compared group would show even under perfect calibration; a gap inside
    it carries no information about fairness, only about how many rows that group has.
    """
    if not np.isfinite(gap):
        return "n/a"
    if gap < max(_GAP_GREEN, noise_floor):
        return "green"
    if gap < max(_GAP_RED, 2.0 * noise_floor):
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
    per-group ECE worst-first and annotates the MAX-MIN gap + a traffic-light. The light grades the gap against the
    LARGER of a fixed bar (0.05 green / 0.10 amber) and this data's own noise floor, so a small group's sampling
    error cannot masquerade as a fairness failure -- see :func:`null_ece_scale`. A large gap means the model is
    calibrated UNEQUALLY across groups -- a recognised fairness failure that equal-accuracy diagnostics miss. Degenerate groups (single-class / fewer than the row floor) are listed in the
    title and excluded from both panels.

    O(n): groups via one ``np.unique`` + a vectorised code remap, each kept group binned by the shared njit path.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    grp = np.asarray(subgroups).ravel()
    n = yt.size
    if not (ys.size == n == grp.size):
        raise ValueError(f"fairness_calibration: y_true ({n}), y_score ({ys.size}), subgroups ({grp.size}) must have equal length")

    finite = np.isfinite(yt) & np.isfinite(ys)
    yt, ys, grp = yt[finite], ys[finite], grp[finite]
    if yt.size == 0:
        return FigureSpec(
            suptitle="",
            panels=((AnnotationPanelSpec(text="calibration fairness unavailable: no finite (label, score) rows", title=title),),),
            figsize=figsize or (8.0, 5.0),
        )

    codes, labels, _supports = _prepare_group_codes(grp, max_groups)

    centers = (np.arange(n_bins) + 0.5) / n_bins  # shared bin-centre grid for the overlay diagonal series
    curve_x: list[np.ndarray] = [centers]
    curve_y: list[np.ndarray] = [centers]
    curve_labels: list[str] = ["perfect"]
    curve_styles: list[str] = [":"]
    curve_colors: list[str] = ["#888888"]

    bar_labels: list[str] = []
    bar_eces: list[float] = []
    bar_colors: list[str] = []
    bar_ns: list[int] = []
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
        bar_ns.append(gn)

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
    # The smallest compared group sets the floor: its own sampling noise is the largest a gap can be while still
    # saying nothing about fairness. Reported on the panel so a real disparity is distinguishable from a small group.
    prevalence = float(yt.mean()) if yt.size else 0.0
    noise_floor = max(null_ece_scale(gn_i, prevalence, n_bins) for gn_i in bar_ns)
    light = _gap_traffic_light(gap, noise_floor)

    sort_idx = np.argsort(eces)[::-1]  # worst-first
    bar_cats = tuple(f"{bar_labels[i]} (n={bar_ns[i]:,})" for i in sort_idx)
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
        title=(f"per-subgroup ECE  |  disparity gap (max-min)={gap:.3f}  [{light}]" f"  |  noise floor at the smallest group={noise_floor:.3f}"),
        # For a horizontal bar the VALUE axis is x and the CATEGORY axis is y, matching every other builder.
        xlabel="ECE (lower = better calibrated)",
        ylabel="subgroup",
        orientation="horizontal",
        hline=(float(eces.min()), "#2ca02c", "best-group ECE"),
    )
    return FigureSpec(
        suptitle="",
        panels=((overlay,), (bar,)),
        figsize=figsize or (8.0, 9.0),
        row_height_ratios=(3.0, 2.0),
        caption=(
            "Top: one reliability curve per subgroup against the perfect-calibration diagonal -- x is the predicted "
            "probability in a bin, y the frequency actually observed in it. Bottom: each group's ECE (its mean gap "
            "to the diagonal, lower is better), worst-first. The headline is the GAP between groups, not the level: "
            "a model can rank equally well everywhere yet be systematically over-confident about one group. ECE is "
            f"noisy at small group sizes, so the gap is graded against a noise floor of {noise_floor:.3f} computed "
            "from the smallest group here -- a gap inside that floor says nothing about fairness."
        ),
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
    n = yt.size
    if not (ys.size == n == grp.size):
        raise ValueError(f"fairness_calibration: y_true ({n}), y_score ({ys.size}), subgroups ({grp.size}) must have equal length")
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
    "null_ece_scale",
]
