"""Per-feature calibration: reliability + ECE conditioned on quantile bins of a continuous feature.

A single pooled reliability curve can look fine while the model is calibrated for low feature values yet
over-confident for high ones -- the miscalibration cancels in the pooled average. This composer slices
``(y_true, y_score)`` by QUANTILE bins of a continuous feature, builds a reliability curve + ECE per
feature-bin, and surfaces the MAX-MIN ECE across feature-bins as a "calibration heterogeneity" metric.
It complements the categorical subgroup-calibration fairness chart by conditioning on a CONTINUOUS feature.

Layout: a small-multiples row of mini reliability curves (one per feature-bin, shared axes) over the shared
diagonal, plus an ECE-vs-feature-bin line showing whether calibration degrades across the feature's range.
Degenerate bins (single-class / too few rows / no populated prob-bin) are annotated and skipped. NaN feature
values are dropped.

Efficiency: O(n) total. Feature-bin assignment is a single ``np.quantile`` + ``np.searchsorted``; each kept bin
is binned by the shared ``fast_calibration_binning`` njit path on its contiguous slice. Huge bins are subsampled
to a cap before the njit pass so the cost stays bounded at n>=1e6 without changing the curve materially.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.charts.calibration import standard_ece
from mlframe.reporting.spec import AnnotationPanelSpec, FigureSpec, LinePanelSpec, PanelSpec

# Below this many finite rows OR with a single class present a feature-bin's reliability curve / ECE is meaningless
# noise; annotate the bin and skip its curve (mirrors the fairness-calibration degenerate-input guard style).
_MIN_BIN_ROWS: int = 30
# Heterogeneity-gap traffic-light thresholds: below green is "calibrated uniformly across the feature", above red
# is a clear miscalibration that varies with the feature value.
_HET_GREEN: float = 0.05
_HET_RED: float = 0.10
# Subsample cap per feature-bin before the njit reliability pass: the curve is read on n_prob_bins points, so a
# uniform subsample of this size is visually identical while bounding the binning cost at n>=1e6.
_BIN_SUBSAMPLE_CAP: int = 200_000
_BIN_COLORS: Tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def _is_single_class(y_true: np.ndarray) -> bool:
    """True iff the binary labels are all-0 or all-1 -- a reliability curve needs both classes. O(n)."""
    s = float(y_true.sum())
    return s == 0.0 or s == float(y_true.size)


def _reliability_points(y_true: np.ndarray, y_score: np.ndarray, n_bins: int):
    """Per-bin (mean-pred, observed-freq) + standard ECE for one feature-bin, via the shared njit binning.

    Returns ``(freqs_predicted, freqs_true, ece)`` or ``None`` when the bin is degenerate (single class /
    all-equal scores / no populated bin). Reuses ``fast_calibration_binning`` so binning matches the suite's
    reliability diagram exactly.
    """
    from mlframe.metrics.calibration import fast_calibration_binning

    fp, ft, hits = fast_calibration_binning(y_true, y_score, nbins=n_bins)
    if fp.size == 0:
        return None
    ece = standard_ece(fp, ft, hits)
    if not np.isfinite(ece):
        return None
    return fp, ft, ece


def _het_traffic_light(gap: float) -> str:
    """green/amber/red verdict on the MAX-MIN ECE heterogeneity across feature-bins."""
    if not np.isfinite(gap):
        return "n/a"
    if gap < _HET_GREEN:
        return "green"
    if gap < _HET_RED:
        return "amber"
    return "red"


def _quantile_bin_codes(feature: np.ndarray, n_feature_bins: int):
    """Assign each row to a quantile bin of ``feature``. Returns ``(codes, edges, labels)``.

    Quantile edges collapse on a near-constant feature (ties), yielding fewer distinct bins than requested; the
    caller treats an under-populated / empty code as a degenerate bin. ``codes`` is parallel to ``feature``.
    """
    qs = np.linspace(0.0, 1.0, n_feature_bins + 1)
    edges = np.quantile(feature, qs)
    edges = np.unique(edges)  # collapse tied quantiles on a low-cardinality / constant feature
    if edges.size < 2:
        return None, edges, []
    inner = edges[1:-1]
    codes = np.searchsorted(inner, feature, side="right") if inner.size else np.zeros(feature.size, dtype=np.int64)
    codes = np.asarray(codes, dtype=np.int64)
    n_bins = edges.size - 1
    labels = [f"[{edges[i]:.3g}, {edges[i + 1]:.3g}{')' if i < n_bins - 1 else ']'}" for i in range(n_bins)]
    return codes, edges, labels


def _per_bin_ece(
    y_true: np.ndarray,
    y_score: np.ndarray,
    feature_values: np.ndarray,
    *,
    n_feature_bins: int,
    n_prob_bins: int,
    random_state: int,
):
    """Core: drop non-finite rows, quantile-bin by feature, compute a reliability curve + ECE per kept bin.

    Returns ``(records, skipped)`` where ``records`` is a list of dicts
    ``{label, center, n, fp, ft, ece, color}`` (one per non-degenerate bin, in feature order) and ``skipped``
    a list of human-readable degenerate-bin notes. O(n): one quantile + searchsorted, each kept bin binned by
    the shared njit path on its contiguous slice (subsampled above the cap).
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    fv = np.asarray(feature_values, dtype=np.float64).ravel()
    n = min(yt.size, ys.size, fv.size)
    yt, ys, fv = yt[:n], ys[:n], fv[:n]

    finite = np.isfinite(yt) & np.isfinite(ys) & np.isfinite(fv)
    yt, ys, fv = yt[finite], ys[finite], fv[finite]
    if yt.size == 0:
        return [], ["no finite (label, score, feature) rows"]

    codes, _edges, labels = _quantile_bin_codes(fv, n_feature_bins)
    if codes is None:
        return [], ["constant feature: a single quantile bin spans the whole range"]

    rng = np.random.default_rng(random_state)
    records = []
    skipped = []
    for bi, label in enumerate(labels):
        mask = codes == bi
        bn = int(mask.sum())
        by, bs, bf = yt[mask], ys[mask], fv[mask]
        center = float(np.median(bf)) if bn else float("nan")
        if bn < _MIN_BIN_ROWS or _is_single_class(by):
            skipped.append(f"{label} (n={bn})")
            continue
        if bn > _BIN_SUBSAMPLE_CAP:
            sel = rng.choice(bn, size=_BIN_SUBSAMPLE_CAP, replace=False)
            by, bs = by[sel], bs[sel]
        pts = _reliability_points(by, bs, n_prob_bins)
        if pts is None:
            skipped.append(f"{label} (degenerate)")
            continue
        fp, ft, ece = pts
        records.append({
            "label": label, "center": center, "n": bn,
            "fp": fp, "ft": ft, "ece": float(ece), "color": _BIN_COLORS[bi % len(_BIN_COLORS)],
        })
    return records, skipped


def compute_calibration_by_feature_heterogeneity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    feature_values: np.ndarray,
    *,
    n_feature_bins: int = 4,
    n_prob_bins: int = 10,
    random_state: int = 0,
) -> Mapping[str, object]:
    """Per-feature-bin ECE dict + MAX-MIN heterogeneity gap + traffic-light, without building a figure.

    Returns ``{"per_bin_ece": {label: ece}, "bin_centers": {label: center}, "heterogeneity": float,
    "traffic_light": str, "skipped": [...]}``. The figure composer is the visual surface; this is the headless
    metric the biz_value test + the suite metrics dict consume. ``heterogeneity`` is NaN with <2 usable bins.
    """
    records, skipped = _per_bin_ece(
        y_true, y_score, np.asarray(feature_values),
        n_feature_bins=n_feature_bins, n_prob_bins=n_prob_bins, random_state=random_state,
    )
    per_bin = {r["label"]: r["ece"] for r in records}
    centers = {r["label"]: r["center"] for r in records}
    if len(records) < 2:
        return {"per_bin_ece": per_bin, "bin_centers": centers, "heterogeneity": float("nan"), "traffic_light": "n/a", "skipped": skipped}
    eces = np.asarray([r["ece"] for r in records], dtype=np.float64)
    gap = float(eces.max() - eces.min())
    return {"per_bin_ece": per_bin, "bin_centers": centers, "heterogeneity": gap, "traffic_light": _het_traffic_light(gap), "skipped": skipped}


def compose_calibration_by_feature_figure(
    y_true: np.ndarray,
    y_score: np.ndarray,
    feature_values: np.ndarray | Sequence,
    *,
    feature_name: str = "feature",
    n_feature_bins: int = 4,
    n_prob_bins: int = 10,
    random_state: int = 0,
    figsize: Optional[Tuple[float, float]] = None,
) -> FigureSpec:
    """Small-multiples reliability per feature-bin + an ECE-vs-feature-bin line, with the calibration-heterogeneity metric.

    Rows are quantile-binned on ``feature_values`` (a continuous feature). For each bin a reliability curve + standard
    ECE are computed over that bin's rows; the top row shows one mini reliability curve per feature-bin over the shared
    perfect-calibration diagonal (shared axes), and the bottom panel plots per-bin ECE against the feature-bin so the
    operator sees whether calibration DEGRADES across the feature's range. The MAX-MIN ECE across bins is the
    "calibration heterogeneity" metric (annotated with a traffic-light: < {green} green, < {red} amber, else red). A
    large value means the model is calibrated UNEVENLY across the feature -- a failure a single pooled reliability curve
    hides. Degenerate bins (single-class / too few rows / constant feature) are listed in the title and excluded.

    O(n): one ``np.quantile`` + ``np.searchsorted`` for bin assignment, each kept bin binned by the shared njit path.
    """
    title = f"Calibration by {feature_name}"
    records, skipped = _per_bin_ece(
        y_true, y_score, np.asarray(feature_values),
        n_feature_bins=n_feature_bins, n_prob_bins=n_prob_bins, random_state=random_state,
    )
    skipped_note = ("  skipped: " + ", ".join(skipped)) if skipped else ""

    if len(records) < 2:
        text = f"per-feature calibration needs >=2 non-degenerate feature-bins (got {len(records)})." + (skipped_note or "")
        return FigureSpec(
            suptitle="",
            panels=((AnnotationPanelSpec(text=text, title=title),),),
            figsize=figsize or (9.0, 4.0),
        )

    centers = (np.arange(n_prob_bins) + 0.5) / n_prob_bins  # shared diagonal grid for every mini panel
    mini_panels: list[PanelSpec] = []
    for r in records:
        mini_panels.append(LinePanelSpec(
            x=(centers, r["fp"]),
            y=(centers, r["ft"]),
            series_labels=("perfect", "observed"),
            line_styles=(":", "lines+markers"),
            colors=("#888888", r["color"]),
            title=f"{r['label']}\nn={r['n']:,}  ECE={r['ece']:.3f}",
            xlabel="predicted probability",
            ylabel="observed frequency",
        ))

    eces = np.asarray([r["ece"] for r in records], dtype=np.float64)
    gap = float(eces.max() - eces.min())
    light = _het_traffic_light(gap)
    bin_index = np.arange(len(records), dtype=np.float64)
    ece_line = LinePanelSpec(
        x=bin_index,
        y=eces,
        series_labels=(f"ECE by {feature_name}-bin",),
        line_styles=("lines+markers",),
        colors=("#d62728",),
        title=f"ECE across {feature_name} range  |  heterogeneity (max-min)={gap:.3f}  [{light}]",
        xlabel=f"{feature_name} quantile bin (low -> high)",
        ylabel="ECE (lower = better calibrated)",
    )

    panels: Tuple[Tuple[PanelSpec, ...], ...] = (
        tuple(mini_panels),
        (ece_line,),
    )
    width = figsize[0] if figsize else max(9.0, 3.2 * len(records))
    height = figsize[1] if figsize else 8.0
    return FigureSpec(
        suptitle=f"{title}{skipped_note}",
        panels=panels,
        figsize=(width, height),
        row_height_ratios=(3.0, 2.0),
        sharey=False,
    )


__all__ = [
    "compose_calibration_by_feature_figure",
    "compute_calibration_by_feature_heterogeneity",
]
