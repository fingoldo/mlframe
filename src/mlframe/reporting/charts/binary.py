"""Binary-classification quality-visualisation panels + composer.

Mirrors the multiclass / regression composer pattern: each token names a panel
builder, ``compose_binary_figure`` parses a space-separated template and packs
the resulting panels into a grid. Binary tasks previously had no curve charts at
all (only a reliability diagram), so this is the curve toolkit the most common
task type was missing.

Token catalogue:
- ``ROC``        -- TPR vs FPR with the chance diagonal; AUC in the title.
- ``PR``         -- precision vs recall with the prevalence no-skill baseline; AP in the title.
- ``SCORE_DIST`` -- overlaid step-histograms of the score for y=0 vs y=1, shared bins, threshold marker.
- ``KS``         -- class-conditional score ECDFs with the max-gap (KS statistic) marked; KS in the title.
- ``THRESHOLD``  -- precision / recall / F1 / queue-rate vs threshold from one score sort + cumulative sums
                    (operating-point picker). An optional cost(t)=c_fp*FP+c_fn*FN series rides along when a
                    cost ratio is supplied.
- ``GAIN``       -- cumulative-gain curve (% positives captured vs % population, score-sorted) + baseline diagonal.
- ``PIT``        -- probability-integral-transform histogram for the binary score (folds the orphan
                    plot_pit_diagram into a spec); uniform = perfect calibration, KS in the title.

Every panel is driven by ONE descending-by-score sort (``_ScoreSort``) computed up
front and shared: THRESHOLD / GAIN / KS read its cumulative TP/FP counts directly,
and ROC / PR / AP are derived from the same per-distinct-threshold counts instead of
re-sorting via sklearn (numerically identical to sklearn, ~1e-16, on distinct AND
tied scores). A 2M-row figure therefore pays a single O(n log n) sort. Curves
decimate to ``_CURVE_VERTEX_CAP`` vertices so a multi-million-row line stays light.

cProfile (n=2M, default template, best of 3): ~0.96s spec build. The remaining cost
is the one shared ``argsort`` (~0.43s); reusing it across ROC/PR/AP removed three
redundant full-n sklearn argsorts (2.72s -> 0.96s, ~2.8x). No further actionable
speedup: the lone sort is irreducible for rank-threshold curves.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mlframe.reporting.charts._layout import (
    figsize_for_grid, pack_panels, parse_panel_template,
)
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec, PanelSpec,
)

# Max vertices kept for any drawn curve. A 200-pt display cannot resolve more, and a
# multi-million-vertex SVG/Scattergl line is both slow to draw and a solid blob.
_CURVE_VERTEX_CAP: int = 2_000
# Max plotted thresholds for the THRESHOLD sweep (one cumulative metric per kept score).
_THRESHOLD_PLOT_CAP: int = 500
# Shared score-distribution / SCORE_DIST histogram bin count.
_SCORE_DIST_BINS: int = 50
# PIT histogram bins (matches the orphan plot_pit_diagram default of 20).
_PIT_BINS: int = 20


# ----------------------------------------------------------------------------
# Shared data prep
# ----------------------------------------------------------------------------


def _finite_binary(y_true, y_score) -> Tuple[np.ndarray, np.ndarray]:
    """Return finite (y_true in {0,1}, y_score) pairs as float64 / int8 arrays.

    Non-finite scores and labels outside {0, 1} are dropped (the binary panels are one-vs-rest
    on the positive class), mirroring how the regression panels drop non-finite pairs up front.
    """
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    mask = np.isfinite(ys)
    yt_f = np.asarray(yt, dtype=np.float64)
    mask &= np.isfinite(yt_f) & ((yt_f == 0.0) | (yt_f == 1.0))
    return yt_f[mask].astype(np.int8), ys[mask]


def _decimate(x: np.ndarray, *ys: np.ndarray, cap: int = _CURVE_VERTEX_CAP):
    """Evenly thin a curve to ``cap`` vertices, always keeping the first and last point.

    Returns ``(x_thin, [y_thin, ...])``. Curve endpoints anchor the axes and the (0,0)/(1,1)
    corners of ROC/PR/gain, so they are forced in even after subsampling the interior.
    """
    n = len(x)
    if n <= cap:
        return x, [y for y in ys]
    idx = np.linspace(0, n - 1, cap).round().astype(np.int64)
    idx[0] = 0
    idx[-1] = n - 1
    idx = np.unique(idx)
    return x[idx], [y[idx] for y in ys]


class _ScoreSort:
    """One descending-by-score sort + the cumulative positive/negative counts every
    rank-threshold panel needs.

    THRESHOLD, GAIN and KS all reduce to "walk the scores from high to low and read a
    cumulative count". Computing the sort once here keeps a 2M-row figure to a single
    O(n log n) pass instead of one per panel.

    Attributes (all length n, aligned to the descending-score order):
        scores_desc : scores sorted high -> low
        cum_tp      : positives seen at/above each rank (1-based prefix count)
        cum_fp      : negatives seen at/above each rank
        n_pos / n_neg : totals
    """

    __slots__ = ("scores_desc", "cum_tp", "cum_fp", "n_pos", "n_neg", "n", "_run_end", "_dtc")

    def __init__(self, y_true: np.ndarray, y_score: np.ndarray):
        order = np.argsort(y_score, kind="stable")[::-1]
        self.scores_desc = y_score[order]
        y_desc = y_true[order].astype(np.int64)
        self.cum_tp = np.cumsum(y_desc)
        self.cum_fp = np.cumsum(1 - y_desc)
        self.n = len(y_true)
        self.n_pos = int(self.cum_tp[-1]) if self.n else 0
        self.n_neg = int(self.cum_fp[-1]) if self.n else 0
        if self.n:
            run_end = np.empty(self.n, dtype=bool)
            run_end[-1] = True
            run_end[:-1] = self.scores_desc[:-1] != self.scores_desc[1:]
        else:
            run_end = np.empty(0, dtype=bool)
        self._run_end = run_end
        self._dtc = None

    def distinct_threshold_counts(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(tps, fps, thresholds) at each DISTINCT score (descending) -- sklearn ``_binary_clf_curve`` shape.

        ROC / PR / AP all reduce to these per-distinct-threshold cumulative TP/FP counts; deriving them from
        the shared sort here avoids the redundant full-n argsort each sklearn curve call would otherwise do.
        Memoised so ROC and PR (both callers) share the one gather.
        """
        if self._dtc is None:
            re = self._run_end
            self._dtc = (self.cum_tp[re].astype(np.float64), self.cum_fp[re].astype(np.float64), self.scores_desc[re])
        return self._dtc


# ----------------------------------------------------------------------------
# Per-token panel builders
# ----------------------------------------------------------------------------


def _roc_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None) -> PanelSpec:
    """ROC curve (TPR vs FPR) with the chance diagonal; AUC in the title.

    Built from the shared score sort's per-distinct-threshold cumulative TP/FP counts (the same quantities
    sklearn's ``_binary_clf_curve`` derives, here without the redundant full-n argsort), prepended with the
    (0,0) origin, then decimated to ``_CURVE_VERTEX_CAP`` so a multi-million-row curve stays light to draw.
    AUC is the trapezoid area on the full per-threshold curve (before decimation).
    """
    if sort.n_pos == 0 or sort.n_neg == 0:
        return AnnotationPanelSpec(text="ROC undefined\n(only one class present)", title="ROC curve")
    tps, fps, _ = sort.distinct_threshold_counts()
    tpr = np.concatenate(([0.0], tps / sort.n_pos))
    fpr = np.concatenate(([0.0], fps / sort.n_neg))
    roc_auc = float(np.trapezoid(tpr, fpr))
    x_thin, (tpr_thin,) = _decimate(fpr, tpr)
    diag = x_thin.copy()
    return LinePanelSpec(
        x=x_thin,
        y=(tpr_thin, diag),
        series_labels=("ROC", "chance"),
        title=f"ROC (AUC={roc_auc:.3f})",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        line_styles=("-", ":"),
        colors=("#1f77b4", "gray"),
    )


def _pr_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None) -> PanelSpec:
    """Precision-recall curve with the prevalence no-skill baseline; AP in the title.

    Built from the shared score sort's per-distinct-threshold cumulative TP/FP counts, matching sklearn's
    ``precision_recall_curve`` (recall ascending, precision = TP/(TP+FP), curve prepended with recall=0 at
    precision=1). AP is the step-sum ``sum (R_i - R_{i-1}) * P_i`` -- sklearn's ``average_precision_score``
    definition -- so a separate full-n argsort is avoided. A curve hugging the dotted prevalence line reads
    as no better than always-positive.
    """
    if sort.n_pos == 0 or sort.n_neg == 0:
        return AnnotationPanelSpec(text="PR undefined\n(only one class present)", title="Precision-Recall curve")
    tps, fps, _ = sort.distinct_threshold_counts()
    precision = tps / np.maximum(tps + fps, 1.0)
    recall = tps / sort.n_pos
    ap = float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))
    # sklearn prepends (recall=0, precision=1) so the curve starts at the y-axis; reverse to recall-ascending for plotting.
    rec_plot = np.concatenate(([0.0], recall))
    prec_plot = np.concatenate(([1.0], precision))
    x_thin, (prec_thin,) = _decimate(rec_plot, prec_plot)
    prevalence = sort.n_pos / max(1, sort.n)
    baseline = np.full_like(x_thin, prevalence)
    return LinePanelSpec(
        x=x_thin,
        y=(prec_thin, baseline),
        series_labels=("PR", f"no-skill (prev={prevalence:.3f})"),
        title=f"Precision-Recall (AP={ap:.3f})",
        xlabel="Recall",
        ylabel="Precision",
        line_styles=("-", ":"),
        colors=("#2ca02c", "gray"),
    )


def _score_dist_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None) -> PanelSpec:
    """Overlaid step-histograms of the score for y=0 vs y=1 on shared bins, with a threshold marker.

    Two well-separated humps = a discriminating model; heavy overlap = poor separability. Pre-binned with
    a single ``np.histogram`` per class on common edges so a 2M-row figure does not ship raw points to the renderer.
    """
    if sort.n == 0:
        return AnnotationPanelSpec(text="No finite (label, score) pairs", title="Score distribution by class")
    lo, hi = float(ys.min()), float(ys.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, _SCORE_DIST_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    neg = ys[yt == 0]
    pos = ys[yt == 1]
    h_neg, _ = np.histogram(neg, bins=edges, density=True)
    h_pos, _ = np.histogram(pos, bins=edges, density=True)
    return LinePanelSpec(
        x=centers,
        y=(h_neg, h_pos),
        series_labels=("y=0", "y=1"),
        title="Score distribution by class",
        xlabel="Predicted score",
        ylabel="Density",
        line_styles=("-", "-"),
        colors=("#d62728", "#1f77b4"),
        vlines=((float(threshold), "black", f"threshold={threshold:.2f}"),),
        # Translucent step fills under each class histogram so the two distributions read as filled humps; step_fill
        # keeps the left-closed histogram shape rather than linearly interpolating between bin centers.
        fill_to_baseline=(True, True),
        step_fill=True,
    )


def _ks_curve(sort: _ScoreSort) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Class-conditional score ECDFs evaluated at every distinct score, plus the KS gap.

    Returns ``(score_grid, cdf_pos, cdf_neg, ks_stat, ks_score)`` where the ECDFs are the
    fraction of each class with score <= grid value (ascending), ks_stat = max |cdf_neg - cdf_pos|,
    and ks_score is the score at which that maximum occurs. O(n) given the precomputed descending cumsums.
    """
    # cum_tp / cum_fp are descending-prefix counts (score >= rank); reverse to ascending CDFs (score <= rank).
    pos_le = sort.n_pos - np.concatenate(([0], sort.cum_tp[:-1]))[::-1]
    neg_le = sort.n_neg - np.concatenate(([0], sort.cum_fp[:-1]))[::-1]
    grid = sort.scores_desc[::-1]
    cdf_pos = pos_le / max(1, sort.n_pos)
    cdf_neg = neg_le / max(1, sort.n_neg)
    gap = np.abs(cdf_neg - cdf_pos)
    j = int(np.argmax(gap))
    return grid, cdf_pos, cdf_neg, float(gap[j]), float(grid[j])


def _ks_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None) -> PanelSpec:
    """Class-conditional score ECDFs with the max-gap (KS statistic) marked; KS value in the title.

    The credit-scoring separability standard: KS = max vertical distance between the y=0 and y=1
    score CDFs. The dashed vline marks the score where that gap is largest.
    """
    if sort.n_pos == 0 or sort.n_neg == 0:
        return AnnotationPanelSpec(text="KS undefined\n(only one class present)", title="KS statistic")
    grid, cdf_pos, cdf_neg, ks_stat, ks_score = _ks_curve(sort)
    x_thin, (pos_thin, neg_thin) = _decimate(grid, cdf_pos, cdf_neg)
    return LinePanelSpec(
        x=x_thin,
        y=(neg_thin, pos_thin),
        series_labels=("CDF y=0", "CDF y=1"),
        title=f"KS statistic = {ks_stat:.3f}",
        xlabel="Score threshold",
        ylabel="Cumulative fraction (score <= t)",
        line_styles=("-", "-"),
        colors=("#d62728", "#1f77b4"),
        vlines=((ks_score, "black", f"KS@{ks_score:.2f}"),),
    )


def _threshold_sweep(sort: _ScoreSort) -> Dict[str, np.ndarray]:
    """Vectorized precision / recall / F1 / queue-rate at every distinct score threshold.

    For threshold t, the operating point predicts positive where ``score >= t``. With the scores in
    descending order, that predicate's boundary sits at the LAST rank of each tied-score run, so the
    sweep is reported at those run-ends only -- this makes it bit-identical to a per-threshold
    ``precision_score((score >= t))`` reference even when scores tie (rank-i mid-run counts would not be,
    since a tie cannot be split). At a kept rank i (1-based): TP=cum_tp[i-1], FP=cum_fp[i-1],
    FN=n_pos-TP, queue=i/n. All series come from the one precomputed cumsum pair, so this is O(n) with no
    per-threshold sklearn call.
    """
    n = sort.n
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return {k: empty for k in ("thresholds", "precision", "recall", "f1", "queue_rate", "fp", "fn")}
    scores = sort.scores_desc
    # Keep one row per distinct score (the run-ends precomputed on the shared sort); the score >= t
    # predicate boundary sits at the last rank of each tied-score run, so reporting only there is
    # bit-identical to a per-threshold sklearn reference even under ties (a tie cannot be split).
    run_end = sort._run_end
    ranks = (np.arange(1, n + 1, dtype=np.float64))[run_end]
    tp = sort.cum_tp.astype(np.float64)[run_end]
    fp = sort.cum_fp.astype(np.float64)[run_end]
    fn = sort.n_pos - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = tp / max(1, sort.n_pos)
        denom = 2.0 * tp + fp + fn
        f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    queue_rate = ranks / max(1, n)
    return {
        "thresholds": scores[run_end],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "queue_rate": queue_rate,
        "fp": fp,
        "fn": fn,
    }


def _threshold_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None) -> PanelSpec:
    """Precision / recall / F1 / queue-rate vs threshold (operating-point picker).

    Computed vectorized from the shared score sort. When ``cost_ratio`` (= c_fp / c_fn, or a
    ``(c_fp, c_fn)`` pair) is supplied, a normalised cost(t) = c_fp*FP + c_fn*FN series is added so
    the operator can read the money-minimising threshold off the same axes.
    """
    if sort.n_pos == 0 or sort.n_neg == 0:
        return AnnotationPanelSpec(text="Threshold sweep undefined\n(only one class present)", title="Threshold sweep")
    sweep = _threshold_sweep(sort)
    thr, ys_list = _decimate(
        sweep["thresholds"], sweep["precision"], sweep["recall"], sweep["f1"], sweep["queue_rate"],
        cap=_THRESHOLD_PLOT_CAP,
    )
    prec, rec, f1, queue = ys_list
    series: List[np.ndarray] = [prec, rec, f1, queue]
    labels: List[str] = ["precision", "recall", "F1", "queue-rate"]
    styles: List[str] = ["-", "-", "-", "--"]
    colors: List[str] = ["#1f77b4", "#2ca02c", "#9467bd", "#7f7f7f"]
    # Queue-rate is the fraction of the population flagged at each threshold -- an operating-volume axis, not a quality
    # metric -- so it rides a secondary y-axis where its own range reads independently of the [0,1] precision/recall/F1.
    secondary: List[bool] = [False, False, False, True]
    if cost_ratio is not None:
        if isinstance(cost_ratio, (tuple, list)) and len(cost_ratio) == 2:
            c_fp, c_fn = float(cost_ratio[0]), float(cost_ratio[1])
        else:
            c_fp, c_fn = float(cost_ratio), 1.0
        cost = c_fp * sweep["fp"] + c_fn * sweep["fn"]
        cmax = float(cost.max()) if cost.size and cost.max() > 0 else 1.0
        _, (cost_thin,) = _decimate(sweep["thresholds"], cost / cmax, cap=_THRESHOLD_PLOT_CAP)
        series.append(cost_thin)
        labels.append(f"cost (c_fp={c_fp:g}, c_fn={c_fn:g}, norm)")
        styles.append(":")
        colors.append("#d62728")
        secondary.append(False)
    return LinePanelSpec(
        x=thr,
        y=tuple(series),
        series_labels=tuple(labels),
        title="Metrics vs threshold",
        xlabel="Score threshold",
        ylabel="Metric value",
        line_styles=tuple(styles),
        colors=tuple(colors),
        secondary_y=tuple(secondary),
        secondary_ylabel="Queue rate (fraction flagged)",
    )


def _gain_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None) -> PanelSpec:
    """Cumulative-gain curve: fraction of positives captured vs fraction of population (score-sorted).

    The baseline diagonal is random targeting; the gap above it is the model's lift. Decimated to a
    bounded vertex count; the (0,0) and (1,1) corners are forced in by the endpoint-preserving decimator.
    """
    if sort.n_pos == 0:
        return AnnotationPanelSpec(text="Gain undefined\n(no positive samples)", title="Cumulative gain")
    pop_frac = np.arange(1, sort.n + 1, dtype=np.float64) / sort.n
    gain = sort.cum_tp.astype(np.float64) / sort.n_pos
    # Prepend (0, 0) so the curve starts at the origin like the canonical gains chart.
    pop_frac = np.concatenate(([0.0], pop_frac))
    gain = np.concatenate(([0.0], gain))
    x_thin, (gain_thin,) = _decimate(pop_frac, gain)
    diag = x_thin.copy()
    return LinePanelSpec(
        x=x_thin,
        y=(gain_thin, diag),
        series_labels=("model", "baseline"),
        title="Cumulative gain",
        xlabel="Fraction of population (score-sorted)",
        ylabel="Fraction of positives captured",
        line_styles=("-", ":"),
        colors=("#1f77b4", "gray"),
        # Fill the model gain curve down to the baseline so the lift area (gap above the random diagonal) is shaded.
        fill_to_baseline=(True, False),
    )


def _pit_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None) -> PanelSpec:
    """Probability-integral-transform histogram for the binary score; uniform = perfect calibration.

    PIT value = score when y=1 else (1 - score). A well-calibrated model makes PIT ~ Uniform(0,1), so a
    flat histogram at density 1 is the target; humps reveal over/under-confidence. KS-vs-uniform in the title.
    """
    if sort.n == 0:
        return AnnotationPanelSpec(text="PIT undefined\n(no finite pairs)", title="PIT diagram")
    pit = np.where(yt == 1, ys, 1.0 - ys)
    pit = np.clip(pit, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, _PIT_BINS + 1)
    heights, _ = np.histogram(pit, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ks_stat = _ks_vs_uniform(pit)
    return HistogramPanelSpec(
        values=heights,
        bin_centers=centers,
        bin_width=float(edges[1] - edges[0]),
        title=f"PIT diagram (KS-vs-uniform={ks_stat:.3f})",
        xlabel="PIT value",
        ylabel="Density",
        density=False,
    )


def _ks_vs_uniform(pit: np.ndarray) -> float:
    """KS distance of the PIT sample from Uniform(0,1): max |ECDF(x) - x|. O(n log n) sort, no scipy dep."""
    n = len(pit)
    if n == 0:
        return float("nan")
    s = np.sort(pit)
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    return float(np.maximum(np.abs(ecdf_hi - s), np.abs(s - ecdf_lo)).max())


# ----------------------------------------------------------------------------
# Decile table (data the integrator surfaces in metrics, not a panel)
# ----------------------------------------------------------------------------


def binary_decile_table(y_true, y_score, *, n_deciles: int = 10) -> Dict[str, np.ndarray]:
    """Per-decile gains/lift/KS table for a binary scorer (score-sorted, decile 1 = highest scores).

    Returns a dict of length-``n_deciles`` arrays:
        decile        : 1..n_deciles (1 = top-scored group)
        count         : rows in the decile
        positives     : positive rows in the decile
        response_rate : positives / count
        gain          : cumulative fraction of all positives captured by deciles 1..d
        lift          : (cumulative positive rate through decile d) / overall prevalence
        cum_ks        : cumulative |%positives - %negatives| captured through decile d (the decile-resolution KS)

    The gains/lift toolkit that completes the existing lift curve; the integrator surfaces this
    as a metrics table rather than a chart panel.
    """
    yt, ys = _finite_binary(y_true, y_score)
    n = len(yt)
    out = {
        "decile": np.arange(1, n_deciles + 1, dtype=np.int64),
        "count": np.zeros(n_deciles, dtype=np.int64),
        "positives": np.zeros(n_deciles, dtype=np.int64),
        "response_rate": np.full(n_deciles, np.nan),
        "gain": np.full(n_deciles, np.nan),
        "lift": np.full(n_deciles, np.nan),
        "cum_ks": np.full(n_deciles, np.nan),
    }
    if n == 0:
        return out
    order = np.argsort(ys, kind="stable")[::-1]
    y_desc = yt[order].astype(np.int64)
    # Split the score-sorted rows into ~equal deciles via integer boundaries (handles n not divisible by n_deciles).
    bounds = (np.arange(n_deciles + 1) * n / n_deciles).round().astype(np.int64)
    n_pos_total = int(y_desc.sum())
    n_neg_total = n - n_pos_total
    prevalence = n_pos_total / n if n else 0.0
    cum_pos = 0
    cum_neg = 0
    cum_count = 0
    for d in range(n_deciles):
        lo, hi = bounds[d], bounds[d + 1]
        seg = y_desc[lo:hi]
        cnt = len(seg)
        pos = int(seg.sum())
        out["count"][d] = cnt
        out["positives"][d] = pos
        if cnt > 0:
            out["response_rate"][d] = pos / cnt
        cum_pos += pos
        cum_neg += cnt - pos
        cum_count += cnt
        if n_pos_total > 0:
            out["gain"][d] = cum_pos / n_pos_total
        if prevalence > 0 and cum_count > 0:
            out["lift"][d] = (cum_pos / cum_count) / prevalence
        frac_pos = cum_pos / n_pos_total if n_pos_total > 0 else 0.0
        frac_neg = cum_neg / n_neg_total if n_neg_total > 0 else 0.0
        out["cum_ks"][d] = abs(frac_pos - frac_neg)
    return out


# ----------------------------------------------------------------------------
# Token registry + composer
# ----------------------------------------------------------------------------


_TOKEN_BUILDERS: Dict[str, Callable] = {
    "ROC": _roc_panel,
    "PR": _pr_panel,
    "SCORE_DIST": _score_dist_panel,
    "KS": _ks_panel,
    "THRESHOLD": _threshold_panel,
    "GAIN": _gain_panel,
    "PIT": _pit_panel,
}

ALLOWED_BINARY_PANEL_TOKENS = frozenset(_TOKEN_BUILDERS)

DEFAULT_BINARY_PANELS: str = "ROC PR SCORE_DIST KS THRESHOLD GAIN"


def compose_binary_figure(
    y_true,
    y_score: np.ndarray,
    *,
    panels_template: str = DEFAULT_BINARY_PANELS,
    threshold: float = 0.5,
    cost_ratio=None,
    suptitle: str = "",
    max_cols: int = 2,
    cell_width: float = 6.0,
    cell_height: float = 4.0,
) -> FigureSpec:
    """Build a binary-classification quality FigureSpec from a panel template.

    Parameters
    ----------
    y_true : (N,) array of binary labels in {0, 1}.
    y_score : (N,) array of positive-class scores / probabilities.
    panels_template : space-separated token list (see the module docstring). Default
        ``DEFAULT_BINARY_PANELS`` = "ROC PR SCORE_DIST KS THRESHOLD GAIN".
    threshold : operating-point marker drawn on SCORE_DIST (default 0.5).
    cost_ratio : optional c_fp/c_fn scalar or ``(c_fp, c_fn)`` pair; adds the cost(t) series
        to the THRESHOLD panel (default off).
    suptitle : figure suptitle (model identity).
    max_cols : grid width (default 2).
    """
    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(
            f"Unknown binary panel tokens {unknown}. "
            f"Allowed: {sorted(ALLOWED_BINARY_PANEL_TOKENS)}"
        )
    yt, ys = _finite_binary(y_true, y_score)
    # Single descending-score sort + cumulative counts shared by every rank-threshold panel.
    sort = _ScoreSort(yt, ys)
    panels: List[PanelSpec] = []
    for tok in tokens:
        panels.append(_TOKEN_BUILDERS[tok](yt, ys, sort=sort, threshold=threshold, cost_ratio=cost_ratio))
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    return FigureSpec(
        suptitle=suptitle,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols, cell_width=cell_width, cell_height=cell_height),
    )


__all__ = [
    "ALLOWED_BINARY_PANEL_TOKENS",
    "DEFAULT_BINARY_PANELS",
    "compose_binary_figure",
    "binary_decile_table",
]
