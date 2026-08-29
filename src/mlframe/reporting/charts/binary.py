"""Binary-classification quality-visualisation panels + composer.

Mirrors the multiclass / regression composer pattern: each token names a panel
builder, ``compose_binary_figure`` parses a space-separated template and packs
the resulting panels into a grid. Binary tasks previously had no curve charts at
all (only a reliability diagram), so this is the curve toolkit the most common
task type was missing.

Token catalogue:
- ``ROC``  -- TPR vs FPR with the chance diagonal; AUC in the title.
- ``PR``  -- precision vs recall with the prevalence no-skill baseline; AP in the title.
- ``SCORE_DIST`` -- overlaid step-histograms of the score for y=0 vs y=1, shared bins, threshold marker.
- ``KS``  -- class-conditional score ECDFs with the max-gap (KS statistic) marked; KS in the title.
- ``THRESHOLD``  -- precision / recall / F1 / queue-rate vs threshold from one score sort + cumulative sums
                    (operating-point picker). An optional cost(t)=c_fp*FP+c_fn*FN series rides along when a
                    cost ratio is supplied.
- ``GAIN``  -- cumulative-gain curve (% positives captured vs % population, score-sorted) + baseline diagonal.
- ``PIT``  -- probability-integral-transform histogram for the binary score (folds the orphan
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

import logging

import hashlib
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ._captions import caption_for_tokens

logger = logging.getLogger(__name__)

from mlframe.metrics import trapezoid
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
# Bootstrap resamples for the PR-AUC (average precision) CI. AP has no closed-form variance (unlike ROC-AUC's DeLong),
# so we resample rows; B in this range keeps the 2.5/97.5 percentiles stable while staying sub-second under the cap.
_AP_BOOTSTRAP_B: int = 500
# AP CI width is driven by n; a subsample of this size has a representative CI and bounds the (B, m) gather cost.
_AP_BOOTSTRAP_ROW_CAP: int = 50_000
# Below this many rows (or positives) a bootstrap AP CI is too noisy to be informative; annotate AP only.
_AP_BOOTSTRAP_MIN_N: int = 30
# Resample-index rows drawn at a time. One (B, m) matrix is 200 MB at B=500 / m=50k for a buffer consumed one
# row at a time; chunking keeps the peak small without changing the draw order (so the interval is unchanged).
_AP_BOOTSTRAP_IDX_CHUNK: int = 32


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
    label_ok = np.isfinite(yt_f) & ((yt_f == 0.0) | (yt_f == 1.0))
    mask &= label_ok
    # Dropping an off-{0,1} label silently means a multiclass target passed here by mistake yields confident
    # binary curves computed on whatever subset happened to be 0 or 1 -- a plausible-looking chart about a
    # question nobody asked. Say how many went, and name the offending labels.
    n_bad_label = int(np.count_nonzero(np.isfinite(yt_f) & ~label_ok))
    if n_bad_label:
        offenders = np.unique(yt_f[np.isfinite(yt_f) & ~label_ok])[:5].tolist()
        logger.warning(
            "binary charts: dropped %d of %d rows whose label is outside {0, 1} (saw %s). These panels are "
            "one-vs-rest on the positive class; pass a binarised target, or use the multiclass composer.",
            n_bad_label, yt_f.size, offenders,
        )
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

    __slots__ = ("_dtc", "_run_end", "cum_fp", "cum_tp", "n", "n_neg", "n_pos", "scores_desc")

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
        self._dtc: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

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


def _operating_point(sort: _ScoreSort, threshold: float) -> Optional[Tuple[float, float, float, float]]:
    """Confusion-derived (FPR, TPR, recall, precision) at the ``score >= threshold`` operating point, or None.

    Locates the threshold's rank via ``searchsorted`` on the SHARED descending-score array (O(log n), no new full-n
    pass): the rank ``k`` is the count of scores ``>= threshold``, read off as TP=cum_tp[k-1], FP=cum_fp[k-1]. Returns
    None when the marker is undefined -- single-class (no TPR or no FPR axis) or a threshold outside the score range
    that flags nobody (k==0) -- so the caller omits the marker but keeps the curve.
    """
    if sort.n == 0 or sort.n_pos == 0 or sort.n_neg == 0:
        return None
    # scores_desc is descending; -scores_desc is ascending, so searchsorted(side='right') counts scores >= threshold.
    k = int(np.searchsorted(-sort.scores_desc, -float(threshold), side="right"))
    if k == 0:
        return None
    tp = float(sort.cum_tp[k - 1])
    fp = float(sort.cum_fp[k - 1])
    tpr = tp / sort.n_pos
    fpr = fp / sort.n_neg
    precision = tp / max(1.0, tp + fp)
    return fpr, tpr, tpr, precision


def _operating_point_label(threshold: float, a: float, b: float, kind: str) -> str:
    """Short marker label: ``thr=0.50: TPR=0.81 FPR=0.12`` (ROC) or ``thr=0.50: R=0.81 P=0.74`` (PR)."""
    if kind == "roc":
        return f"thr={threshold:.2f}: TPR={a:.2f} FPR={b:.2f}"
    return f"thr={threshold:.2f}: R={a:.2f} P={b:.2f}"


def _roc_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None, operating_point: bool = True) -> PanelSpec:
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
    roc_auc = trapezoid(tpr, fpr)
    x_thin, (tpr_thin,) = _decimate(fpr, tpr)
    diag = x_thin.copy()
    markers = None
    if operating_point:
        op = _operating_point(sort, threshold)
        if op is not None:
            op_fpr, op_tpr, _, _ = op
            markers = ((op_fpr, op_tpr, _operating_point_label(threshold, op_tpr, op_fpr, "roc"), "#d62728", "*"),)
    return LinePanelSpec(
        x=x_thin,
        y=(tpr_thin, diag),
        series_labels=("ROC", "chance"),
        title=f"ROC (AUC={roc_auc:.3f})",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        line_styles=("-", ":"),
        colors=("#1f77b4", "gray"),
        point_markers=markers,
    )


def bootstrap_ap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_boot: int = _AP_BOOTSTRAP_B,
    alpha: float = 0.05,
    seed: int = 0,
    sort: Optional["_ScoreSort"] = None,
) -> Tuple[float, float, float]:
    """Bootstrap percentile CI for PR-AUC (average precision), resampling rows with replacement.

    Returns ``(ap, lower, upper)`` -- the full-data AP plus the ``1-alpha`` percentile interval. AP has no closed-form
    variance (unlike ROC-AUC's DeLong), so the row bootstrap is the honest uncertainty estimate. Vectorised: subsample
    to ``_AP_BOOTSTRAP_ROW_CAP`` rows first (the CI width tracks n, which the cap preserves up to its bound), sort that
    subsample by score ONCE descending, then draw a single ``(n_boot, m)`` index gather. Each resample's AP comes from
    per-rank cumulative TP/FP counts accumulated against the shared descending order -- ``np.add.at`` scatters each
    drawn row's multiplicity to its rank, so one cumsum per resample yields the step-sum AP with no per-B re-sort.

    Deterministic given ``seed``. Returns ``(ap, nan, nan)`` when n < ``_AP_BOOTSTRAP_MIN_N`` or one class is absent
    (AP undefined / CI uninformative); the caller annotates AP without an interval there.

    ``sort``: the caller's existing ``_ScoreSort`` over the SAME rows. The full-data AP needs the labels in
    descending-score order, which that object already holds, so passing it skips a second full-n argsort on the
    default-on path. Optional and keyword-only -- callers that do not have one are unaffected.
    """
    yt = np.asarray(y_true).ravel().astype(np.int8)
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    n = yt.size
    n_pos_full = int(yt.sum())
    if n == 0 or n_pos_full == 0 or n_pos_full == n:
        return float("nan"), float("nan"), float("nan")
    if sort is not None and sort.n == n:
        # cum_tp is the 1-based prefix count of positives in descending-score order, so its first difference IS the
        # descending label sequence -- the same array the argsort below would produce, without the sort.
        labels_desc = np.diff(np.concatenate(([0], sort.cum_tp)))
    else:
        labels_desc = yt[np.argsort(ys, kind="stable")[::-1]]
    full_ap = _ap_from_labels_desc(labels_desc)
    if n < _AP_BOOTSTRAP_MIN_N:
        return full_ap, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = min(n, _AP_BOOTSTRAP_ROW_CAP)
    if m < n:
        sub = rng.choice(n, size=m, replace=False)
        yt, ys = yt[sub], ys[sub]
    # Sort the subsample by score ONCE (descending); resamples then index into this fixed rank order.
    order = np.argsort(ys, kind="stable")[::-1]
    yt_desc = yt[order].astype(np.int64)
    # Draw the resample indices in CHUNKS. One ``(n_boot, m)`` matrix is 200 MB at B=500 / m=50k, allocated up
    # front and then consumed one row at a time -- against the house RAM rule for a buffer nothing needs whole.
    # Chunking keeps the peak at ``_AP_BOOTSTRAP_IDX_CHUNK`` rows while drawing from the SAME rng in the same
    # order, so the interval is unchanged.
    boot_ap = np.empty(n_boot, dtype=np.float64)
    pos_desc = yt_desc.astype(np.float64)
    idx_chunk: np.ndarray = np.empty((0, 0), dtype=np.int64)
    for b in range(n_boot):
        if b % _AP_BOOTSTRAP_IDX_CHUNK == 0:
            idx_chunk = rng.integers(0, m, size=(min(_AP_BOOTSTRAP_IDX_CHUNK, n_boot - b), m))
        # Multiplicity of each fixed-rank row in resample b (a row can be drawn 0..k times); scatter-add to its rank.
        mult = np.bincount(idx_chunk[b % _AP_BOOTSTRAP_IDX_CHUNK], minlength=m).astype(np.float64)
        tp = np.cumsum(mult * pos_desc)
        total = np.cumsum(mult)
        n_pos = tp[-1]
        if n_pos <= 0:
            boot_ap[b] = float("nan")
            continue
        # AP = sum (R_i - R_{i-1}) * P_i; the recall step is the resampled positive mass at rank i, so AP collapses to
        # the precision-weighted positive multiplicity / n_pos -- avoids a per-resample concat + diff.
        precision = tp / np.maximum(total, 1.0)
        boot_ap[b] = np.dot(mult * pos_desc, precision) / n_pos
    boot_ap = boot_ap[~np.isnan(boot_ap)]
    if boot_ap.size == 0:
        return full_ap, float("nan"), float("nan")
    lo = float(np.percentile(boot_ap, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_ap, 100.0 * (1.0 - alpha / 2.0)))
    if m < n:
        # The point estimate is computed on all n rows but the interval was bootstrapped from m of them, so the
        # title paired a full-n AP with the uncertainty of an m-row study -- about 6x too wide at n=2M against
        # the 50k cap. A bootstrap standard error scales as 1/sqrt(rows), so rescale the half-width back onto
        # the real row count. The interval stays centred on the point estimate the title actually reports.
        shrink = float(np.sqrt(m / n))
        centre = float(np.median(boot_ap))
        lo = full_ap + (lo - centre) * shrink
        hi = full_ap + (hi - centre) * shrink
    return full_ap, lo, hi


def _ap_from_labels_desc(labels_desc: np.ndarray) -> float:
    """Average precision from labels already sorted descending-by-score (step-sum ``sum (R_i - R_{i-1}) * P_i``)."""
    lab = labels_desc.astype(np.float64)
    tp = np.cumsum(lab)
    fp = np.cumsum(1.0 - lab)
    n_pos = tp[-1]
    if n_pos <= 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def _pr_panel(yt: np.ndarray, ys: np.ndarray, *, sort: _ScoreSort, threshold: float, cost_ratio=None,
              ap_ci: bool = True, ap_ci_seed: int = 0, operating_point: bool = True) -> PanelSpec:
    """Precision-recall curve with the prevalence no-skill baseline; AP + bootstrap 95% CI in the title.

    Built from the shared score sort's per-distinct-threshold cumulative TP/FP counts, matching sklearn's
    ``precision_recall_curve`` (recall ascending, precision = TP/(TP+FP), curve prepended with recall=0 at
    precision=1). AP is the step-sum ``sum (R_i - R_{i-1}) * P_i`` -- sklearn's ``average_precision_score``
    definition -- so a separate full-n argsort is avoided. A curve hugging the dotted prevalence line reads
    as no better than always-positive.

    ``ap_ci`` (default on) appends a 95% percentile bootstrap CI to the title -- AP has no closed-form variance, so
    this is the honest counterpart to ROC's DeLong CI. Single-class / tiny-n -> AP shown without an interval.
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
    title = f"Precision-Recall (AP={ap:.3f})"
    if ap_ci:
        _, lo, hi = bootstrap_ap_ci(yt, ys, seed=ap_ci_seed, sort=sort)
        if np.isfinite(lo) and np.isfinite(hi):
            title = f"Precision-Recall (AP={ap:.3f} [{lo:.3f}, {hi:.3f}], 95% CI)"
    markers = None
    if operating_point:
        op = _operating_point(sort, threshold)
        if op is not None:
            _, _, op_rec, op_prec = op
            markers = ((op_rec, op_prec, _operating_point_label(threshold, op_rec, op_prec, "pr"), "#d62728", "*"),)
    return LinePanelSpec(
        x=x_thin,
        y=(prec_thin, baseline),
        series_labels=("PR", f"no-skill (prev={prevalence:.3f})"),
        title=title,
        xlabel="Recall",
        ylabel="Precision",
        line_styles=("-", ":"),
        colors=("#2ca02c", "gray"),
        point_markers=markers,
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
        # Densities each integrate to 1, so a 1000:1 imbalance draws two equally tall humps and the reader
        # cannot see it at all. The shape comparison is still what the panel is FOR, so keep the density and
        # put the counts and shares in the labels, where the imbalance becomes readable.
        series_labels=(
            f"y=0 (n={neg.size:,}, {neg.size / max(sort.n, 1):.1%})",
            f"y=1 (n={pos.size:,}, {pos.size / max(sort.n, 1):.1%})",
        ),
        title="Score distribution by class (density: each curve integrates to 1, so heights are NOT counts)",
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
    # Evaluate at DISTINCT scores only. The per-ROW form below stepped the ECDFs through the interior of a
    # tied block, i.e. at cut points no threshold can actually realise, which biases KS upward on quantised
    # scores -- exactly the shape a tree ensemble emits. Measured on 20-level scores: 0.494441 against a
    # 0.493870 reference computed over realisable thresholds only. `distinct_threshold_counts` already
    # collapses ties and is memoised, so this is also strictly less work.
    tps, fps, thr = sort.distinct_threshold_counts()
    # tps[i] counts positives with score >= thr[i] (thr descending), so positives with score <= thr[i] is
    # n_pos minus those strictly ABOVE it, which is tps at the previous (higher) distinct threshold.
    pos_gt = np.concatenate(([0.0], tps[:-1]))
    neg_gt = np.concatenate(([0.0], fps[:-1]))
    grid = thr[::-1]
    cdf_pos = (sort.n_pos - pos_gt)[::-1] / max(1, sort.n_pos)
    cdf_neg = (sort.n_neg - neg_gt)[::-1] / max(1, sort.n_neg)
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
    markers: List[tuple] = []
    _f1_full = sweep["f1"]
    if _f1_full.size and np.isfinite(_f1_full).any():
        _bi = int(np.nanargmax(_f1_full))
        markers.append((float(sweep["thresholds"][_bi]), float(_f1_full[_bi]),
                        f"F1 optimum @ {sweep['thresholds'][_bi]:.3f} (F1={_f1_full[_bi]:.3f})", "#9467bd", "*"))
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
        # Min-max, not max-only: a cost that never comes near zero (both error types always present) compressed into
        # a flat band near the top of an axis it shares with [0,1] metrics, hiding the minimum this series exists to
        # show. Normalising to its own range puts that minimum on the axis.
        cmin = float(cost.min()) if cost.size else 0.0
        cmax = float(cost.max()) if cost.size else 1.0
        span = (cmax - cmin) if cmax > cmin else 1.0
        cost_norm = (cost - cmin) / span
        _, (cost_thin,) = _decimate(sweep["thresholds"], cost_norm, cap=_THRESHOLD_PLOT_CAP)
        if cost.size:
            _ci = int(np.argmin(cost))
            markers.append((float(sweep["thresholds"][_ci]), float(cost_norm[_ci]), f"min cost @ {sweep['thresholds'][_ci]:.3f}", "#d62728", "D"))
        series.append(cost_thin)
        labels.append(f"cost (c_fp={c_fp:g}, c_fn={c_fn:g}, min-max normalised)")
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
        point_markers=tuple(markers) or None,
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
    """Randomised probability-integral-transform histogram; uniform = perfect calibration.

    The PIT is only uniform when the outcome is CONTINUOUS. A binary outcome makes the predictive CDF a step
    function, so the plain transform ``score if y==1 else 1 - score`` is not uniform even for a perfectly
    calibrated model: it yields a triangular density rising from 0.10 to 1.90 across the deciles, and a
    KS-vs-uniform of 0.247 (measured on 200k rows drawn as ``y ~ Bernoulli(p)``, i.e. calibrated by
    construction). The panel therefore condemned exactly the models it was built to certify.

    Czado/Gneiting-Held randomisation restores uniformity for a discrete outcome by drawing within the CDF's
    jump at the realised value: ``U ~ Uniform(F(y-1), F(y))``, which for Bernoulli(p) is ``Uniform(0, 1-p)``
    when y=0 and ``Uniform(1-p, 1)`` when y=1. On the same calibrated sample that gives KS 0.002 and a flat
    density, while genuine over/under-confidence still shows as the same humps it always did. The draw is
    seeded so a re-render of the same data reproduces the same figure.
    """
    if sort.n == 0:
        return AnnotationPanelSpec(text="PIT undefined\n(no finite pairs)", title="PIT diagram")
    p = np.clip(ys, 0.0, 1.0)
    # Seed from the DATA, not a fixed constant. A constant seed is reproducible but can silently reproduce the
    # caller's own stream: a test generating scores from default_rng(0) and a panel drawing from default_rng(0)
    # get u == p exactly, the randomisation cancels, and the KS jumps back to the 0.25 this fix exists to remove.
    # Hashing the scores keeps the figure reproducible for a given dataset while making that collision impossible.
    _digest = hashlib.blake2b(np.ascontiguousarray(p).tobytes(), digest_size=8).digest()
    u = np.random.default_rng(int.from_bytes(_digest, "little")).random(p.shape[0])
    pit = np.where(yt == 1, (1.0 - p) + u * p, u * (1.0 - p))
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


# Columns drawn in the decile table, in order: (table-header, source-key-or-None, value-formatter).
_DECILE_TABLE_COLUMNS: Tuple[Tuple[str, str, Callable], ...] = (
    ("decile", "decile", lambda v: f"{int(v)}"),
    ("n", "count", lambda v: f"{int(v):,}"),
    ("positives", "positives", lambda v: f"{int(v):,}"),
    ("response", "response_rate", lambda v: "-" if not np.isfinite(v) else f"{v:.1%}"),
    ("cum gain", "gain", lambda v: "-" if not np.isfinite(v) else f"{v:.1%}"),
    ("lift", "lift", lambda v: "-" if not np.isfinite(v) else f"{v:.2f}"),
    ("cum KS", "cum_ks", lambda v: "-" if not np.isfinite(v) else f"{v:.3f}"),
)


def binary_decile_table_figure(
    y_true,
    y_score,
    *,
    n_deciles: int = 10,
    highlight_top: int = 3,
    title: str = "Decile gain / lift table",
    figsize: Optional[Tuple[float, float]] = None,
):
    """Render the score-sorted decile gain/lift/KS table (decile 1 = top scores) as a styled matplotlib table figure.

    The tabular complement to the GAIN curve: stakeholders read the exact per-decile capture / lift / cumulative-KS
    numbers a curve only shows graphically. All numbers come from ONE call to ``binary_decile_table`` (a single
    O(n log n) score sort) -- no per-decile rescans. The top ``highlight_top`` deciles are tinted, the cumulative-gain
    column carries a light value-proportional shade, and a TOTAL row sums n / positives with the overall response rate.

    Edge cases mirror the iter-1 guard style: a single-class target (gain/lift undefined) or fewer than ``n_deciles``
    finite rows renders a centered annotation instead of a misleading table. Returns a matplotlib ``Figure`` (the
    SHAP-style direct-matplotlib path; the heavy aggregation stays spec-pure in ``binary_decile_table``).
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    yt, ys = _finite_binary(y_true, y_score)
    n = len(yt)
    n_pos = int(yt.sum()) if n else 0

    def _annotated(msg: str):
        """Render a bare, title-only figure carrying a centered explanatory message in place of a table (degenerate-input fallback)."""
        fig = Figure(figsize=figsize or (8.0, 2.4))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.set_title(title, fontsize=11)
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11, transform=ax.transAxes)
        return fig

    if n == 0:
        return _annotated("Decile table undefined\n(no finite (label, score) pairs)")
    if n_pos == 0 or n_pos == n:
        return _annotated("Decile gain / lift undefined\n(only one class present)")
    # With fewer rows than deciles every decile would hold <=1 row -- the per-decile rates are noise; bin to n rows.
    eff_deciles = n_deciles if n >= n_deciles else max(1, n)
    note = "" if n >= n_deciles else f" (n={n} < {n_deciles}: {eff_deciles} bins)"

    tbl = binary_decile_table(yt, ys, n_deciles=eff_deciles)
    n_rows = len(tbl["decile"])

    col_headers = [c[0] for c in _DECILE_TABLE_COLUMNS]
    cells: List[List[str]] = [[fmt(tbl[key][d]) for _, key, fmt in _DECILE_TABLE_COLUMNS] for d in range(n_rows)]
    total_pos = int(tbl["positives"].sum())
    total_n = int(tbl["count"].sum())
    total_resp = total_pos / total_n if total_n else float("nan")
    # TOTAL row: cumulative gain/KS are 100% / 0 by construction at the full population; lift is 1.0 (the baseline).
    total_row = ["TOTAL", f"{total_n:,}", f"{total_pos:,}", "-" if not np.isfinite(total_resp) else f"{total_resp:.1%}", "100.0%", "1.00", "0.000"]
    cells.append(total_row)

    fig = Figure(figsize=figsize or (8.0, 0.42 * (n_rows + 3)))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_title(title + note, fontsize=11)
    table = ax.table(cellText=cells, colLabels=col_headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    gain_col = col_headers.index("cum gain")
    gain_vals = tbl["gain"]
    header_color = "#34495e"
    highlight = "#fff3cd"
    total_color = "#d6eaf8"
    gain_shade = (0.66, 0.78, 0.91)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", fontweight="bold")
        elif row == n_rows + 1:
            cell.set_facecolor(total_color)
            cell.set_text_props(fontweight="bold")
        else:
            d = row - 1
            if col == gain_col and np.isfinite(gain_vals[d]):
                a = 0.18 + 0.55 * float(gain_vals[d])
                cell.set_facecolor((gain_shade[0], gain_shade[1], gain_shade[2], a))
            elif d < highlight_top:
                cell.set_facecolor(highlight)
            else:
                cell.set_facecolor("white")
    return fig


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

# One sentence per token, joined for the tokens ACTUALLY rendered (see ``_captions.caption_for_tokens``). The
# figure-level caption used to describe the DEFAULT template, so a caller asking for a narrower mix read about
# panels that were not on their figure.
_TOKEN_CAPTIONS: Dict[str, str] = {
    "ROC": (
        "ROC plots the true-positive rate against the false-positive rate over every threshold: it answers RANKING, not calibration, and its negative-heavy denominator flatters a model at a low base rate."
    ),
    "PR": (
        "The PR curve trades precision against recall and, unlike ROC, its baseline moves with the positive rate, which makes it the honest curve on rare positives."
    ),
    "KS": (
        "The KS statistic is the widest vertical gap between the two class score distributions; it summarises separation as one number and says nothing about where to cut."
    ),
    "GAIN": ("The gain curve reads bottom-up: how much of the positive mass you capture by contacting the top X% of the ranking."),
    "SCORE_DIST": ("The score distributions show the two classes separately; overlap here is exactly what the ranking metrics are summarising."),
    "PIT": ("The PIT histogram is flat when the predicted probabilities are calibrated; a U shape means over-confidence and a hump means under-confidence."),
    "THRESHOLD": (
        "The threshold panel answers OPERATION -- where to cut -- and a model can rank well while being badly calibrated, so a strong AUC is not permission to use the scores as probabilities."
    ),
}


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
    ap_ci: bool = True,
    ap_ci_seed: int = 0,
    operating_point: bool = True,
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
    ap_ci : annotate the PR panel's AP with a 95% bootstrap CI (default on; AP has no closed-form variance).
    ap_ci_seed : seed for the AP bootstrap, so the interval is reproducible.
    operating_point : mark the chosen ``threshold`` operating point on the ROC (FPR, TPR) and PR (recall, precision)
        curves (default on), so the user sees exactly where on the sweep they operate. Omitted gracefully when
        single-class / the threshold flags nobody.
    """
    tokens = parse_panel_template(panels_template)
    unknown = [t for t in tokens if t not in _TOKEN_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown binary panel tokens {unknown}. " f"Allowed: {sorted(ALLOWED_BINARY_PANEL_TOKENS)}")
    yt, ys = _finite_binary(y_true, y_score)
    # Single descending-score sort + cumulative counts shared by every rank-threshold panel.
    sort = _ScoreSort(yt, ys)
    panels: List[PanelSpec] = []
    for tok in tokens:
        kw = dict(sort=sort, threshold=threshold, cost_ratio=cost_ratio)
        if tok in ("ROC", "PR"):
            kw.update(operating_point=operating_point)
        if tok == "PR":
            kw.update(ap_ci=ap_ci, ap_ci_seed=ap_ci_seed)
        panels.append(_TOKEN_BUILDERS[tok](yt, ys, **kw))
    grid = pack_panels(panels, max_cols=max_cols)
    n_rows = len(grid)
    n_cols = max_cols if grid else 0
    # RUX-21 class: a constant score column still produces real-looking panels (KS 1.000, AUC 0.500) with nothing
    # saying the score never varies -- the one fact that explains every number on the figure.
    n_unique = int(np.unique(ys).size) if ys.size else 0
    degenerate_note = (
        f"  |  WARNING: the score has a single distinct value ({ys[0]:.6g}) over {ys.size:,} rows, so every "
        "ranking metric here is the degenerate no-ordering case, not a measurement of this model."
        if n_unique == 1 else ""
    )
    return FigureSpec(
        suptitle=suptitle + degenerate_note,
        panels=grid,
        figsize=figsize_for_grid(n_rows, n_cols, cell_width=cell_width, cell_height=cell_height),
        caption=caption_for_tokens(
            "How to read: each panel answers one of three separate questions -- ranking, calibration, or operation -- and a model can be strong at one while failing another.",
            tokens,
            _TOKEN_CAPTIONS,
        ),
    )


__all__ = [
    "ALLOWED_BINARY_PANEL_TOKENS",
    "DEFAULT_BINARY_PANELS",
    "compose_binary_figure",
    "bootstrap_ap_ci",
    "binary_decile_table",
    "binary_decile_table_figure",
]
