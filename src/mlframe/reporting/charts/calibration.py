"""Calibration-report chart spec builder.

Produces a 2-row FigureSpec:
- top: ScatterPanelSpec (reliability scatter + perfect-fit diagonal +
  per-bin colormap-driven population colors + inline population labels)
- bottom (optional): HistogramPanelSpec (bin populations as colored bars
  matching the scatter colormap)

Reusable by both backends; replaces the matplotlib-only renderer in
``mlframe.metrics.core::show_calibration_plot`` (which stays as a back-compat
wrapper).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from mlframe.reporting.colors import HEATMAP_CMAP
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec, ScatterPanelSpec,
)

# Cap on per-point bubble area so a single dominant bin can't occlude its
# neighbours; populations above the cap are sqrt-compressed instead of clipped
# flat so relative ordering is still visible.
MAX_BUBBLE_AREA: float = 800.0
# Inline per-bin population labels turn into unreadable soup past this many
# bins (nbins=100 is a supported value); auto-disable above it.
INLINE_LABEL_MAX_BINS: int = 25
# z for a two-sided 95% Wilson binomial interval.
_WILSON_Z_95: float = 1.959963984540054
# Cap on rows fed to the isotonic fit: the curve is read on a small grid, so a subsample of this size gives a
# visually identical map while keeping the O(n log n) PAVA fit bounded at n>=1e6.
_SMOOTHED_FIT_MAX_ROWS: int = 100_000
# Grid resolution at which the smoothed calibration map is evaluated (bin-count-independent by construction).
_SMOOTHED_GRID_POINTS: int = 100
# Below this many finite (score, label) rows the smoothed map is too noisy to be meaningful -> skip the overlay.
_SMOOTHED_MIN_ROWS: int = 50


def smoothed_reliability_curve(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    n_grid: int = _SMOOTHED_GRID_POINTS,
    max_rows: int = _SMOOTHED_FIT_MAX_ROWS,
    random_state: int = 0,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Binning-free calibration map via isotonic regression on raw ``(score, label)`` pairs.

    Returns ``(grid, calibrated)`` over a uniform ``[score_min, score_max]`` grid, or ``None`` when the input is
    degenerate (single class, all-equal scores, or fewer than ``_SMOOTHED_MIN_ROWS`` finite rows). Isotonic gives a
    monotone, bandwidth-free calibration curve whose shape does not depend on a bin count -- the property the binned
    reliability points lack. The fit (PAVA) is O(n log n); rows above ``max_rows`` are subsampled since the curve is
    only read on ``n_grid`` points, so the result is visually identical while bounding the cost at large n.
    """
    s = np.asarray(y_score, dtype=np.float64).ravel()
    t = np.asarray(y_true, dtype=np.float64).ravel()
    finite = np.isfinite(s) & np.isfinite(t)
    s, t = s[finite], t[finite]
    if s.size < _SMOOTHED_MIN_ROWS:
        return None
    classes = np.unique(t)
    if classes.size < 2:
        return None
    smin, smax = float(s.min()), float(s.max())
    if not (smax > smin):
        return None
    if s.size > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(s.size, size=max_rows, replace=False)
        s, t = s[idx], t[idx]
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s, t)
    grid = np.linspace(smin, smax, n_grid)
    return grid, np.asarray(iso.predict(grid), dtype=np.float64)


def _format_population(n: float) -> str:
    """Compact thousands/millions/billions for inline scatter labels."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return f"{n:.0f}"


def _resolve_yscale(yscale: str, hits: np.ndarray) -> str:
    """auto -> log iff max/min population skew > 100, else linear. Explicit modes pass through."""
    if yscale != "auto":
        return yscale
    if len(hits) == 0:
        return "linear"
    max_h = float(np.max(hits))
    min_h = max(float(np.min(hits)), 1.0)
    return "log" if (max_h / min_h) > 100.0 else "linear"


def wilson_ci(p_hat: np.ndarray, n: np.ndarray, z: float = _WILSON_Z_95):
    """Wilson score interval for a binomial proportion.

    Returns ``(lower, upper)`` arrays clipped to ``[0, 1]``. The Wilson interval
    is preferred over the normal-approximation (Wald) interval for the small / extreme-p
    per-bin counts a reliability diagram produces -- it never escapes [0, 1] and stays
    sensible at p_hat in {0, 1}. Bins with n == 0 yield (nan, nan).

    Center / half-width:
        denom  = 1 + z^2/n
        center = (p_hat + z^2/(2n)) / denom
        half   = z*sqrt(p_hat*(1-p_hat)/n + z^2/(4n^2)) / denom
    """
    p = np.asarray(p_hat, dtype=np.float64)
    nn = np.asarray(n, dtype=np.float64)
    lower = np.full(p.shape, np.nan)
    upper = np.full(p.shape, np.nan)
    valid = nn > 0
    if not np.any(valid):
        return lower, upper
    pv = p[valid]
    nv = nn[valid]
    z2 = z * z
    denom = 1.0 + z2 / nv
    center = (pv + z2 / (2.0 * nv)) / denom
    half = (z * np.sqrt(pv * (1.0 - pv) / nv + z2 / (4.0 * nv * nv))) / denom
    lower[valid] = np.clip(center - half, 0.0, 1.0)
    upper[valid] = np.clip(center + half, 0.0, 1.0)
    return lower, upper


def _bubble_point_size(hits: np.ndarray) -> np.ndarray:
    """Population -> bubble area with a cap so one dominant bin can't dwarf the rest.

    Below the cap the area is the legacy ``5000*h/sum(h)`` scaling; above it the area
    is sqrt-compressed toward ``MAX_BUBBLE_AREA`` so high-population bins stay the largest
    without occluding neighbours.
    """
    h = np.asarray(hits, dtype=np.float64)
    total = float(h.sum()) if h.size else 1.0
    raw = 5000.0 * h / max(total, 1.0)
    over = raw > MAX_BUBBLE_AREA
    if np.any(over):
        raw[over] = MAX_BUBBLE_AREA * np.sqrt(raw[over] / MAX_BUBBLE_AREA)
    return raw


def build_calibration_spec(
    freqs_predicted: np.ndarray,
    freqs_true: np.ndarray,
    hits: np.ndarray,
    *,
    plot_title: str = "",
    show_prob_histogram: bool = True,
    show_inline_population_labels: bool = True,
    label_freq: str = "Observed Frequency",
    label_prob: str = "Predicted Probability",
    label_histogram: str = "Bin population",
    colorbar_label: str = "Bin population",
    figsize: Tuple[float, float] = (12.0, 6.0),
    yscale: str = "auto",
    show_wilson_ci: bool = True,
    reliability_smoothed: bool = True,
    raw_probs: Optional[np.ndarray] = None,
    raw_labels: Optional[np.ndarray] = None,
) -> FigureSpec:
    """Build a calibration-report FigureSpec.

    Parameters mirror the legacy ``show_calibration_plot`` signature for
    back-compat. The histogram bottom panel uses the same colormap as
    the scatter so the colorbar reads consistently across both subplots.

    ``yscale`` controls the bottom bin-population histogram ("auto"/"log"/"linear");
    "auto" goes log only when the population max/min skew exceeds 100 so a
    heavy-tailed rare-event distribution does not hide low-population bins.

    ``show_wilson_ci`` (default on) attaches a 95% Wilson binomial CI band on
    ``freqs_true`` (computed from ``hits``) so each reliability point carries its
    uncertainty. Bubble area is capped (``MAX_BUBBLE_AREA``) so one dominant bin
    cannot occlude its neighbours, and inline labels auto-disable past
    ``INLINE_LABEL_MAX_BINS`` to avoid label soup at large ``nbins``.

    ``reliability_smoothed`` (default on) overlays a binning-free isotonic calibration curve, fit on the raw
    ``(raw_probs, raw_labels)`` pairs (subsampled to a bounded row count). Unlike the binned bubbles, its shape does
    not depend on the chosen bin count. It is additive (the binned points + Wilson CI + histogram are unchanged) and
    degrades to no overlay when the raw pairs are absent or degenerate (single class / all-equal scores / too few
    rows). The suite caller threads ``raw_probs``/``raw_labels`` through; passing only ``freqs_*`` skips the overlay.
    """
    freqs_predicted = np.asarray(freqs_predicted, dtype=np.float64)
    freqs_true = np.asarray(freqs_true, dtype=np.float64)
    hits = np.asarray(hits)

    # No bins, or no bin with any finite (pred, obs) point: the scatter would set NaN axis limits and the
    # bin-population histogram would reduce over an empty array. Emit an honest placeholder instead.
    finite_bin = np.isfinite(freqs_predicted) & np.isfinite(freqs_true)
    if freqs_predicted.size == 0 or not finite_bin.any():
        ann = AnnotationPanelSpec(text=(plot_title + "\n" if plot_title else "") + "calibration unavailable: no finite bins",
                                  title=plot_title or "Calibration")
        return FigureSpec(suptitle="", panels=((ann,),), figsize=figsize)

    if len(freqs_predicted) > 1:
        bar_width = float(np.mean(np.diff(np.sort(freqs_predicted))))
    else:
        bar_width = 0.05

    inline_labels: Optional[Tuple[Tuple[float, float, str], ...]] = None
    if (
        show_inline_population_labels
        and len(hits) > 0
        and len(freqs_predicted) <= INLINE_LABEL_MAX_BINS
    ):
        inline_labels = tuple(
            (float(x), float(y), _format_population(float(h)))
            for x, y, h in zip(freqs_predicted, freqs_true, hits)
        )

    point_size = _bubble_point_size(hits)

    overlay_line: Optional[Tuple[np.ndarray, np.ndarray, str]] = None
    if reliability_smoothed and raw_probs is not None and raw_labels is not None:
        curve = smoothed_reliability_curve(raw_probs, raw_labels)
        if curve is not None:
            overlay_line = (curve[0], curve[1], "smoothed (isotonic)")

    # Wilson CI band on the observed frequency per bin: ``freqs_true`` is the per-bin positive rate, ``hits`` its
    # count, so the binomial interval reflects sampling uncertainty (wide where a bin holds few points). The spec
    # carries asymmetric error bars as (lower_distance, upper_distance) from the point; nan-CI bins (n==0) -> 0 dist.
    y_err: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if show_wilson_ci and len(hits) > 0:
        lower, upper = wilson_ci(freqs_true, hits.astype(np.float64))
        lo_dist = np.where(np.isfinite(lower), freqs_true - lower, 0.0)
        hi_dist = np.where(np.isfinite(upper), upper - freqs_true, 0.0)
        y_err = (np.clip(lo_dist, 0.0, None), np.clip(hi_dist, 0.0, None))

    scatter = ScatterPanelSpec(
        x=freqs_predicted,
        y=freqs_true,
        title=plot_title,
        xlabel=label_prob if not show_prob_histogram else "",
        ylabel=label_freq,
        perfect_fit_line=True,
        point_color=hits.astype(np.float64),
        colormap=HEATMAP_CMAP,
        point_alpha=0.7,
        point_size=point_size,
        inline_labels=inline_labels,
        colorbar_label=colorbar_label,
        y_err=y_err,
        overlay_line=overlay_line,
    )

    resolved_yscale = _resolve_yscale(yscale, hits)
    if not show_prob_histogram:
        return FigureSpec(
            suptitle="",
            panels=((scatter,),),
            figsize=figsize,
        )

    hist = HistogramPanelSpec(
        values=hits,            # heights = bin populations
        bin_centers=freqs_predicted,
        bin_width=bar_width,
        bar_colors=hits.astype(np.float64),
        colormap=HEATMAP_CMAP,
        title="",
        xlabel=label_prob,
        ylabel=label_histogram,
        yscale=resolved_yscale if resolved_yscale in ("linear", "log") else "linear",
        density=False,
    )

    return FigureSpec(
        suptitle="",
        panels=((scatter,), (hist,)),
        figsize=figsize,
        row_height_ratios=(3.0, 1.0),
        sharex=True,
    )


def _reliability_curve(p: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Observed positive rate per uniform bin (nan for empty bins). Vectorized via bincount."""
    nbins = len(edges) - 1
    bin_ids = np.clip(np.digitize(p, edges, right=False) - 1, 0, nbins - 1)
    counts = np.bincount(bin_ids, minlength=nbins).astype(np.float64)
    sums = np.bincount(bin_ids, weights=y, minlength=nbins)
    out = np.full(nbins, np.nan)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out


def build_reliability_overlay_spec(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    *,
    calibrated_probs: Optional[dict] = None,
    series_labels: Optional[dict] = None,
    n_bins: int = 15,
    title: str = "Reliability diagram (OOF)",
    figsize: Tuple[float, float] = (6.0, 5.0),
) -> FigureSpec:
    """Overlay reliability curves (perfect diagonal + raw + each calibrated candidate).

    Single source for the multi-curve reliability diagram (calibration-policy OOF plot).
    All curves share the uniform-bin centre x-grid; each series y is the observed positive
    rate per bin. Empty bins are NaN (the renderer skips them). Returns a one-panel FigureSpec.
    """
    raw_probs = np.asarray(raw_probs, dtype=np.float64).ravel()
    y = np.asarray(y_true, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    series = [centers.copy()]  # perfect diagonal as the first series
    labels = ["perfect"]
    styles = [":"]
    series.append(_reliability_curve(raw_probs, y, edges))
    labels.append("raw OOF")
    styles.append("lines+markers")

    calibrated_probs = calibrated_probs or {}
    series_labels = series_labels or {}
    for name, cal_p in calibrated_probs.items():
        series.append(_reliability_curve(np.asarray(cal_p, dtype=np.float64).ravel(), y, edges))
        labels.append(series_labels.get(name, str(name)))
        styles.append("lines+markers")

    line = LinePanelSpec(
        x=centers,
        y=tuple(series),
        series_labels=tuple(labels),
        line_styles=tuple(styles),
        title=title,
        xlabel="predicted probability",
        ylabel="empirical frequency",
    )
    return FigureSpec(suptitle="", panels=((line,),), figsize=figsize)


def _midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x`` (tied values share the mean of the ranks they would occupy). O(n log n) via one argsort.

    Midranks are the core primitive of DeLong's fast AUC-variance algorithm: AUC equals a function of the midranks of
    the pooled, the positive-only, and the negative-only score samples, so the variance follows in closed form.

    Fully vectorised: after sorting, each tie block spans ``[start, end)`` of equal values and gets the 1-indexed mean
    rank ``0.5*(start + end - 1) + 1``. ``searchsorted`` on the sorted values gives every value's left/right tie bound
    in two O(n log n) passes (no python per-element loop -- the loop form was ~93% of DeLong's wall at n=1e6).
    """
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="quicksort")
    sorted_x = x[order]
    # For each sorted position, the inclusive tie block is [left, right): searchsorted finds the first equal and the
    # first strictly-greater value. The mean 0-based position is 0.5*(left + right - 1); +1 makes it a 1-indexed rank.
    left = np.searchsorted(sorted_x, sorted_x, side="left")
    right = np.searchsorted(sorted_x, sorted_x, side="right")
    sorted_ranks = 0.5 * (left + right - 1) + 1.0
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def delong_auc_variance(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """ROC-AUC and its DeLong variance (Sun & Xu 2014 fast O(n log n) form), single binary problem.

    Returns ``(auc, var)``. ``y_true`` is binary {0,1}; ``y_score`` higher = more positive. The DeLong estimator is
    the standard closed-form AUC-variance: no bootstrap needed. ``var`` is NaN when either class is empty (AUC and
    its variance are undefined). Ties are handled via midranks, so tied scores give the correct (smaller) variance.
    """
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    pos = ys[yt == 1]
    neg = ys[yt == 0]
    m = pos.size
    n = neg.size
    if m == 0 or n == 0:
        return float("nan"), float("nan")
    # Sun-Xu midrank decomposition: structural components V10 (per positive) and V01 (per negative).
    tz = _midrank(np.concatenate([pos, neg]))
    tx = _midrank(pos)
    ty = _midrank(neg)
    # AUC = (sum of positive midranks in the pooled sample - m(m+1)/2) / (m n) -- the Mann-Whitney form via midranks.
    auc = (tz[:m].sum() - m * (m + 1.0) / 2.0) / (m * n)
    v10 = (tz[:m] - tx) / n
    v01 = 1.0 - (tz[m:] - ty) / m
    s10 = float(np.var(v10, ddof=1)) if m > 1 else 0.0
    s01 = float(np.var(v01, ddof=1)) if n > 1 else 0.0
    var = s10 / m + s01 / n
    return float(auc), float(var)


def delong_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """ROC-AUC with a two-sided ``1-alpha`` DeLong confidence interval, clipped to [0, 1].

    Returns ``(auc, lower, upper)``. The CI is the normal-approximation interval ``auc +- z * sqrt(var)`` on the AUC
    scale (the simple, widely-reported DeLong CI). It narrows as ``~1/sqrt(n)``, so a large-n problem gives a tight
    bracket around the true AUC -- the property the biz_value test pins. ``(nan, nan, nan)`` when a class is empty.
    """
    auc, var = delong_auc_variance(y_true, y_score)
    if not np.isfinite(auc) or not np.isfinite(var):
        return auc, float("nan"), float("nan")
    from scipy.stats import norm

    z = float(norm.ppf(1.0 - alpha / 2.0))
    half = z * float(np.sqrt(max(var, 0.0)))
    return auc, float(np.clip(auc - half, 0.0, 1.0)), float(np.clip(auc + half, 0.0, 1.0))


__all__ = [
    "build_calibration_spec",
    "build_reliability_overlay_spec",
    "smoothed_reliability_curve",
    "wilson_ci",
    "delong_auc_variance",
    "delong_auc_ci",
    "MAX_BUBBLE_AREA",
    "INLINE_LABEL_MAX_BINS",
]
