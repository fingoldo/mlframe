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

from mlframe.reporting.colors import CALIBRATION
from mlframe.reporting.spec import (
    FigureSpec, HistogramPanelSpec, LinePanelSpec, ScatterPanelSpec,
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
    """
    freqs_predicted = np.asarray(freqs_predicted, dtype=np.float64)
    freqs_true = np.asarray(freqs_true, dtype=np.float64)
    hits = np.asarray(hits)

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
        colormap=CALIBRATION,
        point_alpha=0.7,
        point_size=point_size,
        inline_labels=inline_labels,
        colorbar_label=colorbar_label,
        y_err=y_err,
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
        colormap=CALIBRATION,
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


__all__ = [
    "build_calibration_spec",
    "build_reliability_overlay_spec",
    "wilson_ci",
    "MAX_BUBBLE_AREA",
    "INLINE_LABEL_MAX_BINS",
]
