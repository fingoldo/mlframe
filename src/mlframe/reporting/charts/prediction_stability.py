"""Ensemble prediction-stability charts: see WHERE ensemble members DISAGREE.

mlframe is ensemble-heavy (bagged folds, multi-seed, stacking) yet had no view of per-row epistemic uncertainty -- the
disagreement between members. High per-row spread across members flags rows the ensemble is unsure about; if that
spread tracks actual error, it is an informative uncertainty estimate (the high-value validation here).

Builders take an ``(n_rows, n_members)`` matrix of per-member predictions (regression) or class-1 probabilities
(binary) and return ``FigureSpec`` / panel specs. Nothing imports training internals; the integrator feeds the matrix.

Panels:

* spread distribution -- histogram of per-row member-spread (where most of the disagreement mass sits).
* spread vs ensemble-mean -- subsampled scatter showing whether uncertainty concentrates at certain prediction ranges.
* uncertainty calibration -- mean |error| per spread-bin (does higher disagreement => higher actual error?). The
  monotone-increasing shape + a positive Spearman(spread, |error|) is what makes the spread trustworthy.

All per-row reductions are vectorized over ``axis=1`` (O(n*M), no Python loops); scatters subsample to <=5000 rows.
M<2 (no spread) and degenerate (all-NaN / tiny n) inputs return an honest ``AnnotationPanelSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mlframe.reporting.charts._layout import figsize_for_grid, pack_panels
from mlframe.reporting.charts._sampling import prebin_histogram, subsample_for_density
from mlframe.reporting.spec import (
    AnnotationPanelSpec, FigureSpec, HistogramPanelSpec, LinePanelSpec, PanelSpec, ScatterPanelSpec,
)

# Bins for the spread histogram; >~50 turns the density into a noisy comb at the row counts we see.
DEFAULT_SPREAD_BINS: int = 40
# Equal-frequency spread bins for the uncertainty-calibration curve; few enough that each bin keeps a stable mean |error|.
DEFAULT_CALIB_BINS: int = 10
# Below this many finite rows the per-spread-bin mean |error| is too noisy to read; the calibration panel annotates instead.
MIN_CALIB_ROWS: int = 50
# Scatter subsample cap (matches the density-panel convention) so the spread-vs-mean panel stays readable at millions of rows.
DEFAULT_SCATTER_CAP: int = 5_000


@dataclass(frozen=True)
class PredictionStabilityResult:
    """Per-row ensemble-disagreement summary.

    ``spread_std`` / ``spread_iqr`` are per-row dispersion across members; ``ensemble_mean`` is the per-row mean
    prediction. ``mean_spread`` / ``agreement`` summarise the whole set (agreement = fraction of rows whose std is
    below ``low_spread_threshold``). ``n_members`` < 2 means spread is undefined (the arrays are zero-filled).
    """

    spread_std: np.ndarray
    spread_iqr: np.ndarray
    ensemble_mean: np.ndarray
    mean_spread: float
    agreement: float
    n_rows: int
    n_members: int


def _as_member_matrix(member_preds: np.ndarray) -> np.ndarray:
    """Coerce input to a 2-D float ``(n_rows, n_members)`` matrix; a 1-D array is treated as a single member (M=1)."""
    mat = np.asarray(member_preds, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    return mat


def compute_prediction_stability(
    member_preds: np.ndarray,
    *,
    low_spread_threshold: Optional[float] = None,
) -> PredictionStabilityResult:
    """Per-row member-disagreement: std + IQR across members, the ensemble mean, and an agreement summary.

    ``member_preds`` is ``(n_rows, n_members)``. Reductions are vectorized over ``axis=1`` (O(n*M), no Python loops);
    NaN members are ignored per row via ``nanstd`` / ``nanmean`` / ``nanpercentile``. With M<2 spread is undefined, so
    the dispersion arrays are zero-filled and ``n_members`` records the degeneracy for the composer to annotate.
    ``low_spread_threshold`` defaults to the median per-row std -- ``agreement`` is then the fraction of rows below it.
    """
    mat = _as_member_matrix(member_preds)
    n_rows, n_members = mat.shape

    if n_members < 2 or n_rows == 0:
        zeros = np.zeros(n_rows, dtype=np.float64)
        emean = np.nanmean(mat, axis=1) if n_rows and n_members else zeros
        return PredictionStabilityResult(zeros, zeros.copy(), np.asarray(emean, dtype=np.float64),
                                         0.0, 1.0, int(n_rows), int(n_members))

    has_nan = bool(np.isnan(mat).any())
    with np.errstate(invalid="ignore"):
        if has_nan:
            spread_std = np.nanstd(mat, axis=1)
            ensemble_mean = np.nanmean(mat, axis=1)
            q75, q25 = np.nanpercentile(mat, [75.0, 25.0], axis=1)
        else:
            # Vectorized path: np.percentile(axis=1) stays in C; nanpercentile falls back to a per-row apply_along_axis
            # loop (~90s at 1e6x10), so we only pay that cost when NaNs are actually present.
            spread_std = mat.std(axis=1)
            ensemble_mean = mat.mean(axis=1)
            q75, q25 = np.percentile(mat, [75.0, 25.0], axis=1)
        spread_iqr = q75 - q25

    spread_std = np.nan_to_num(spread_std, nan=0.0)
    spread_iqr = np.nan_to_num(spread_iqr, nan=0.0)

    finite = spread_std[np.isfinite(spread_std)]
    mean_spread = float(finite.mean()) if finite.size else 0.0
    thr = float(np.median(finite)) if low_spread_threshold is None and finite.size else (low_spread_threshold or 0.0)
    agreement = float(np.mean(spread_std <= thr)) if spread_std.size else 1.0

    return PredictionStabilityResult(
        spread_std, spread_iqr, np.asarray(ensemble_mean, dtype=np.float64),
        mean_spread, agreement, int(n_rows), int(n_members),
    )


def _spread_histogram_panel(spread: np.ndarray, *, bins: int) -> PanelSpec:
    """Pre-binned histogram of per-row member-spread (where the disagreement mass concentrates)."""
    heights, centers, width = prebin_histogram(spread, bins=bins, density=True)
    if heights is None:
        return AnnotationPanelSpec(text="Spread distribution\n(no finite spread)", title="Member-spread distribution")
    return HistogramPanelSpec(
        values=heights, bin_centers=centers, bin_width=width, density=True,
        title="Per-row member-spread distribution",
        xlabel="member std", ylabel="Density",
    )


def _spread_vs_mean_panel(
    spread: np.ndarray,
    ensemble_mean: np.ndarray,
    *,
    cap: int,
    seed: int,
) -> PanelSpec:
    """Subsampled scatter of per-row spread vs ensemble-mean prediction (does uncertainty concentrate at some ranges?)."""
    finite = np.isfinite(spread) & np.isfinite(ensemble_mean)
    x = ensemble_mean[finite]
    y = spread[finite]
    if x.size == 0:
        return AnnotationPanelSpec(text="Spread vs mean\n(no finite rows)", title="Spread vs ensemble mean")
    if x.size > cap:
        idx = subsample_for_density(np.arange(x.size), cap=cap, seed=seed)
        x, y = x[idx], y[idx]
    return ScatterPanelSpec(
        x=x, y=y, title=f"Member-spread vs ensemble mean ({x.size:_} pts)",
        xlabel="ensemble mean prediction", ylabel="member std", point_alpha=0.25,
    )


def _uncertainty_calibration(
    spread: np.ndarray,
    abs_error: np.ndarray,
    *,
    nbins: int,
):
    """Bin rows by per-row spread (equal-frequency) and return (bin_mid_spread, mean_abs_error, spearman).

    The high-value validation: if higher member-disagreement tracks higher actual error, the mean |error| rises
    monotonically across spread bins and ``Spearman(spread, |error|)`` is positive -- the spread is INFORMATIVE.
    Binning is via ``np.quantile`` edges + ``np.digitize`` + ``bincount`` (O(n)); the Spearman is on the full vector.
    """
    finite = np.isfinite(spread) & np.isfinite(abs_error)
    s = spread[finite]
    e = abs_error[finite]
    if s.size < MIN_CALIB_ROWS or float(np.ptp(s)) == 0.0:
        return None, None, float("nan")

    edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, nbins + 1)))
    if edges.size < 2:
        return None, None, float("nan")
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
    nb = len(edges) - 1
    counts = np.bincount(idx, minlength=nb).astype(np.float64)
    err_sum = np.bincount(idx, weights=e, minlength=nb)
    spr_sum = np.bincount(idx, weights=s, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_err = np.where(counts > 0, err_sum / np.where(counts > 0, counts, 1.0), np.nan)
        mid_spread = np.where(counts > 0, spr_sum / np.where(counts > 0, counts, 1.0), np.nan)

    spearman = _spearman(s, e)
    keep = counts > 0
    return mid_spread[keep], mean_err[keep], spearman


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via average-tied ranks + Pearson on the ranks; O(n log n), no scipy dependency."""
    if a.size < 2:
        return float("nan")
    ra = _rankdata(a)
    rb = _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(ra * rb) / denom)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their rank span), matching scipy.stats.rankdata's default."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    ranks[order] = np.arange(1, x.size + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    n = x.size
    while i < n:
        j = i + 1
        while j < n and sx[j] == sx[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _uncertainty_calibration_panel(
    spread: np.ndarray,
    abs_error: np.ndarray,
    *,
    nbins: int,
) -> PanelSpec:
    """Line panel: mean |error| per spread bin. A rising curve + positive Spearman => the spread is an informative uncertainty estimate."""
    mid, mean_err, rho = _uncertainty_calibration(spread, abs_error, nbins=nbins)
    if mid is None:
        return AnnotationPanelSpec(
            text="Uncertainty calibration\n(needs varying spread + y_true)",
            title="Spread vs actual error",
        )
    return LinePanelSpec(
        x=mid, y=mean_err, series_labels=("mean |error| per spread bin",),
        line_styles=("lines+markers",),
        title=f"Uncertainty calibration: error vs disagreement\nSpearman(spread, |error|) = {rho:.3f}",
        xlabel="per-row member std (bin mean)", ylabel="mean |error|",
    )


def compose_prediction_stability_figure(
    member_preds: np.ndarray,
    *,
    y_true: Optional[np.ndarray] = None,
    spread_bins: int = DEFAULT_SPREAD_BINS,
    calib_bins: int = DEFAULT_CALIB_BINS,
    scatter_cap: int = DEFAULT_SCATTER_CAP,
    suptitle: str = "Ensemble prediction stability (member disagreement)",
    seed: int = 0,
) -> FigureSpec:
    """Compose the prediction-stability panel grid from an ``(n_rows, n_members)`` member matrix.

    Panels: (a) per-row member-spread histogram; (b) spread vs ensemble-mean subsampled scatter; (c) -- only when
    ``y_true`` is given -- the uncertainty-calibration curve (mean |error| per spread bin) that validates whether the
    disagreement is informative. With M<2 the figure is a single honest annotation (spread is undefined for one member).
    """
    res = compute_prediction_stability(member_preds)

    if res.n_members < 2:
        ann = AnnotationPanelSpec(
            text=f"{suptitle}\n\nNeed >=2 ensemble members to measure disagreement "
                 f"(got n_members={res.n_members}).\nNo per-row spread is defined for a single member.",
            title=suptitle,
        )
        return FigureSpec(suptitle=suptitle, panels=((ann,),), figsize=(8.0, 3.5))

    panels = [
        _spread_histogram_panel(res.spread_std, bins=spread_bins),
        _spread_vs_mean_panel(res.spread_std, res.ensemble_mean, cap=scatter_cap, seed=seed),
    ]
    if y_true is not None:
        yt = np.asarray(y_true, dtype=np.float64).ravel()
        abs_error = np.abs(yt - res.ensemble_mean)
        panels.append(_uncertainty_calibration_panel(res.spread_std, abs_error, nbins=calib_bins))

    grid = pack_panels(panels, max_cols=2)
    n_rows = len(grid)
    return FigureSpec(
        suptitle=suptitle + f"  |  mean spread={res.mean_spread:.3g}, agreement={res.agreement:.2f}",
        panels=grid,
        figsize=figsize_for_grid(max(n_rows, 1), 2, cell_width=6.5, cell_height=4.5),
    )


__all__ = [
    "PredictionStabilityResult",
    "compute_prediction_stability",
    "compose_prediction_stability_figure",
    "DEFAULT_SPREAD_BINS",
    "DEFAULT_CALIB_BINS",
    "DEFAULT_SCATTER_CAP",
]
