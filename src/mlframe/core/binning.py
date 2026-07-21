"""Binning-smoothing: replace each value by its bin's representative (PZAD datapreprocessing).

The data-preprocessing lecture (Дьяконов 2020, slides 55-62) covers discretization and the slide-62 exercise
«написать кодировщик бинов»: after binning a numeric feature, encode each value by a bin REPRESENTATIVE rather
than the raw value — the classic Han & Kamber "smoothing by bin means / bin medians / bin boundaries" noise-
reduction step. This keeps the feature on its original scale (unlike `sklearn.KBinsDiscretizer`, which returns
the bin INDEX or a one-hot / kmeans-ordinal code) while collapsing within-bin noise.

- ``fit_bin_smoother`` learns the bin edges + per-bin representatives on the training data (so it can be applied
  to held-out data without leakage — the lecture's repeated "предобработка + классификация — общий пайплайн, не
  допускать утечки" caution).
- ``apply_bin_smoother`` maps new values through fitted edges/representatives.
- ``bin_smooth`` is the in-sample convenience (fit + transform on the same array).

Strategies: ``mean`` / ``median`` (bin central value) or ``boundary`` (nearer of the bin's two edges, Han & Kamber
"smoothing by bin boundaries"). Binning: ``quantile`` (equal-depth) or ``uniform`` (equal-width). NaNs pass through.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["fit_bin_smoother", "apply_bin_smoother", "bin_smooth", "BIN_STRATEGIES", "BIN_METHODS"]

BIN_STRATEGIES = ("mean", "median", "boundary")
BIN_METHODS = ("quantile", "uniform")


def fit_bin_smoother(x: np.ndarray, *, n_bins: int = 10, binning: str = "quantile") -> dict:
    """Learn bin edges and per-bin mean/median representatives from ``x``. Returns a picklable dict smoother.

    Parameters
    ----------
    x : np.ndarray
        Numeric values (NaNs ignored for fitting).
    n_bins : int
        Target number of bins (fewer result when quantile edges collide on ties).
    binning : {'quantile', 'uniform'}
        Equal-depth (quantile) or equal-width (uniform) partitioning.
    """
    if binning not in BIN_METHODS:
        raise ValueError(f"fit_bin_smoother: binning must be one of {BIN_METHODS}, got {binning!r}.")
    if n_bins < 1:
        raise ValueError(f"fit_bin_smoother: n_bins must be >= 1, got {n_bins}.")
    xf = np.ascontiguousarray(x, dtype=np.float64)
    finite = xf[np.isfinite(xf)]
    if finite.size == 0:
        raise ValueError("fit_bin_smoother: x has no finite values.")

    if binning == "quantile":
        edges = np.quantile(finite, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        lo, hi = float(finite.min()), float(finite.max())
        edges = np.linspace(lo, hi, n_bins + 1) if hi > lo else np.array([lo, hi + 1.0])
    edges = np.unique(edges)  # collapse duplicate quantiles / degenerate ranges
    if len(edges) < 2:
        # Every finite value collapsed to a single quantile/uniform edge (a constant / zero-variance
        # column -- not an exotic corner case, a realistic one: a zero-variance feature, an all-same-
        # category-coded numeric column, a post-filter column constant within a subset). Pre-fix this left
        # a smoother with zero-length bin_mean/bin_median and an empty interior, which apply_bin_smoother
        # then crashed on (np.clip(..., 0, -1) clips every index to -1, and reps[-1] on a zero-length array
        # raises IndexError). A single degenerate
        # bin spanning the constant value (mean == median == that value) is a more useful smoother than
        # raising, and apply_bin_smoother's existing digitize/clip logic already handles a 1-bin smoother
        # correctly with no further changes there.
        v = float(edges[0])
        return {
            "edges": np.array([v, v]), "interior": np.array([], dtype=np.float64),
            "bin_mean": np.array([v]), "bin_median": np.array([v]),
        }
    interior = edges[1:-1]  # cut points fed to np.digitize
    n_actual = len(edges) - 1

    assign = np.digitize(finite, interior)  # 0 .. n_actual-1
    bin_mean = np.empty(n_actual, dtype=np.float64)
    bin_median = np.empty(n_actual, dtype=np.float64)
    for b in range(n_actual):
        members = finite[assign == b]
        if members.size:
            bin_mean[b] = members.mean()
            bin_median[b] = np.median(members)
        else:  # empty bin (possible with uniform edges on skewed data): fall back to the bin midpoint
            bin_mean[b] = bin_median[b] = 0.5 * (edges[b] + edges[b + 1])
    return {"edges": edges, "interior": interior, "bin_mean": bin_mean, "bin_median": bin_median}


def apply_bin_smoother(x: np.ndarray, smoother: dict, *, strategy: str = "mean") -> np.ndarray:
    """Map ``x`` through a fitted ``smoother``, replacing each value by its bin representative. NaNs pass through unchanged."""
    if strategy not in BIN_STRATEGIES:
        raise ValueError(f"apply_bin_smoother: strategy must be one of {BIN_STRATEGIES}, got {strategy!r}.")
    xf = np.ascontiguousarray(x, dtype=np.float64)
    edges = smoother["edges"]
    interior = smoother["interior"]
    n_bins = len(edges) - 1
    assign = np.clip(np.digitize(xf, interior), 0, n_bins - 1)
    out = xf.copy()
    mask = np.isfinite(xf)

    if strategy == "boundary":
        lo = edges[assign]
        hi = edges[assign + 1]
        nearer = np.where(np.abs(xf - lo) <= np.abs(hi - xf), lo, hi)
        out[mask] = nearer[mask]
    else:
        reps = smoother["bin_mean"] if strategy == "mean" else smoother["bin_median"]
        out[mask] = reps[assign[mask]]
    return np.asarray(out)


def bin_smooth(x: np.ndarray, *, n_bins: int = 10, strategy: str = "mean", binning: str = "quantile") -> np.ndarray:
    """In-sample binning-smoothing: fit on ``x`` and return the smoothed array (each value -> its bin representative)."""
    sm = fit_bin_smoother(x, n_bins=n_bins, binning=binning)
    return apply_bin_smoother(x, sm, strategy=strategy)
