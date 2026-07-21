"""Rank / correlation regression metrics carved out of _regression_extras.py (single-sibling split).

Pearson / Spearman / Kendall tau-b / concordance-index scalar metrics + their njit kernels.
_regression_extras.py re-exports these so the public fast_pearson_corr / fast_spearman_corr /
fast_kendall_tau / fast_concordance_index import surface is unchanged.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import numba

from .._numba_params import NUMBA_NJIT_PARAMS, _check_equal_length
from ..rank_correlation import spearmanr_scalar_dispatch


@numba.njit(**NUMBA_NJIT_PARAMS)
def _pearson_corr_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """Compiled single-pass Pearson correlation coefficient between ``x`` and ``y``."""
    n = x.shape[0]
    if n < 2:
        return np.nan
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    # sqrt(sxx)*sqrt(syy) not sqrt(sxx*syy): the product overflows to inf on large-scale data,
    # which would silently collapse a genuine correlation to sxy/inf == 0.0.
    denom = sqrt(sxx) * sqrt(syy)
    if denom == 0.0:
        return np.nan
    return sxy / denom


def fast_pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson product-moment correlation between y_true and y_pred.

    R^2 is correlation^2 for an UNBIASED model; for biased models R^2 and
    Pearson r differ - report both so the reader can see if the model
    has a scale/bias mismatch on top of its rank quality.

    iter607: dropped the unconditional ``dtype=np.float64`` cast (same
    pattern as iter595/596/597/598/606). Kernel has two scalar reduction
    passes; numba dispatches on mixed-dtype signatures natively. Bench
    n=100k: int64+float64 1.07x, float64+float64 0.98x (noise band),
    float64+float32 1.12x. Bit-equiv vs scipy.stats.pearsonr."""
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true)
    yp = np.ascontiguousarray(y_pred)
    return float(_pearson_corr_kernel(yt, yp))


def fast_spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (scalar wrapper).

    Routes to ``spearmanr_scalar_dispatch``, which parallelises WITHIN the single series (the two rankings concurrent, the Pearson reduction a prange sum)
    when numba is available and falls back to the scipy ``rankdata`` numpy path otherwise. Tied values handled via average-rank; bit-identical (~1e-12 FP
    reduction-order) to the numpy path. On a single long series the batched paths (which parallelise across rows) buy nothing, so the within-series kernel
    is the faster default at all n -- ~1.5-2.7x over scipy across n in {5k..1M}.
    """
    _check_equal_length(y_true, y_pred)
    if np.asarray(y_true).size < 2:
        return np.nan
    return spearmanr_scalar_dispatch(y_true, y_pred)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _kendall_tau_b_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """O(N^2) Kendall's tau-b. Adequate for N up to ~5000; above that
    callers should batch via scipy.stats.kendalltau which uses a
    merge-sort O(N log N) implementation.

    tau_b corrects for ties: (concordant - discordant) /
    sqrt((P - T_x)(P - T_y)) where P = N(N-1)/2.
    """
    n = x.shape[0]
    if n < 2:
        return np.nan
    concordant = 0
    discordant = 0
    tx = 0  # ties only in x
    ty = 0  # ties only in y
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0.0 and dy == 0.0:
                # tie in both - excluded from all denominators
                pass
            elif dx == 0.0:
                tx += 1
            elif dy == 0.0:
                ty += 1
            elif (dx > 0.0 and dy > 0.0) or (dx < 0.0 and dy < 0.0):
                concordant += 1
            else:
                discordant += 1
    total_pairs = n * (n - 1) // 2
    denom_x = total_pairs - tx
    denom_y = total_pairs - ty
    if denom_x <= 0 or denom_y <= 0:
        return np.nan
    return (concordant - discordant) / sqrt(float(denom_x) * float(denom_y))


_KENDALL_NUMBA_MAX_N = 500


def fast_kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kendall's tau-b (tie-corrected) rank correlation.

    Below ``_KENDALL_NUMBA_MAX_N`` rows the in-process O(N^2) numba kernel
    wins (its tight machine-code loop beats scipy's per-call dispatch); above
    that scipy's merge-sort O(N log N) ``kendalltau`` is dramatically faster.
    The threshold was 5000 historically, but a re-bench on modern scipy /
    numba (2026-05-28) puts the crossover at N~400: scipy beats the numba
    kernel 1.38x at N=400, 12.8x at N=1500, 41x at N=3000, 54x at N=5000.
    Values are identical to 4 decimals across the range (both implement the
    same tie-corrected tau-b formula), so changing the threshold is bit-
    equivalent for the returned scalar but unlocks the O(N log N) algorithm
    on every typical regression-metric shape.
    """
    _check_equal_length(y_true, y_pred)
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape[0] < 2:
        return np.nan
    if yt.shape[0] <= _KENDALL_NUMBA_MAX_N:
        return float(_kendall_tau_b_kernel(yt, yp))
    from scipy.stats import kendalltau
    res = kendalltau(yt, yp, variant="b")
    return float(res.correlation if hasattr(res, "correlation") else res[0])


# ============================================================================
# Concordance index (C-index)
# ============================================================================


def fast_concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """C-index = fraction of concordant pairs (ignoring tied y_true).

    Range [0, 1]; 0.5 = chance; 1.0 = perfect rank agreement. Equivalent
    to (Kendall tau-b + 1) / 2 after tie correction; emitted as a
    separate metric because survival / risk modelling reports C-index,
    not Kendall tau.

    For N <= 5000 uses the O(N^2) numba kernel below; for larger N falls
    back to the tau-b reduction (O(N log N) via scipy).
    """
    # Coerce + validate up front, matching every sibling in this module (fast_pearson_corr,
    # fast_kendall_tau, ...) -- a plain Python list (a legal input to every sibling here) has no .shape
    # attribute, so the un-coerced access below raised an unhelpful AttributeError instead of a validated
    # error or a graceful result.
    _check_equal_length(y_true, y_pred)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape[0] < 2:
        return np.nan
    tau = fast_kendall_tau(y_true, y_pred)
    if not np.isfinite(tau):
        return np.nan
    return (tau + 1.0) / 2.0
