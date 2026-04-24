"""Numerically-stable variants of the moment / aggregate kernels in
``feature_engineering.numerical``.

Implements:
- ``welford_mean_var_seq`` — single-pass Welford for mean + variance
- ``welford_moments_seq`` — single-pass Welford for mean, variance, skewness, kurtosis
  (uses Bennett 2009 / Pébay generalised online algorithm with M2/M3/M4
  central-moment accumulators)
- ``kahan_sum_seq`` — Kahan-Babuška-Neumaier compensated sum
- ``kahan_dot_seq`` — Kahan-compensated dot-product (for slope/correlation
  numerator)

Used by the benchmarks in ``test_numerical_stability_bench.py`` to quantify
the precision improvement vs the naive accumulators currently used in
``numerical.compute_numerical_aggregates_numba`` and
``numerical.compute_moments_slope_mi``.

References:
- Welford 1962 — single-pass mean+var
- Pébay 2008 — generalised central-moment online formulae (used for skew/kurt)
- Neumaier 1974 — improved Kahan compensated summation
"""
from __future__ import annotations

import numpy as np
import numba

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)


# ============================================================================
# Welford — single-pass mean + variance (numerically stable)
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def welford_mean_var_seq(arr: np.ndarray) -> tuple:
    """Single-pass mean + variance via Welford.

    Returns (mean, var, n). Variance is biased (divide by n, not n-1) to
    match ``np.var(arr, ddof=0)``.

    Numerical advantage vs naive ``sum_x2/n - (sum_x/n)**2``: each
    ``delta = x - mean`` operates on differences of similar magnitude to
    the variance itself, so no catastrophic cancellation when
    `mean^2 >> var`.
    """
    n = 0
    mean = 0.0
    M2 = 0.0
    for x in arr:
        if not np.isfinite(x):
            continue
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean  # mean already updated
        M2 += delta * delta2
    if n > 0:
        var = M2 / n
    else:
        var = 0.0
    return mean, var, n


# ============================================================================
# Welford — single-pass mean + variance + skewness + kurtosis (Pébay)
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def welford_moments_seq(arr: np.ndarray) -> tuple:
    """Single-pass mean + variance + skewness + excess-kurtosis via Pébay.

    Returns (mean, var, skew, kurt, n) where:
    - var is biased (ddof=0)
    - skew is the 'biased' g1 = m3 / m2^(3/2)
    - kurt is the 'biased' g2 = m4 / m2^2 - 3 (excess kurtosis)

    Online central moment update (Pébay 2008, Eq. 1.1):
        delta = x - mean_old
        delta_n = delta / n_new
        term1 = delta * delta_n * (n - 1)
        mean += delta_n
        M4 += term1 * delta_n^2 * (n^2 - 3*n + 3) + 6 * delta_n^2 * M2 - 4 * delta_n * M3
        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
        M2 += term1
    """
    n = 0
    mean = 0.0
    M2 = 0.0
    M3 = 0.0
    M4 = 0.0
    for x in arr:
        if not np.isfinite(x):
            continue
        n_old = n
        n += 1
        delta = x - mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n_old
        mean += delta_n
        M4 += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M2 - 4 * delta_n * M3
        M3 += term1 * delta_n * (n - 2) - 3 * delta_n * M2
        M2 += term1
    if n < 2:
        return mean, 0.0, 0.0, 0.0, n
    var = M2 / n
    if M2 == 0.0:
        return mean, var, 0.0, 0.0, n
    # Skewness g1 = sqrt(n) * M3 / M2^(3/2) (biased moment-form)
    # Pébay's 'biased' skew: M3/n / (var^1.5)
    skew = (M3 / n) / (var ** 1.5)
    # Kurtosis g2 = n * M4 / M2^2 - 3 (excess)
    kurt = (n * M4) / (M2 * M2) - 3.0
    return mean, var, skew, kurt, n


# ============================================================================
# Kahan-Babuška-Neumaier compensated sum (cheaper than Welford for plain sum)
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def kahan_sum_seq(arr: np.ndarray) -> float:
    """Neumaier-improved Kahan compensated sum.

    Recovers ~log10(N) digits vs naive sum at marginal cost (one extra
    add/subtract per element). Numerically equivalent to ``np.sum`` for
    well-conditioned inputs; superior for ill-conditioned (mixed-magnitude)
    sums.
    """
    s = 0.0
    c = 0.0
    for x in arr:
        if not np.isfinite(x):
            continue
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return s + c


@numba.njit(**NUMBA_NJIT_PARAMS)
def kahan_dot_seq(a: np.ndarray, b: np.ndarray) -> float:
    """Kahan-compensated dot product. Used for slope/correlation numerators
    where ``sum(x_i * y_i)`` accumulates many terms of varied magnitude.
    """
    n = min(a.shape[0], b.shape[0])
    s = 0.0
    c = 0.0
    for i in range(n):
        x = a[i] * b[i]
        if not np.isfinite(x):
            continue
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    return s + c


# ============================================================================
# Naive baseline implementations (matching the patterns in numerical.py)
# Used by the benchmark script to quantify the improvement.
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def naive_mean_var_two_pass_seq(arr: np.ndarray) -> tuple:
    """Two-pass mean + variance (matches `compute_simple_stats_numba` pattern).

    First pass: sum to compute mean. Second pass: sum of squared deviations
    (UNCOMPENSATED — matches the current naive accumulator).
    """
    n = 0
    total = 0.0
    for x in arr:
        if np.isfinite(x):
            n += 1
            total += x
    if n == 0:
        return 0.0, 0.0, 0
    mean = total / n
    s = 0.0
    for x in arr:
        if np.isfinite(x):
            d = x - mean
            s += d * d  # UNCOMPENSATED
    return mean, s / n, n


@numba.njit(**NUMBA_NJIT_PARAMS)
def kahan_two_pass_var_seq(arr: np.ndarray) -> tuple:
    """Two-pass mean + variance with Kahan-Babuška-Neumaier compensation
    in BOTH sums. Best precision across both 'long array drift' and
    'large-mean-small-var cancellation' regimes.

    Pass 1: Kahan-compensated sum to get exact mean.
    Pass 2: Kahan-compensated sum of (x - mean)^2.

    Cost: ~2x naive, ~equal to Welford. Stable in all input regimes
    (Welford wins on raw long-array sums but loses on large-mean cases
    because its running mean accumulates per-element rounding; Kahan-
    two-pass avoids both pitfalls).
    """
    n = 0
    s = 0.0
    c = 0.0
    for x in arr:
        if not np.isfinite(x):
            continue
        n += 1
        t = s + x
        if abs(s) >= abs(x):
            c += (s - t) + x
        else:
            c += (x - t) + s
        s = t
    if n == 0:
        return 0.0, 0.0, 0
    mean = (s + c) / n
    s2 = 0.0
    c2 = 0.0
    for x in arr:
        if not np.isfinite(x):
            continue
        d = x - mean
        d2 = d * d
        t = s2 + d2
        if abs(s2) >= abs(d2):
            c2 += (s2 - t) + d2
        else:
            c2 += (d2 - t) + s2
        s2 = t
    return mean, (s2 + c2) / n, n


@numba.njit(**NUMBA_NJIT_PARAMS)
def naive_moments_two_pass_seq(arr: np.ndarray) -> tuple:
    """Two-pass mean / variance / skewness / kurtosis (matches
    `compute_moments_slope_mi` pattern). Uncompensated sums of d^2/d^3/d^4."""
    n = 0
    total = 0.0
    for x in arr:
        if np.isfinite(x):
            n += 1
            total += x
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0
    mean = total / n
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for x in arr:
        if np.isfinite(x):
            d = x - mean
            d2 = d * d
            s2 += d2
            s3 += d2 * d
            s4 += d2 * d2
    if n < 2:
        return mean, 0.0, 0.0, 0.0, n
    var = s2 / n
    if var == 0:
        return mean, 0.0, 0.0, 0.0, n
    std = np.sqrt(var)
    skew = (s3 / n) / (std ** 3)
    kurt = (s4 / n) / (var ** 2) - 3.0
    return mean, var, skew, kurt, n
