"""Shared numerically stable running-variance primitives for njit kernels.

A running ``sum(v*v) - n*mean^2`` variance cancels catastrophically when the level dwarfs the spread (epoch seconds, prices at 1e5 with cent moves).
Welford's update keeps a running mean and the sum of squared deviations ``m2`` instead, so no large terms are ever subtracted.
"""

from __future__ import annotations

import numba
import numpy as np


@numba.njit(cache=True, nogil=True)
def welford_push(cnt: int, mean: float, m2: float, v: float) -> tuple:
    """Fold ``v`` into a running ``(mean, m2)`` state that currently holds ``cnt`` values; returns the updated ``(mean, m2)``."""
    delta = v - mean
    mean += delta / (cnt + 1)
    m2 += delta * (v - mean)
    return mean, m2


@numba.njit(cache=True, nogil=True)
def welford_std(m2: float, cnt: int, ddof: int) -> float:
    """Standard deviation from a Welford ``m2`` over ``cnt`` values; 0.0 when ``cnt <= ddof`` or the variance is not positive."""
    if cnt <= ddof:
        return 0.0
    var = m2 / (cnt - ddof)
    return np.sqrt(var) if var > 0.0 else 0.0
