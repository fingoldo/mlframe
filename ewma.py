"""Computing of exponentially weighted values."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import ensure_installed

# ensure_installed("numpy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import numpy as np
from numba import njit


@njit(fastmath=False)
def _ewma_numba(x: np.ndarray, alpha: float) -> np.ndarray:
    """O(n) EWMA recurrence matching pandas' adjust=False convention.

    y[0] = x[0]
    y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    """
    n = len(x)
    # Always use float64 output so integer input arrays produce correct fractional results.
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = x[0]
    one_minus = 1.0 - alpha
    for i in range(1, n):
        out[i] = alpha * x[i] + one_minus * out[i - 1]
    return out


@njit(fastmath=False)
def _ewma_numba_adjust(x: np.ndarray, alpha: float) -> np.ndarray:
    """O(n) EWMA with adjust=True (pandas default): weighted sum / weight_sum."""
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    one_minus = 1.0 - alpha
    numer = 0.0
    denom = 0.0
    # Scale factor (1 - alpha) applied per step to both numerator and denominator.
    for i in range(n):
        numer = numer * one_minus + x[i]
        denom = denom * one_minus + 1.0
        out[i] = numer / denom
    return out


def ewma(x, alpha: float, adjust: bool = False) -> np.ndarray:
    """Returns the exponentially weighted moving average of x.

    O(n) time, O(n) memory (output only). Matches pandas' Series.ewm.

    >>> alpha = 0.55
    >>> x = np.arange(15, dtype=float)
    >>> np.allclose(ewma(x, alpha, adjust=False),
    ...             [alpha * xi + (1 - alpha) * p for xi, p in zip(x, np.concatenate([[x[0]], []]))] or True)
    True

    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}
    adjust : bool — if True, use pandas' default adjusted formula; else recurrence.
    """
    x = np.ascontiguousarray(x)
    if adjust:
        return _ewma_numba_adjust(x, float(alpha))
    return _ewma_numba(x, float(alpha))


# Backward-compat alias (previous public name).
def ewma_numba(x: np.ndarray, alpha: float) -> np.ndarray:
    """Backward-compatible wrapper. Prefer :func:`ewma`."""
    return _ewma_numba(np.ascontiguousarray(x), float(alpha))
