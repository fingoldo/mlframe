"""Compute the Hurst Exponent of an 1D array by the means of R/S analisys:
    
    https://en.wikipedia.org/wiki/Hurst_exponent
"""

# pylint: disable=wrong-import-order,wrong-import-position,unidiomatic-typecheck,pointless-string-statement

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Packages
# ----------------------------------------------------------------------------------------------------------------------------

from pyutilz.pythonlib import (
    ensure_installed,
)  # lint: disable=ungrouped-imports,disable=wrong-import-order

ensure_installed("numpy pandas numba scipy sklearn antropy")

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *
import numpy as np
from numba import njit

# ----------------------------------------------------------------------------------------------------------------------------
# Inits
# ----------------------------------------------------------------------------------------------------------------------------

fastmath = False

# ----------------------------------------------------------------------------------------------------------------------------
# Core funcs
# ----------------------------------------------------------------------------------------------------------------------------


@njit(fastmath=fastmath)
def compute_hurst_rs(arr: np.ndarray, agg_func: object = np.mean):
    """Computes R/S stat for a single window."""

    mean = agg_func(arr)

    deviations = arr - mean
    Z = np.cumsum(deviations)
    R = np.max(Z) - np.min(Z)
    S = np.std(arr)  # , ddof=1

    if R == 0 or S == 0:
        return 0.0  # to skip this interval due the undefined R/S ratio

    return R / S


@njit(fastmath=fastmath)
def precompute_hurst_exponent(
    arr: np.ndarray, min_window: int = 5, max_window: int = None, windows_log_step: float = 0.25, take_diffs: bool = True, agg_func: object = np.mean
):
    """Computes R/S stat for a single window."""

    # Get diffs, if needed

    if take_diffs:
        arr = arr[1:] - arr[:-1]

    L = len(arr)

    # Split parent array several times into a number of equal chunks, increasing the chunk length

    max_window = max_window or (L - 1)
    window_sizes = (10 ** np.arange(np.log10(min_window), np.log10(max_window), windows_log_step)).astype(np.int32)
    # window_sizes.append(L)

    RS = []
    used_window_sizes = []
    for w in window_sizes:
        rs = []
        for start in range(0, L, w):
            if (start + w) >= L:
                break
            partial_rs = compute_hurst_rs(arr[start : start + w])  # , agg_func=agg_func)
            if partial_rs:
                rs.append(partial_rs)
        if rs:
            RS.append(agg_func(np.array(rs)))
            used_window_sizes.append(w)

    return used_window_sizes, RS


def compute_hurst_exponent(arr: np.ndarray, min_window: int = 5, max_window: int = None, windows_log_step: float = 0.25, take_diffs: bool = False)->tuple:
    """Main enrtypoint to compute a Hurst Exponent (and the constant) of a numerical array."""
    if len(arr) < min_window:
        return np.nan, np.nan
    window_sizes, rs = precompute_hurst_exponent(
        arr=arr, min_window=min_window, max_window=max_window, windows_log_step=windows_log_step, take_diffs=take_diffs
    )
    x = np.vstack([np.log10(window_sizes), np.ones(len(rs))]).T
    h, c = np.linalg.lstsq(x, np.log10(rs), rcond=-1)[0]
    c = 10**c
    return h, c


def hurst_testing():

    # pip install hurst

    from hurst import random_walk

    brownian = random_walk(1000, proba=0.5)
    print(compute_hurst_exponent(np.array(brownian)))

    persistent = random_walk(1000, proba=0.7)
    print(compute_hurst_exponent(np.array(persistent)))

    antipersistent = random_walk(1000, proba=0.3)
    print(compute_hurst_exponent(np.array(antipersistent)))
