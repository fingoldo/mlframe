"""Compute the Hurst Exponent of a 1D array via R/S analysis.

https://en.wikipedia.org/wiki/Hurst_exponent
"""

__all__ = [
    "compute_hurst_rs",
    "precompute_hurst_exponent",
    "compute_hurst_exponent",
]

import logging

logger = logging.getLogger(__name__)

import numpy as np
from numba import njit


_FASTMATH = False
_ZERO_EPS = 1e-12


@njit(fastmath=_FASTMATH)
def compute_hurst_rs(arr: np.ndarray) -> float:
    """Rescaled-range (R/S) statistic for one window.

    Standard deviation uses ddof=1 (sample std) per the Mandelbrot / Hurst (1951) convention
    for R/S analysis: the rescaled range is divided by the unbiased estimator of dispersion.
    """
    mean = np.mean(arr)
    deviations = arr - mean
    Z = np.cumsum(deviations)
    R = np.max(Z) - np.min(Z)
    n = len(arr)
    if n < 2:
        return np.nan
    var = np.sum(deviations * deviations) / (n - 1)
    S = np.sqrt(var)
    if R <= _ZERO_EPS or S <= _ZERO_EPS:
        return np.nan
    return R / S


@njit(fastmath=_FASTMATH)
def precompute_hurst_exponent(
    arr: np.ndarray,
    min_window: int = 5,
    max_window: int = -1,
    windows_log_step: float = 0.25,
    take_diffs: bool = False,
):
    """R/S aggregated across a geometric ladder of window sizes.

    ``max_window=-1`` is the "auto" sentinel and resolves to ``L-1``.  ``take_diffs`` defaults to
    ``False`` to match the consumer-facing ``compute_hurst_exponent``; pass ``True`` for raw price
    or random-walk paths where the Hurst signal lives in the increment series.
    """
    if take_diffs:
        arr = arr[1:] - arr[:-1]

    L = len(arr)
    if L < 2 or min_window < 2:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    if max_window <= 0:
        max_window = L - 1
    if max_window <= min_window:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    raw_sizes = (10.0 ** np.arange(np.log10(min_window), np.log10(max_window), windows_log_step)).astype(np.int64)
    window_sizes = np.unique(raw_sizes)

    n_sizes = len(window_sizes)
    out_sizes = np.empty(n_sizes, dtype=np.int64)
    out_rs = np.empty(n_sizes, dtype=np.float64)
    out_count = 0

    for idx in range(n_sizes):
        w = window_sizes[idx]
        if w < 2 or w > L:
            continue
        n_windows = L // w
        if n_windows == 0:
            continue
        sum_rs = 0.0
        cnt_rs = 0
        for k in range(n_windows):
            start = k * w
            partial_rs = compute_hurst_rs(arr[start : start + w])
            if not np.isnan(partial_rs):
                sum_rs += partial_rs
                cnt_rs += 1
        if cnt_rs > 0:
            out_sizes[out_count] = w
            out_rs[out_count] = sum_rs / cnt_rs
            out_count += 1

    return out_sizes[:out_count], out_rs[:out_count]


def compute_hurst_exponent(
    arr: np.ndarray,
    min_window: int = 5,
    max_window=None,
    windows_log_step: float = 0.25,
    take_diffs: bool = False,
) -> tuple:
    """Hurst exponent + intercept constant for a 1D array.

    Returns ``(H, c)`` such that ``E[R/S(n)] ~= c * n**H``.  Returns ``(nan, nan)`` when the array
    is too short or the R/S ladder is degenerate (no positive points to log-fit).

    ``max_window=None`` resolves to ``len(arr) - 1`` inside the kernel; the sentinel must be an int
    because the kernel is ``@njit`` and cannot accept ``None``.
    """
    if len(arr) < min_window:
        return np.nan, np.nan
    max_window_int = -1 if max_window is None else int(max_window)
    window_sizes, rs = precompute_hurst_exponent(
        arr=arr,
        min_window=min_window,
        max_window=max_window_int,
        windows_log_step=windows_log_step,
        take_diffs=take_diffs,
    )
    if len(rs) < 2:
        return np.nan, np.nan
    rs_arr = np.asarray(rs, dtype=float)
    window_sizes_arr = np.asarray(window_sizes, dtype=float)
    if np.any(rs_arr <= 0) or np.any(window_sizes_arr <= 0):
        return np.nan, np.nan
    x = np.vstack([np.log10(window_sizes_arr), np.ones(len(rs_arr))]).T
    h, c = np.linalg.lstsq(x, np.log10(rs_arr), rcond=None)[0]
    c = 10 ** c
    return h, c
