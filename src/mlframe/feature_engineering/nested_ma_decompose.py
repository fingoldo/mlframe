"""``nested_ma_decompose``: recover the exclusive trailing-window average from two precomputed moving averages.

Source: av_top3_rampaging_datahulk_minihack2017.md -- ``MA_last_10_3 = (Ten_Day_MA*10 - Three_Day_MA*3)/7``,
recovering "the average of the seven days preceding the last three days" algebraically. Given ``MA(w1)`` and
``MA(w2)`` (``w1 < w2``, both ending at the same point), ``MA_w2*w2`` is the sum over ``w2`` days and
``MA_w1*w1`` is the sum over the most recent ``w1`` of those days -- subtracting and dividing by the remaining
``w2-w1`` days gives that EXCLUSIVE window's average without a third rolling pass over the raw series.
"""
from __future__ import annotations

from typing import List, Sequence, Union

import numpy as np
import pandas as pd

ArrayLike = Union[np.ndarray, pd.Series]


def nested_ma_decompose(ma_short: ArrayLike, ma_long: ArrayLike, window_short: int, window_long: int) -> np.ndarray:
    """Average of the exclusive ``(window_long - window_short)``-length window preceding ``ma_short``'s window.

    Parameters
    ----------
    ma_short, ma_long
        ``(n,)`` precomputed moving averages of window sizes ``window_short``/``window_long``, aligned
        (same index/end-point per row).
    window_short, window_long
        The two MA window sizes; ``window_long`` must be strictly greater than ``window_short``.

    Returns
    -------
    np.ndarray
        ``(n,)`` average of the exclusive ``window_long - window_short`` days immediately preceding each
        row's ``window_short``-day window.
    """
    if window_long <= window_short:
        raise ValueError(f"nested_ma_decompose: window_long ({window_long}) must be > window_short ({window_short}).")
    ma_short_arr = np.asarray(ma_short, dtype=np.float64)
    ma_long_arr = np.asarray(ma_long, dtype=np.float64)
    exclusive_window = window_long - window_short
    return np.asarray((ma_long_arr * window_long - ma_short_arr * window_short) / exclusive_window)


def nested_ma_decompose_chain(mas: Sequence[ArrayLike], windows: Sequence[int]) -> List[np.ndarray]:
    """Decompose a full ladder of nested MA windows into all consecutive exclusive-window averages at once.

    Opt-in vectorized equivalent of calling ``nested_ma_decompose`` once per consecutive pair
    ``(windows[i], windows[i+1])`` -- e.g. given ``MA(3), MA(10), MA(20)`` this returns, in one pass, the
    average of "the 7 days before the last 3" and the average of "the 10 days before the last 10", instead of
    requiring two separate pairwise calls. Numerically bit-identical to the pairwise chain: each pair applies
    the exact same ``(long*w_long - short*w_short) / (w_long - w_short)`` arithmetic in the same order.

    Parameters
    ----------
    mas
        ``k >= 2`` precomputed moving averages, each ``(n,)``, aligned and ordered to match ``windows``.
    windows
        ``k`` strictly increasing MA window sizes, one per entry in ``mas``.

    Returns
    -------
    list[np.ndarray]
        ``k - 1`` arrays, each ``(n,)``: entry ``i`` is the exclusive average of the
        ``windows[i+1] - windows[i]`` days preceding ``mas[i]``'s window (same as
        ``nested_ma_decompose(mas[i], mas[i+1], windows[i], windows[i+1])``).
    """
    if len(mas) != len(windows):
        raise ValueError(f"nested_ma_decompose_chain: got {len(mas)} mas but {len(windows)} windows; must match.")
    if len(windows) < 2:
        raise ValueError(f"nested_ma_decompose_chain: need >= 2 windows to form a chain, got {len(windows)}.")
    for i in range(len(windows) - 1):
        if windows[i + 1] <= windows[i]:
            raise ValueError(f"nested_ma_decompose_chain: windows must be strictly increasing, got {list(windows)}.")

    mas_arr = np.stack([np.asarray(ma, dtype=np.float64) for ma in mas], axis=0)  # (k, n)
    windows_arr = np.asarray(windows, dtype=np.float64)  # (k,)
    sums = mas_arr * windows_arr[:, None]  # (k, n): MA(w)*w per rung, matches the pairwise per-call product
    exclusive_windows = windows_arr[1:] - windows_arr[:-1]  # (k-1,)
    exclusive_sums = sums[1:] - sums[:-1]  # (k-1, n)
    result = exclusive_sums / exclusive_windows[:, None]  # (k-1, n)
    return [result[i] for i in range(result.shape[0])]


__all__ = ["nested_ma_decompose", "nested_ma_decompose_chain"]
