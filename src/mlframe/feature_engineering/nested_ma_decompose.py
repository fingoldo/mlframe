"""``nested_ma_decompose``: recover the exclusive trailing-window average from two precomputed moving averages.

Source: av_top3_rampaging_datahulk_minihack2017.md -- ``MA_last_10_3 = (Ten_Day_MA*10 - Three_Day_MA*3)/7``,
recovering "the average of the seven days preceding the last three days" algebraically. Given ``MA(w1)`` and
``MA(w2)`` (``w1 < w2``, both ending at the same point), ``MA_w2*w2`` is the sum over ``w2`` days and
``MA_w1*w1`` is the sum over the most recent ``w1`` of those days -- subtracting and dividing by the remaining
``w2-w1`` days gives that EXCLUSIVE window's average without a third rolling pass over the raw series.
"""
from __future__ import annotations

from typing import Union

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


__all__ = ["nested_ma_decompose"]
