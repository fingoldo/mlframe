"""Stationarity transforms for time-series features.

Currently ships:

* ``frac_diff`` -- Lopez de Prado's fractional differencing
  (Advances in Financial Machine Learning, Ch. 5). Removes a fractional
  amount of the unit root so the series becomes stationary while
  preserving more memory than the integer differencing
  (``d=1`` first-difference) standard.

Future additions in this file: ADF / KPSS / PP test wrappers, ARMA
order detection.

Per-group support via ``group_ids`` kwarg so panel / clustered data
gets per-group differencing (avoiding bleed across group boundaries).
"""

from __future__ import annotations

__all__ = [
    "frac_diff_weights",
    "frac_diff",
]

from typing import Iterable, Optional, Sequence

import numpy as np


def frac_diff_weights(d: float, K: int) -> np.ndarray:
    """Coefficient vector for fractional differencing at exponent ``d``.

    Returns weights ``w_k`` for ``k = 0, ..., K-1`` such that
    ``y_t ~= sum_{k=0..K-1} w_k * x_{t-k}`` (truncated).

    Formula (de Prado 5.4): ``w_0 = 1``,
    ``w_k = -w_{k-1} * (d - k + 1) / k``.

    ``K`` is the truncation length; longer K = more memory preserved
    but more pre-roll missing values at the start of the series.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")
    w = np.empty(K, dtype=np.float64)
    w[0] = 1.0
    for n in range(1, K):
        w[n] = -w[n - 1] * (d - n + 1) / n
    return w


def _frac_diff_single(arr: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Convolve a single 1-D array with truncated weights ``w``.

    First ``K`` rows are NaN (no full-window history). NaN inputs
    propagate the way numpy's ``np.convolve`` handles them (filled
    with 0 by caller; downstream NaN is caller's responsibility).
    """
    K = w.size
    if arr.size < K:
        return np.full(arr.size, np.nan, dtype=np.float64)
    arr_f = np.where(np.isfinite(arr), arr, 0.0).astype(np.float64)
    # Causal convolution: y[t] = sum_k w[k] * x[t-k].
    # np.convolve(x, h, mode='full')[t] = sum_k h[k] * x[t-k] directly,
    # so pass w (not w[::-1]). Truncate to length n to drop the
    # post-arr tail.
    conv = np.convolve(arr_f, w, mode="full")[: arr.size]
    conv[:K] = np.nan
    return conv


def frac_diff(
    values: np.ndarray,
    d: float | Iterable[float] = 0.5,
    K: int = 30,
    *,
    group_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fractional differencing of a 1-D series at one or more ``d``.

    Parameters
    ----------
    values
        1-D input series.
    d
        Single float or iterable of floats. When iterable, output
        shape is ``(n, len(d))`` with one column per d. Common picks:
        d in (0.3, 0.5, 0.7) sweeps the stationarity/memory tradeoff;
        the smallest d that gives a stationary series (per ADF) is
        the de Prado-optimal value.
    K
        Truncation length of the weight kernel. Default 30 retains
        ~99% of the weight mass for d in [0.1, 0.9]; raise for d <
        0.1 where the geometric decay is slower.
    group_ids
        Optional per-row group identifiers. When supplied, the
        fractional difference is computed PER GROUP so the convolution
        doesn't bleed across group boundaries (each group gets its
        own NaN-padded prefix). Same shape contract as
        ``feature_engineering.grouped.iter_group_segments``.

    Returns
    -------
    np.ndarray
        1-D of length ``n`` when ``d`` is scalar; 2-D ``(n, len(d))``
        when ``d`` is iterable. NaN-filled for the first ``K`` rows
        of each group (no full-history window).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size

    if isinstance(d, (int, float)):
        d_list: Sequence[float] = (float(d),)
        is_scalar = True
    else:
        d_list = tuple(float(x) for x in d)
        is_scalar = False
    if not d_list:
        raise ValueError("`d` must be a float or non-empty iterable of floats")

    weights = {di: frac_diff_weights(di, K) for di in d_list}

    if group_ids is None:
        if is_scalar:
            return _frac_diff_single(arr, weights[d_list[0]])
        out = np.full((n, len(d_list)), np.nan, dtype=np.float64)
        for j, di in enumerate(d_list):
            out[:, j] = _frac_diff_single(arr, weights[di])
        return out

    # per-group
    from .grouped import iter_group_segments
    sort_idx, starts, ends = iter_group_segments(group_ids)
    if is_scalar:
        out = np.full(n, np.nan, dtype=np.float64)
        arr_sorted = arr[sort_idx]
        for s, e in zip(starts, ends):
            seg = arr_sorted[s:e]
            if seg.size < K:
                continue
            out[sort_idx[s:e]] = _frac_diff_single(seg, weights[d_list[0]])
        return out
    out = np.full((n, len(d_list)), np.nan, dtype=np.float64)
    arr_sorted = arr[sort_idx]
    for s, e in zip(starts, ends):
        seg = arr_sorted[s:e]
        if seg.size < K:
            continue
        for j, di in enumerate(d_list):
            out[sort_idx[s:e], j] = _frac_diff_single(seg, weights[di])
    return out
