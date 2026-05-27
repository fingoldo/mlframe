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
    "ewma_residual",
    "local_linear_detrend",
    "cusum_features",
    "quantile_normalize_per_group",
]

from typing import Iterable, Optional, Sequence

import numpy as np

from .grouped import iter_group_segments


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


def ewma_residual(
    values: np.ndarray,
    half_life: float | Iterable[float] = 20.0,
    *,
    group_ids: Optional[np.ndarray] = None,
    adjust: bool = False,
) -> np.ndarray:
    """Residual after exponential-moving-average detrending.

    Returns ``x - EWMA(x, half_life)`` -- the de-trended series. Sweep
    over multiple half-lives by passing an iterable for ``half_life``;
    output then has shape ``(n, len(half_life))``.

    The default-first stationarity primitive. Half-life is in row units
    (alpha = 1 - 0.5^(1/half_life)). Per-group support guarantees no
    bleed across boundaries.

    Use cases: any drifting signal (price, sensor temperature, request
    rate, sales). Multi-scale (e.g. [5, 20, 60, 240]) gives multi-scale
    detrending almost free (O(n) per half-life).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size

    if isinstance(half_life, (int, float)):
        hl_list = (float(half_life),)
        is_scalar = True
    else:
        hl_list = tuple(float(h) for h in half_life)
        is_scalar = False
    if not hl_list:
        raise ValueError("half_life must be a float or non-empty iterable")
    for h in hl_list:
        if h <= 0:
            raise ValueError(f"half_life must be > 0, got {h}")

    def _ewma_single(seg: np.ndarray, hl: float) -> np.ndarray:
        alpha = 1.0 - 2.0 ** (-1.0 / hl)
        seg_f = np.where(np.isfinite(seg), seg, 0.0)
        ewma = np.empty_like(seg_f)
        ewma[0] = seg_f[0]
        for i in range(1, seg_f.size):
            ewma[i] = alpha * seg_f[i] + (1.0 - alpha) * ewma[i - 1]
        if adjust:
            # Pandas-style bias correction at the start.
            w = (1.0 - alpha) ** np.arange(seg_f.size)
            w_cum = np.cumsum(w[::-1])
            ewma = ewma * w[::-1] / w_cum
        return seg - ewma

    if group_ids is None:
        if is_scalar:
            return _ewma_single(arr, hl_list[0])
        out = np.full((n, len(hl_list)), np.nan, dtype=np.float64)
        for j, hl in enumerate(hl_list):
            out[:, j] = _ewma_single(arr, hl)
        return out

    sort_idx, starts, ends = iter_group_segments(group_ids)
    if is_scalar:
        out = np.full(n, np.nan, dtype=np.float64)
        for s, e in zip(starts, ends):
            idx_seg = sort_idx[s:e]
            if idx_seg.size < 2:
                continue
            out[idx_seg] = _ewma_single(arr[idx_seg], hl_list[0])
        return out
    out = np.full((n, len(hl_list)), np.nan, dtype=np.float64)
    for s, e in zip(starts, ends):
        idx_seg = sort_idx[s:e]
        if idx_seg.size < 2:
            continue
        seg = arr[idx_seg]
        for j, hl in enumerate(hl_list):
            out[idx_seg, j] = _ewma_single(seg, hl)
    return out


def local_linear_detrend(
    values: np.ndarray,
    window_K: int = 50,
    *,
    group_ids: Optional[np.ndarray] = None,
    return_slope: bool = True,
    fill_value: float = np.nan,
) -> dict:
    """Rolling OLS ``y_t ~ a + b * t`` per K-window; emit residual + slope.

    For each row, fit ``y = a + b * t`` on the trailing K-window
    (anchored at the row), return ``(y_actual - y_fit, slope)``. Slope
    alone is a generic momentum / direction feature. Returns
    ``{"residual": ..., "slope": ...}``.

    Use cases: piecewise-linear trends (sensor drift, regime shifts,
    growth curves); momentum signal in finance / ad-spend; direction-
    aware classification.

    Implementation uses prefix-sum (Σx, Σx², Σxy, Σy) so per-row cost
    is O(1) after a single O(n) prefix pass per group. K must be >= 2.
    """
    if window_K < 2:
        raise ValueError(f"window_K must be >= 2, got {window_K}")
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    out_resid = np.full(n, fill_value, dtype=np.float64)
    out_slope = np.full(n, fill_value, dtype=np.float64)

    def _fit_segment(seg: np.ndarray) -> tuple:
        # For each row r >= K-1: fit on rows [r-K+1, r].
        seg_f = np.where(np.isfinite(seg), seg, 0.0)
        m = seg.size
        if m < window_K:
            return np.full(m, np.nan), np.full(m, np.nan)
        t = np.arange(window_K, dtype=np.float64)
        t_mean = t.mean()
        t_var = ((t - t_mean) ** 2).sum() + 1e-12  # scalar
        from numpy.lib.stride_tricks import sliding_window_view
        wins = sliding_window_view(seg_f, window_K)
        y_mean = wins.mean(axis=1)
        # b = Σ(t - t_mean)(y - y_mean) / Σ(t - t_mean)^2
        t_dev = (t - t_mean)
        cov = ((wins - y_mean[:, None]) * t_dev[None, :]).sum(axis=1)
        b = cov / t_var
        a = y_mean - b * t_mean
        # Predicted y at the LAST position of each window (t = K - 1).
        y_pred_last = a + b * (window_K - 1)
        y_actual_last = wins[:, -1]
        resid_out = np.full(m, np.nan, dtype=np.float64)
        slope_out = np.full(m, np.nan, dtype=np.float64)
        resid_out[window_K - 1:] = y_actual_last - y_pred_last
        slope_out[window_K - 1:] = b
        return resid_out, slope_out

    if group_ids is None:
        r, s = _fit_segment(arr)
        out_resid[:] = r
        out_slope[:] = s
    else:
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            idx_seg = sort_idx[s:e]
            if idx_seg.size < window_K:
                continue
            r, sl = _fit_segment(arr[idx_seg])
            out_resid[idx_seg] = r
            out_slope[idx_seg] = sl

    out = {"residual": out_resid}
    if return_slope:
        out["slope"] = out_slope
    return out


def cusum_features(
    values: np.ndarray,
    threshold: Optional[float] = None,
    *,
    group_ids: Optional[np.ndarray] = None,
    drift: float = 0.0,
) -> dict:
    """Two-sided CUSUM (Page-Hinkley) features for change-point detection.

    Computes positive and negative cumulative deviation processes
    relative to the running mean, with reset whenever they exceed
    ``threshold`` (default = ``5 * mad`` of the input). Emits:
    * ``cusum_pos`` / ``cusum_neg`` — instantaneous CUSUM values
    * ``rows_since_reset`` — rows since last threshold crossing
    * ``n_resets_in_window`` — running count of resets

    Use cases: fraud / behaviour shift, predictive maintenance
    (degradation onset), demand (promo start), trading (regime change).
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if threshold is None:
        finite = arr[np.isfinite(arr)]
        if finite.size < 2:
            threshold = 1.0
        else:
            mad = float(np.median(np.abs(finite - np.median(finite))))
            threshold = 5.0 * mad * 1.4826 if mad > 0 else 1.0

    out_pos = np.zeros(n, dtype=np.float64)
    out_neg = np.zeros(n, dtype=np.float64)
    out_since = np.zeros(n, dtype=np.float64)
    out_count = np.zeros(n, dtype=np.float64)

    def _walk(idx_seg: np.ndarray) -> None:
        m = idx_seg.size
        if m == 0:
            return
        seg = arr[idx_seg]
        seg_mean = float(np.nanmean(seg)) if np.isfinite(seg).any() else 0.0
        pos = 0.0
        neg = 0.0
        rows_since = 0.0
        n_resets = 0
        for i in range(m):
            x = seg[i]
            if not np.isfinite(x):
                out_pos[idx_seg[i]] = pos
                out_neg[idx_seg[i]] = neg
                out_since[idx_seg[i]] = rows_since
                out_count[idx_seg[i]] = n_resets
                rows_since += 1
                continue
            dev = x - seg_mean
            pos = max(0.0, pos + dev - drift)
            neg = min(0.0, neg + dev + drift)
            triggered = (pos > threshold) or (neg < -threshold)
            if triggered:
                pos = 0.0
                neg = 0.0
                rows_since = 0.0
                n_resets += 1
            else:
                rows_since += 1
            out_pos[idx_seg[i]] = pos
            out_neg[idx_seg[i]] = neg
            out_since[idx_seg[i]] = rows_since
            out_count[idx_seg[i]] = n_resets

    if group_ids is None:
        _walk(np.arange(n))
    else:
        sort_idx, starts, ends = iter_group_segments(group_ids)
        for s, e in zip(starts, ends):
            _walk(sort_idx[s:e])

    return {
        "cusum_pos": out_pos,
        "cusum_neg": out_neg,
        "rows_since_reset": out_since,
        "n_resets_in_window": out_count,
    }


def quantile_normalize_per_group(
    values: np.ndarray,
    *,
    group_ids: Optional[np.ndarray] = None,
    to_normal: bool = False,
) -> np.ndarray:
    """Map values to within-group uniform CDF rank in [0, 1].

    For each group, transform values to their empirical-CDF rank
    (ties broken by stable sort, scaled to [1/n, n/n]). With
    ``to_normal=True``, applies the inverse standard-normal CDF
    (probit) so the result is N(0, 1)-distributed within each group --
    useful for downstream models that assume Gaussian features.

    Use cases: cross-entity comparability when scales differ (tickers
    on different price levels, sensors with different baselines, users
    with different activity distributions). Complementary to ``frac_diff``
    (temporal stationarity) and ``ewma_residual`` (level detrending).
    """
    from .grouped import per_group_rank
    if group_ids is None:
        group_ids = np.zeros(len(values), dtype=np.int64)
    pct = per_group_rank(values, group_ids, pct=True)
    if not to_normal:
        return pct
    # Clip to avoid +/-inf at exact 0 / 1.
    pct_clip = np.clip(pct, 1e-9, 1.0 - 1e-9)
    # Inverse standard normal CDF via erfinv.
    from scipy.special import ndtri
    return ndtri(pct_clip)
