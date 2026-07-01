"""Nadaraya-Watson kernel regression / smoothing over ordered signals.

From PZAD «задача о пробках» (Dyakonov 2020): to get a continuous "average speed
at every instant" from noisy, irregularly-sampled measurements, average the target
values weighted by kernel proximity in feature space:

    a(x) = sum_i K((x - x_i)/h) * y_i / sum_i K((x - x_i)/h)

The lecture's punchline — "а ведь это тоже весовая схема!" — ties this to the same
recency-weighting family as the caseclients estimators: NW is a *query-dependent*
weighted mean. So this primitive accepts an optional per-sample ``sample_weight``
that MULTIPLIES the kernel weight, letting a caller compose kernel proximity with a
recency / seasonal-analog weight (the lecture's "special averaging" recipe: this day
+ this weekday + yesterday).

Provided:
- ``nadaraya_watson_smooth``: 1-D NW regression/smoothing on a flat signal.
- ``per_group_nadaraya_watson_smooth``: per-entity smoothing (e.g. per road-arc speed
  over time), reusing the sort-by-factor / store-starts-ends trick from the lecture's
  «полезные приёмы» slide.

Known NW caveats (lecture slide 13), surfaced in the docstrings: boundary bias, does
not extrapolate meaningfully beyond the data range, and is sensitive to bandwidth.
``bandwidth <= 0`` falls back to a Silverman rule; callers that care should CV-tune it.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)

__all__ = ["nadaraya_watson_smooth", "per_group_nadaraya_watson_smooth", "KERNELS"]

KERNELS = ("gaussian", "epanechnikov", "boxcar", "tricube")

# Parallelize over query points once there are enough to amortise the prange spawn. Env-overridable.
_NW_PARALLEL_MIN_QUERIES = int(os.environ.get("MLFRAME_NW_PARALLEL_MIN_QUERIES", "2000"))


@njit(fastmath=False, cache=True, inline="always")
def _kernel(u: float, kernel_code: int) -> float:
    """Kernel value at scaled distance ``u = |x - x_i| / h`` (u >= 0). 0=gaussian,1=epanechnikov,2=boxcar,3=tricube."""
    if kernel_code == 0:
        return np.exp(-0.5 * u * u)
    if kernel_code == 1:
        return 1.0 - u * u if u < 1.0 else 0.0
    if kernel_code == 2:
        return 1.0 if u <= 1.0 else 0.0
    # tricube
    if u < 1.0:
        t = 1.0 - u * u * u
        return t * t * t
    return 0.0


@njit(fastmath=False, cache=True)
def _silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman rule-of-thumb bandwidth from the sample std (floored to the value range / 1.0)."""
    n = x.shape[0]
    if n <= 1:
        return 1.0
    mean = 0.0
    xmin = x[0]
    xmax = x[0]
    for i in range(n):
        mean += x[i]
        if x[i] < xmin:
            xmin = x[i]
        if x[i] > xmax:
            xmax = x[i]
    mean /= n
    var = 0.0
    for i in range(n):
        d = x[i] - mean
        var += d * d
    std = (var / n) ** 0.5
    if std <= 0.0:
        span = xmax - xmin
        std = span if span > 0.0 else 1.0
    h = 1.06 * std * (n ** (-0.2))
    return h if h > 0.0 else 1.0


@njit(fastmath=False, cache=True, inline="always")
def _nw_at(xq, x_sorted, y, w, kernel_code, h, use_w):
    """NW estimate a(xq) over samples (x, y) with optional per-sample multiplier w. Returns NaN if no kernel mass."""
    inv_h = 1.0 / h
    num = 0.0
    den = 0.0
    for i in range(x_sorted.shape[0]):
        u = abs(xq - x_sorted[i]) * inv_h
        k = _kernel(u, kernel_code)
        if k != 0.0:
            wi = k * w[i] if use_w else k
            num += wi * y[i]
            den += wi
    return num / den if den > 0.0 else np.nan


@njit(fastmath=False, cache=True)
def _nw_serial(x_query, x, y, w, kernel_code, h, use_w):
    out = np.empty(x_query.shape[0], dtype=np.float64)
    for q in range(x_query.shape[0]):
        out[q] = _nw_at(x_query[q], x, y, w, kernel_code, h, use_w)
    return out


@njit(fastmath=False, cache=True, parallel=True)
def _nw_parallel(x_query, x, y, w, kernel_code, h, use_w):
    out = np.empty(x_query.shape[0], dtype=np.float64)
    for q in prange(x_query.shape[0]):
        out[q] = _nw_at(x_query[q], x, y, w, kernel_code, h, use_w)
    return out


def nadaraya_watson_smooth(
    x: np.ndarray,
    y: np.ndarray,
    *,
    x_query: np.ndarray | None = None,
    bandwidth: float = -1.0,
    kernel: str = "gaussian",
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Nadaraya-Watson kernel regression estimate of ``y`` as a function of ``x``.

    Parameters
    ----------
    x, y : np.ndarray
        1-D training inputs / targets.
    x_query : np.ndarray, optional
        Points to evaluate at. Defaults to ``x`` (in-sample smoothing / denoising).
    bandwidth : float
        Kernel bandwidth ``h``. ``<= 0`` -> Silverman rule from ``x``.
    kernel : {'gaussian', 'epanechnikov', 'boxcar', 'tricube'}
        Kernel shape. Compact kernels (all but gaussian) give exactly-zero weight beyond ``h``.
    sample_weight : np.ndarray, optional
        Per-sample multiplier on the kernel weight (compose kernel proximity with recency / analog weights).

    Returns
    -------
    np.ndarray
        NW estimate at each query point; NaN where no sample falls in the kernel support (compact kernels).

    Notes
    -----
    NW does not extrapolate meaningfully beyond the data range and is biased at boundaries; CV-tune the
    bandwidth for anything quantitative (lecture slide 13).
    """
    if kernel not in KERNELS:
        raise ValueError(f"nadaraya_watson_smooth: kernel must be one of {KERNELS}, got {kernel!r}.")
    x = np.ascontiguousarray(x, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("nadaraya_watson_smooth: x and y length mismatch.")
    if x.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    xq = x if x_query is None else np.ascontiguousarray(x_query, dtype=np.float64)
    h = float(bandwidth) if bandwidth > 0.0 else _silverman_bandwidth(x)
    use_w = sample_weight is not None
    if use_w:
        w = np.ascontiguousarray(sample_weight, dtype=np.float64)
        if w.shape[0] != x.shape[0]:
            raise ValueError("nadaraya_watson_smooth: sample_weight length mismatch.")
    else:
        w = np.ones(1, dtype=np.float64)
    kernel_code = KERNELS.index(kernel)
    if xq.shape[0] >= _NW_PARALLEL_MIN_QUERIES:
        return _nw_parallel(xq, x, y, w, kernel_code, h, use_w)
    return _nw_serial(xq, x, y, w, kernel_code, h, use_w)


@njit(fastmath=False, cache=True)
def _nw_per_group(v_sorted, x_sorted, starts, ends, kernel_code, bandwidth):
    """In-sample NW smoothing within each contiguous group; per-group Silverman bandwidth if ``bandwidth<=0``."""
    n = v_sorted.shape[0]
    out = np.empty(n, dtype=np.float64)
    w_dummy = np.ones(1, dtype=np.float64)
    for g in range(starts.shape[0]):
        s = starts[g]
        e = ends[g]
        m = e - s
        if m <= 0:
            continue
        if m == 1:
            out[s] = v_sorted[s]
            continue
        h = bandwidth if bandwidth > 0.0 else _silverman_bandwidth(x_sorted[s:e])
        for q in range(s, e):
            out[q] = _nw_at(x_sorted[q], x_sorted[s:e], v_sorted[s:e], w_dummy, kernel_code, h, False)
    return out


def per_group_nadaraya_watson_smooth(
    values: np.ndarray,
    group_ids: np.ndarray,
    *,
    order: np.ndarray | None = None,
    bandwidth: float = -1.0,
    kernel: str = "gaussian",
) -> np.ndarray:
    """Per-entity NW smoothing of ``values`` along ``order`` (e.g. denoise each road-arc's speed over time).

    Returns a smoothed value per original row (aligned to the input). ``bandwidth <= 0`` -> per-group Silverman.
    """
    if kernel not in KERNELS:
        raise ValueError(f"per_group_nadaraya_watson_smooth: kernel must be one of {KERNELS}, got {kernel!r}.")
    values = np.ascontiguousarray(values, dtype=np.float64)
    group_ids = np.ascontiguousarray(group_ids)
    n = values.shape[0]
    if group_ids.shape[0] != n:
        raise ValueError("per_group_nadaraya_watson_smooth: values and group_ids length mismatch.")
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if order is not None:
        order = np.ascontiguousarray(order, dtype=np.float64)
        if order.shape[0] != n:
            raise ValueError("per_group_nadaraya_watson_smooth: order length mismatch.")
        sort_idx = np.lexsort((order, group_ids))
        x_axis = order
    else:
        sort_idx = np.argsort(group_ids, kind="stable")
        x_axis = np.arange(n, dtype=np.float64)
    g_sorted = group_ids[sort_idx]
    v_sorted = values[sort_idx]
    x_sorted = np.ascontiguousarray(x_axis[sort_idx], dtype=np.float64)
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), bnd.astype(np.int64)))
    ends = np.concatenate((bnd.astype(np.int64), np.array([n], dtype=np.int64)))
    smoothed_sorted = _nw_per_group(v_sorted, x_sorted, starts, ends, KERNELS.index(kernel), float(bandwidth))
    out = np.empty(n, dtype=np.float64)
    out[sort_idx] = smoothed_sorted
    return out
