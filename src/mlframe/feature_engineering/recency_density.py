"""Per-entity recency-weighted Parzen density: mode prediction + behavioral-stability score.

Two lecture ideas in one pass over each entity's ordered value history:

1. **Weighted Parzen density / mode prediction** (slides 21-27): the best point
   prediction of an entity's next value, assuming it behaves as before, is a
   readout of its own recency-weighted 1-D kernel density. We support two
   readouts: ``mean`` (recency-weighted mean, the smooth estimate) and ``mode``
   (argmax of the recency-weighted Gaussian-KDE, robust to a skewed multi-modal
   spend distribution where the mean sits in an empty valley).

2. **Behavioral stability** (slides 30-31): the HEIGHT of the density peak is a
   per-entity confidence score. A spiky, concentrated history (peak height high)
   means the entity is predictable; a flat, spread-out history (low peak) means
   it is not. This score is a useful feature on its own and a natural gate/weight
   for downstream ensembling.

Weights come from :func:`mlframe.core.recency_weights.recency_weights`; at the
identity parameter the density is the ordinary (uniform) Parzen estimate.
"""

from __future__ import annotations

import logging

import os

import numpy as np
from numba import njit, prange

from mlframe.core.recency_weights import SCHEMES

logger = logging.getLogger(__name__)

__all__ = ["per_group_recency_weighted_mode", "per_group_behavioral_stability"]

# Parallelize the per-entity KDE over groups once there are enough of them to amortise the prange spawn (~50us).
# Crossover measured in _benchmarks/profile_recency_features.py; env-overridable per host.
_KDE_PARALLEL_MIN_GROUPS = int(os.environ.get("MLFRAME_RECENCY_KDE_PARALLEL_MIN_GROUPS", "512"))


@njit(fastmath=False, cache=True)
def _recency_weight(m: int, pos: int, scheme_code: int, param: float) -> float:
    """Single oldest-first recency weight for position ``pos`` in a length-``m`` history (i = m-pos)."""
    i = m - pos
    if scheme_code == 0:
        return float(((m - i + 1) / m) ** param)
    elif scheme_code == 1:
        return float(param**i)
    return float(1.0 / (i**param))


@njit(fastmath=False, cache=True, inline="always")
def _kde_one_group(v_sorted, s, e, scheme_code, param, bandwidth, n_grid, h_span_frac):
    """Recency-weighted Gaussian-KDE mode + peak height for a single contiguous, oldest-first group ``v_sorted[s:e]``.

    Bandwidth selection (in priority order): ``h_span_frac > 0`` -> ``h = h_span_frac * (max-min)`` (scale-relative,
    used for the stability peak so concentration is comparable across entities); else ``bandwidth > 0`` -> that
    absolute value; else a per-group Silverman-like rule from the value spread (used for mode readout).
    peak_height is the max weighted density normalized by sum of weights (in [0, 1]); a spiky history -> near 1.
    """
    m = e - s
    if m <= 0:
        return np.nan, np.nan
    if m == 1:
        return v_sorted[s], 1.0
    vmin = v_sorted[s]
    vmax = v_sorted[s]
    mean = 0.0
    for k in range(s, e):
        val = v_sorted[k]
        if val < vmin:
            vmin = val
        if val > vmax:
            vmax = val
        mean += val
    mean /= m
    span = vmax - vmin
    if h_span_frac > 0.0:
        h = h_span_frac * span if span > 0.0 else 1.0
    elif bandwidth > 0.0:
        h = bandwidth
    else:
        var = 0.0
        for k in range(s, e):
            d = v_sorted[k] - mean
            var += d * d
        std = (var / m) ** 0.5
        if std <= 0.0:
            std = span if span > 0.0 else 1.0
        h = 1.06 * std * (m ** (-0.2))
    if h <= 0.0:
        h = 1.0
    w_sum = 0.0
    for pos in range(m):
        w_sum += _recency_weight(m, pos, scheme_code, param)
    if w_sum <= 0.0:
        w_sum = 1.0
    if span <= 0.0:
        return vmin, 1.0
    best_x = vmin
    best_d = -1.0
    inv2h2 = 1.0 / (2.0 * h * h)
    for gi in range(n_grid):
        x = vmin + span * (gi / (n_grid - 1))
        dens = 0.0
        for pos in range(m):
            w = _recency_weight(m, pos, scheme_code, param)
            diff = x - v_sorted[s + pos]
            dens += w * np.exp(-diff * diff * inv2h2)
        dens /= w_sum
        if dens > best_d:
            best_d = dens
            best_x = x
    return best_x, best_d


@njit(fastmath=False, cache=True)
def _weighted_kde_mode_and_peak(v_sorted, starts, ends, scheme_code, param, bandwidth, n_grid, h_span_frac):
    """Serial per-group KDE mode + peak over all groups."""
    n_groups = starts.shape[0]
    modes = np.empty(n_groups, dtype=np.float64)
    peaks = np.empty(n_groups, dtype=np.float64)
    for g in range(n_groups):
        modes[g], peaks[g] = _kde_one_group(v_sorted, starts[g], ends[g], scheme_code, param, bandwidth, n_grid, h_span_frac)
    return modes, peaks


@njit(fastmath=False, cache=True, parallel=True)
def _weighted_kde_mode_and_peak_parallel(v_sorted, starts, ends, scheme_code, param, bandwidth, n_grid, h_span_frac):
    """Parallel (prange over entities) variant; wins once n_groups amortises the spawn cost."""
    n_groups = starts.shape[0]
    modes = np.empty(n_groups, dtype=np.float64)
    peaks = np.empty(n_groups, dtype=np.float64)
    for g in prange(n_groups):
        modes[g], peaks[g] = _kde_one_group(v_sorted, starts[g], ends[g], scheme_code, param, bandwidth, n_grid, h_span_frac)
    return modes, peaks


def _dispatch_kde(v_sorted, starts, ends, scheme_code, param, bandwidth, n_grid, h_span_frac):
    """Pick serial vs prange by group count (the parallelizable axis)."""
    if starts.shape[0] >= _KDE_PARALLEL_MIN_GROUPS:
        return _weighted_kde_mode_and_peak_parallel(v_sorted, starts, ends, scheme_code, param, bandwidth, n_grid, h_span_frac)
    return _weighted_kde_mode_and_peak(v_sorted, starts, ends, scheme_code, param, bandwidth, n_grid, h_span_frac)


def _sort_into_groups(values, group_ids, order):
    """Stable-sort values/group_ids by (group_ids, order) so each group occupies a contiguous slice, returning the sorted arrays plus per-group [start, end) boundaries the KDE kernels iterate over."""
    values = np.ascontiguousarray(values, dtype=np.float64)
    group_ids = np.ascontiguousarray(group_ids)
    n = values.shape[0]
    if group_ids.shape[0] != n:
        raise ValueError("length mismatch between values and group_ids.")
    if order is not None:
        order = np.ascontiguousarray(order)
        if order.shape[0] != n:
            raise ValueError("order length mismatch.")
        sort_idx = np.lexsort((order, group_ids))
    else:
        sort_idx = np.argsort(group_ids, kind="stable")
    g_sorted = group_ids[sort_idx]
    v_sorted = values[sort_idx]
    bnd = np.where(g_sorted[1:] != g_sorted[:-1])[0] + 1
    starts = np.concatenate((np.array([0], dtype=np.int64), bnd.astype(np.int64)))
    ends = np.concatenate((bnd.astype(np.int64), np.array([n], dtype=np.int64)))
    return v_sorted, sort_idx, starts, ends, n


def _scatter(per_group, sort_idx, starts, ends, n, broadcast):
    """When broadcast is requested, expand each group's scalar result back to one value per original row (undoing the sort from `_sort_into_groups`); otherwise pass the per-group array through unchanged."""
    if not broadcast:
        return per_group
    # Vectorized: expand each group's value to its (sorted-order) rows via repeat, then invert the sort.
    out_sorted = np.repeat(per_group, ends - starts)
    out = np.empty(n, dtype=np.float64)
    out[sort_idx] = out_sorted
    return out


def per_group_recency_weighted_mode(
    values,
    group_ids,
    *,
    order=None,
    scheme="poly",
    param=1.0,
    bandwidth=-1.0,
    n_grid=64,
    broadcast=True,
):
    """Per-entity mode of the recency-weighted Gaussian Parzen density of ``values``.

    ``bandwidth <= 0`` picks a per-group Silverman rule. Returns shape ``(n,)`` if broadcast else one per entity.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}.")
    v_sorted, sort_idx, starts, ends, n = _sort_into_groups(values, group_ids, order)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    modes, _ = _dispatch_kde(v_sorted, starts, ends, SCHEMES.index(scheme), float(param), float(bandwidth), int(n_grid), 0.0)
    return _scatter(modes, sort_idx, starts, ends, n, broadcast)


def per_group_behavioral_stability(
    values,
    group_ids,
    *,
    order=None,
    scheme="poly",
    param=1.0,
    stability_span_frac=0.15,
    n_grid=64,
    broadcast=True,
):
    """Per-entity behavioral-stability score = peak height of the recency-weighted density (higher = more predictable)."""
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}.")
    v_sorted, sort_idx, starts, ends, n = _sort_into_groups(values, group_ids, order)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    _, peaks = _dispatch_kde(v_sorted, starts, ends, SCHEMES.index(scheme), float(param), -1.0, int(n_grid), float(stability_span_frac))
    return _scatter(peaks, sort_idx, starts, ends, n, broadcast)
