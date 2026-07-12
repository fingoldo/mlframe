"""Robust / order-statistic per-category target encodings for the K-fold encoder.

Companion to ``_target_encoding_fe`` (which computes the mean/std/skew/kurt moment stats via raw-moment ``np.bincount``).
Order statistics -- median, symmetric-trimmed mean, target quantiles (q10/q90), IQR, min, max of y within a category --
are NOT expressible from raw moments, so they need y SORTED within each category. This module provides that vectorised
grouped path: one ``np.lexsort`` groups rows by category and sorts y inside each group, then a single ``@numba.njit``
kernel sweeps the contiguous segments and emits every requested order stat -- no per-row / per-category Python loop.

Numerical-kernel note (why a dedicated njit kernel, not the numerical.py aggregate kernels): ``compute_simple_stats_numba``
cleanly returns per-array (min, max, ...) but also computes argmin/argmax/mean/std that the caller here discards, and it
is a Python-level call per category (500+ segments) -- both cost we avoid. ``compute_numaggs`` exposes quantiles only at a
fixed ``default_quantiles`` set / ``median_unbiased`` method and no symmetric-10%-trim mean, so it cannot emit the exact
q10/q90 (numpy ``linear`` method) + 10%-trimmed-mean this encoder specifies. The dedicated segment kernel computes exactly
the requested stats in one contiguous sweep, so we use it instead and reuse numpy/scipy for the GLOBAL fallbacks.

Sample-stability floors mirror ``_binned_numeric_agg_fe._N_MIN`` (mean:5 / std:12 / skew:30 / kurt:100): an order stat
from too few rows is noise (a median from n=1 is the single value), so a category with fewer rows than the stat's floor
falls back to the GLOBAL stat value -- the same rare-cell shrinkage discipline the smoothed mean uses, expressed as a hard
threshold because order stats are not additively blendable the way Micci-Barreca shrinks a mean.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numba
import numpy as np

# Order stats this module can emit + their minimum per-category sample size for a stable estimate.
ORDER_STATS = ("median", "trimmed_mean", "q10", "q90", "iqr", "min", "max")
ORDER_STAT_N_MIN = {"median": 8, "trimmed_mean": 12, "q10": 20, "q90": 20, "iqr": 20, "min": 8, "max": 8}
_TRIM_FRAC = 0.10  # symmetric fraction cut from each tail for trimmed_mean (scipy.stats.trim_mean convention).


@numba.njit(cache=True, fastmath=True)
def _segment_order_stats(
    y_sorted: np.ndarray, starts: np.ndarray, counts: np.ndarray,
    w_median: bool, w_trim: bool, w_q10: bool, w_q90: bool, w_iqr: bool, w_min: bool, w_max: bool,
    trim_frac: float,
):
    """Order stats per contiguous category segment of a within-category-sorted y. ``starts``/``counts`` index the segments
    (segment c is ``y_sorted[starts[c]:starts[c]+counts[c]]``, ascending). Empty categories -> NaN (caller replaces with the
    global fallback). Quantiles use numpy's ``linear`` interpolation; trimmed mean cuts ``floor(k*trim_frac)`` per tail."""
    n_cats = counts.shape[0]
    med = np.full(n_cats, np.nan); trm = np.full(n_cats, np.nan)
    q10 = np.full(n_cats, np.nan); q90 = np.full(n_cats, np.nan)
    iqr = np.full(n_cats, np.nan); mn = np.full(n_cats, np.nan); mx = np.full(n_cats, np.nan)
    for c in range(n_cats):
        k = counts[c]
        if k <= 0:
            continue
        s = starts[c]
        if w_min:
            mn[c] = y_sorted[s]
        if w_max:
            mx[c] = y_sorted[s + k - 1]
        if w_median:
            med[c] = _seg_quantile(y_sorted, s, k, 0.5)
        if w_q10 or w_iqr:
            v10 = _seg_quantile(y_sorted, s, k, 0.10)
            if w_q10:
                q10[c] = v10
        if w_q90 or w_iqr:
            v90 = _seg_quantile(y_sorted, s, k, 0.90)
            if w_q90:
                q90[c] = v90
        if w_iqr:
            iqr[c] = v90 - v10
        if w_trim:
            cut = int(k * trim_frac)
            lo = s + cut
            hi = s + k - cut
            if hi - lo <= 0:  # trimmed everything (only at tiny k below the floor) -> untrimmed mean
                lo, hi = s, s + k
            acc = 0.0
            for j in range(lo, hi):
                acc += y_sorted[j]
            trm[c] = acc / (hi - lo)
    return med, trm, q10, q90, iqr, mn, mx


@numba.njit(cache=True, fastmath=True)
def _seg_quantile(y_sorted: np.ndarray, start: int, k: int, p: float) -> float:
    """Quantile ``p`` of the ascending segment ``y_sorted[start:start+k]`` via numpy's ``linear`` interpolation."""
    if k == 1:
        return float(y_sorted[start])
    pos = (k - 1) * p
    lo = int(pos)
    frac = pos - lo
    v = y_sorted[start + lo]
    if frac > 0.0 and lo + 1 < k:
        v += frac * (y_sorted[start + lo + 1] - v)
    return float(v)


def per_category_order_stats(
    inverse: np.ndarray, y_arr: np.ndarray, n_cats: int, stats: Sequence[str], global_stats: dict,
    *, counts: Optional[np.ndarray] = None,
) -> dict:
    """Per-category order statistics with rare-cell fallback to the global value.

    Returns ``{stat: arr}`` of length ``n_cats`` for each requested ORDER stat. A category with fewer rows than the stat's
    ``ORDER_STAT_N_MIN`` floor (or zero rows) takes the global stat value -- rare cells never emit an unstable order stat.
    Vectorised: one ``np.lexsort`` sorts y within category, then the njit segment kernel emits every requested stat.
    ``counts`` lets a caller that already computed ``np.bincount(inverse, minlength=n_cats)`` (e.g. because moment
    stats are also being computed on the same ``inverse``) pass it in and skip the redundant recount."""
    wanted = [s for s in stats if s in ORDER_STATS]
    if not wanted:
        return {}
    counts = counts.astype(np.int64) if counts is not None else np.bincount(inverse, minlength=n_cats).astype(np.int64)
    # lexsort: primary key = category code (last), secondary = y -> rows grouped by category, y ascending within group.
    order = np.lexsort((y_arr, inverse))
    y_sorted = np.ascontiguousarray(y_arr[order])
    starts = np.zeros(n_cats, dtype=np.int64)
    if n_cats > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    med, trm, q10, q90, iqr, mn, mx = _segment_order_stats(
        y_sorted, starts, counts,
        "median" in wanted, "trimmed_mean" in wanted, "q10" in wanted, "q90" in wanted,
        "iqr" in wanted, "min" in wanted, "max" in wanted, _TRIM_FRAC,
    )
    raw_by_stat = {"median": med, "trimmed_mean": trm, "q10": q10, "q90": q90, "iqr": iqr, "min": mn, "max": mx}
    out: dict = {}
    for stat in wanted:
        raw = raw_by_stat[stat]
        g = float(global_stats[stat])
        floor = ORDER_STAT_N_MIN[stat]
        # Rare-cell + empty-cell fallback to the global stat; NaN from empty segments is also replaced.
        stable = (counts >= floor) & np.isfinite(raw)
        out[stat] = np.where(stable, raw, g)
    return out


def global_order_stats(y_arr: np.ndarray, stats: Sequence[str]) -> dict:
    """Global (all-rows) value of each requested order stat -- the rare-cell / unseen-category fallback.

    Uses numpy (``np.median`` / ``np.quantile`` linear method / ``np.min`` / ``np.max``) and ``scipy.stats.trim_mean`` for
    the symmetric-trimmed mean, matching the per-category kernel's definitions exactly."""
    wanted = [s for s in stats if s in ORDER_STATS]
    if not wanted:
        return {}
    g: dict = {}
    q10 = q90 = None
    if any(s in ("q10", "iqr") for s in wanted):
        q10 = float(np.quantile(y_arr, 0.10))
    if any(s in ("q90", "iqr") for s in wanted):
        q90 = float(np.quantile(y_arr, 0.90))
    for stat in wanted:
        if stat == "median":
            g[stat] = float(np.median(y_arr))
        elif stat == "trimmed_mean":
            from scipy.stats import trim_mean
            g[stat] = float(trim_mean(y_arr, _TRIM_FRAC)) if y_arr.size else 0.0
        elif stat == "q10":
            g[stat] = q10
        elif stat == "q90":
            g[stat] = q90
        elif stat == "iqr":
            assert q90 is not None and q10 is not None  # "iqr" wanted implies both were computed above
            g[stat] = q90 - q10
        elif stat == "min":
            g[stat] = float(np.min(y_arr))
        elif stat == "max":
            g[stat] = float(np.max(y_arr))
    return {k: (v if np.isfinite(v) else 0.0) for k, v in g.items()}
