"""Stages of ``_quantile_bin_njit``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from numba import njit
from mlframe.feature_selection.filters.discretization._multiselect import multiselect_inplace
# --- end imports ---


@njit(cache=True)
def _quantile_crossing_offsets(n_bins, cuts, crossing, n, x):
    """Crossing-run offsets of the equal values that straddle a bin boundary, for ``_quantile_bin_njit``."""
    canon = np.empty(n_bins - 1, dtype=np.int64)
    below = np.zeros(n_bins - 1, dtype=np.int64)
    seen = np.zeros(n_bins - 1, dtype=np.int64)
    for k in range(n_bins - 1):
        if k > 0 and cuts[k] == cuts[k - 1]:
            canon[k] = canon[k - 1]
        else:
            canon[k] = k
            if crossing[k] or (k + 1 < n_bins - 1 and cuts[k + 1] == cuts[k]):  # the run may open exactly at this cut and straddle the next one
                cnt = 0
                v0 = cuts[k]
                for r in range(n):
                    if x[r] < v0:
                        cnt += 1
                below[k] = cnt
    return below, canon, seen


@njit(cache=True)
def _quantile_crossing_cuts(n, n_bins, x):
    """Bin cut values and the boundary-crossing flags of an equi-frequency split, for ``_quantile_bin_njit``."""
    base = n // n_bins
    rem = n % n_bins
    ranks = np.empty(2 * (n_bins - 1), dtype=np.int64)
    starts = np.empty(n_bins, dtype=np.int64)
    pos = 0
    for b in range(n_bins):
        starts[b] = pos
        pos += base + (1 if b < rem else 0)
    t = 0
    for b in range(1, n_bins):
        ranks[t] = starts[b] - 1
        ranks[t + 1] = starts[b]
        t += 2
    # ranks are ascending; equal neighbours cannot occur because every bin holds at least eight rows
    work = x.copy()
    multiselect_inplace(work, ranks)
    cuts = np.empty(n_bins - 1, dtype=x.dtype)
    crossing = np.zeros(n_bins - 1, dtype=np.bool_)  # a run of equal values reaches across the boundary that opens this bin
    any_crossing = False
    for b in range(1, n_bins):
        cuts[b - 1] = work[starts[b]]
        if not (work[starts[b] - 1] < work[starts[b]]):
            crossing[b - 1] = True
            any_crossing = True
    return any_crossing, crossing, cuts, starts
