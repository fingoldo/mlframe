"""Percentile rank within a group: the sorted linear quantiles, the nogil rank kernel and its numpy twin, and the CDF position of a value in a sorted group.

Carved out of ``_grouped_quantile_fe`` (which re-exports every name here) to keep that module under the line budget.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numba import njit
from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype


def _sorted_linear_quantiles(sv: np.ndarray, qs: Sequence[float]) -> np.ndarray:
    """``np.quantile(x, qs)`` (linear method, double precision result) computed from ``sv = np.sort(x)``: the same virtual index and the same lerp as numpy's
    ``_quantile``, so the values equal numpy's bit for bit (checked on 6000 random arrays, in single and double precision, with ties)."""
    n = sv.size
    # Intentional float64: np.quantile returns double precision and these values equal it bit for bit
    out = np.empty(len(qs), dtype=np.float64)
    for k, q in enumerate(qs):
        v = q * (n - 1)
        lo = int(np.floor(v))
        hi = min(lo + 1, n - 1)
        t = v - lo
        a = sv[lo]
        b = sv[hi]
        diff = b - a
        r = a + diff * t
        if t >= 0.5:
            r = b - diff * (1 - t)
        out[k] = r
    return out


@njit(cache=True, nogil=True)
def _member_pct_ranks_njit(order: np.ndarray, sv: np.ndarray, ranks: np.ndarray) -> None:
    """Fill ``ranks[order[k]]`` with the average rank of the run of equal values around sorted position ``k``, as the numpy twin below does.

    One pass instead of its eight (concatenate, compare, flatnonzero, diff, two repeats, a fancy assignment), and it releases the GIL so a lane thread does not hold it for every group of
    every numeric column. The arithmetic is the twin's: the run bounds are integers and ``(lo + hi) / (2.0 * m)`` is one float64 division, so the values are bit-identical.
    """
    m = sv.shape[0]
    denom = 2.0 * m
    i = 0
    while i < m:
        j = i + 1
        while j < m and sv[j] == sv[i]:
            j += 1
        r = (i + j) / denom
        for k in range(i, j):
            ranks[order[k]] = r
        i = j


def _member_pct_ranks(order: np.ndarray, sv: np.ndarray, vals: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """:func:`_pct_rank_in_sorted` of the group's own values ``vals`` against ``sv = vals[finite][order]``, from the sort permutation: a member's left and right insertion ranks are the start and end of its run of
    equal values (ties get the average rank), so no search is needed. A non-finite value is ranked by the search, as before (NaN and +inf at the end, -inf at the start)."""
    m = sv.size
    ranks = np.empty(m, dtype=np.float64)  # Intentional float64: rank positions, as the searchsorted midpoint is formed
    if m:
        _member_pct_ranks_njit(np.ascontiguousarray(order, dtype=np.int64), np.ascontiguousarray(sv), ranks)
    out = np.empty(vals.shape, dtype=np.float64)  # Intentional float64: rank positions, as the searchsorted midpoint is formed
    out[finite] = ranks
    if not finite.all():
        out[~finite] = _pct_rank_in_sorted(sv, vals[~finite])
    return out


def _member_pct_ranks_numpy(order: np.ndarray, sv: np.ndarray, vals: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """The numpy twin of :func:`_member_pct_ranks`, kept as the reference the equality test compares the kernel against."""
    m = sv.size
    starts = np.flatnonzero(np.concatenate(([True], sv[1:] != sv[:-1])))
    counts = np.diff(np.concatenate((starts, [m])))
    lo = np.repeat(starts, counts)
    hi = np.repeat(starts + counts, counts)
    ranks = np.empty(m, dtype=np.float64)  # Intentional float64: rank positions, as the searchsorted midpoint is formed
    ranks[order] = (lo + hi) / (2.0 * m)
    out = np.empty(vals.shape, dtype=np.float64)  # Intentional float64: rank positions, as the searchsorted midpoint is formed
    out[finite] = ranks
    if not finite.all():
        out[~finite] = _pct_rank_in_sorted(sv, vals[~finite])
    return out


def _pct_rank_in_sorted(sorted_vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Empirical CDF position of each ``x`` in ``sorted_vals`` - the fraction
    of group values ``<= x`` via the midpoint of the left / right insertion
    ranks (ties get their average rank, matching ``rank(pct=True)`` semantics).
    Returns values in ``[0, 1]``; an empty reference yields 0.5 (neutral).
    """
    m = sorted_vals.size
    if m == 0:
        return np.full(x.shape, 0.5, dtype=_crit_np_dtype())
    lo = np.searchsorted(sorted_vals, x, side="left")
    # ``hi`` differs from ``lo`` only for a value present in ``sorted_vals``: one more than ``lo`` when it occurs once, a second search only for the tied values
    at = np.minimum(lo, m - 1)
    present = sorted_vals[at] == x
    hi = lo.copy()
    nxt = sorted_vals[np.minimum(lo + 1, m - 1)]
    single = present & ((lo + 1 >= m) | (nxt != x))
    hi[single] = lo[single] + 1
    tied = present & ~single
    if tied.any():
        hi[tied] = np.searchsorted(sorted_vals, x[tied], side="right")
    # Average rank for ties; normalise by m so the result is a CDF position.
    rank: np.ndarray = (lo + hi) / (2.0 * m)
    return rank
