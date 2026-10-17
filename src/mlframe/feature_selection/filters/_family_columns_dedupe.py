"""Distinct columns of a family of integer-coded candidates.

The family-wise tests (null maximum over every candidate, Bonferroni sum over every candidate) depend on the SET of distinct code columns: a column that repeats another one exactly (monotone
transforms of the same operands share their rank codes) has the same MI under every label permutation, so it can neither raise the null maximum nor add an independent chance to exceed the
observed value. Dropping the repeats before the test gives the same null maximum and a Bonferroni sum over the distinct events.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

__all__ = ["unique_columns"]


@njit(cache=True, parallel=True)
def _column_hashes(cols: np.ndarray) -> np.ndarray:
    """FNV-1a hash of every row of ``cols`` ``(K, n)`` (one candidate per row)."""
    k, n = cols.shape
    out = np.empty(k, dtype=np.uint64)
    for j in prange(k):
        h = np.uint64(14695981039346656037)
        for i in range(n):
            h = (h ^ np.uint64(cols[j, i])) * np.uint64(1099511628211)
        out[j] = h
    return out


@njit(cache=True, parallel=True)
def _all_equal_to_representative(cols: np.ndarray, rep: np.ndarray) -> bool:
    """True when row ``j`` of ``cols`` equals row ``rep[j]`` for every ``j`` (guards against a hash collision)."""
    k, n = cols.shape
    bad = np.zeros(k, dtype=np.bool_)
    for j in prange(k):
        r = rep[j]
        if r == j:
            continue
        for i in range(n):
            if cols[j, i] != cols[r, i]:
                bad[j] = True
                break
    return not bad.any()


def unique_columns(cols: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(keep, inverse)`` with ``cols[keep][inverse] == cols`` row for row, or ``None`` when every row is distinct (or a hash collision makes the grouping unsafe).

    ``keep`` holds the first row of each group of equal rows, in order of first appearance."""
    k = int(cols.shape[0])
    if k < 2:
        return None
    hv = _column_hashes(np.ascontiguousarray(cols))
    order = np.argsort(hv, kind="stable")
    hs = hv[order]
    new_group = np.empty(k, dtype=np.bool_)
    new_group[0] = True
    new_group[1:] = hs[1:] != hs[:-1]
    if bool(new_group.all()):
        return None
    group_of_sorted = np.cumsum(new_group) - 1
    first_row_of_group = order[new_group]  # stable sort: the first row (lowest index) of each hash group
    rep = np.empty(k, dtype=np.int64)
    rep[order] = first_row_of_group[group_of_sorted]
    if not _all_equal_to_representative(np.ascontiguousarray(cols), rep):
        return None
    keep = np.flatnonzero(rep == np.arange(k))
    slot = np.full(k, -1, dtype=np.int64)
    slot[keep] = np.arange(keep.shape[0])
    return keep, slot[rep]
