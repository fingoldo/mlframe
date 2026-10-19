"""Distinct partitions of a family of integer-coded candidates.

The family-wise tests (null maximum over every candidate, Bonferroni sum over every candidate) depend on the SET of distinct partitions the code columns induce: a column that induces the same
partition of the rows as another one (monotone transforms of the same operands share their rank codes, a decreasing one reverses them) has the same MI under every label permutation, so it can
neither raise the null maximum nor add an independent chance to exceed the observed value. Dropping the repeats before the test gives the same null maximum and a Bonferroni sum over the
distinct events.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry

__all__ = ["unique_columns"]


@njit(cache=True, parallel=True)
def _partition_hashes(cols: np.ndarray, n_codes: int) -> np.ndarray:
    """FNV-1a hash of the first-appearance relabelling of every row of ``cols`` ``(K, n)`` (rows with the same partition get the same hash)."""
    k, n = cols.shape
    out = np.empty(k, dtype=np.uint64)
    for j in prange(k):
        label = np.full(n_codes, -1, dtype=np.int64)
        nxt = 0
        h = np.uint64(14695981039346656037)
        for i in range(n):
            c = cols[j, i]
            if label[c] < 0:
                label[c] = nxt
                nxt += 1
            h = (h ^ np.uint64(label[c])) * np.uint64(1099511628211)
        out[j] = h
    return out


@njit(cache=True, parallel=True)
def _same_partition_as_representative(cols: np.ndarray, rep: np.ndarray, n_codes: int) -> bool:
    """True when row ``j`` induces the same partition as row ``rep[j]`` for every ``j`` (guards against a hash collision): the codes of the two rows map to each other one to one."""
    k, n = cols.shape
    bad = np.zeros(k, dtype=np.bool_)
    for j in prange(k):
        r = rep[j]
        if r == j:
            continue
        fwd = np.full(n_codes, -1, dtype=np.int64)
        back = np.full(n_codes, -1, dtype=np.int64)
        for i in range(n):
            a = cols[j, i]
            b = cols[r, i]
            if fwd[a] < 0 and back[b] < 0:
                fwd[a] = b
                back[b] = a
            elif fwd[a] != b or back[b] != a:
                bad[j] = True
                break
    return not bad.any()


def unique_columns(cols: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(keep, inverse)``: ``keep`` holds the first row of each group of rows inducing the same partition (in order of first appearance), ``inverse[j]`` is the position in ``keep`` of row ``j``'s group.

    ``None`` when every row has its own partition (or a hash collision makes the grouping unsafe)."""
    k = int(cols.shape[0])
    if k < 2:
        return None
    cols = np.ascontiguousarray(cols)
    n_codes = int(cols.max()) + 1
    with parallel_kernel_entry():
        hv = _partition_hashes(cols, n_codes)
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
    with parallel_kernel_entry():
        same = _same_partition_as_representative(cols, rep, n_codes)
    if not same:
        return None
    keep = np.flatnonzero(rep == np.arange(k))
    slot = np.full(k, -1, dtype=np.int64)
    slot[keep] = np.arange(keep.shape[0])
    return keep, slot[rep]
