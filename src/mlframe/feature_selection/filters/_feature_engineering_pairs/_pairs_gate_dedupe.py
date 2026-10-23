"""The noise gate scores each distinct code column once.

The pair search quantile-bins every candidate, and monotone-equivalent transforms of the same operands (``exp(a)``, ``sqr(a)``, ``a`` for a column in (0, 1)) give byte-identical codes. In the large
calls of a fit about half of the 4000 to 16000 candidate columns are copies of another column, and the permutation test (99 shuffles over every row) ran for each copy.

The gate's result for a column does not depend on the other columns: the observed MI and the failure count use the column's own codes and the label permutations, which are drawn once for the
call. So a column equal to an earlier one, byte for byte, has the earlier one's result; the kernel runs on the distinct columns and the results are copied back. Equality is confirmed on the codes
themselves, so a fingerprint collision can only cost a missed merge, never a wrong result.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry

# Below this many candidate columns the fingerprint pass and the gather cost more than the copies save.
MIN_COLUMNS_TO_DEDUPE = 256
_COLUMN_CHUNK = 256


@njit(parallel=True, cache=True)
def _column_fingerprints(codes: np.ndarray, mul_a: np.ndarray, mul_b: np.ndarray) -> np.ndarray:
    """Two independent 64-bit polynomial fingerprints per column of the ``(n, K)`` code matrix, one row-major pass per chunk of columns."""
    n, k = codes.shape
    out = np.zeros((k, 2), dtype=np.uint64)
    n_chunks = (k + _COLUMN_CHUNK - 1) // _COLUMN_CHUNK
    for ch in prange(n_chunks):
        lo = ch * _COLUMN_CHUNK
        hi = min(k, lo + _COLUMN_CHUNK)
        for r in range(n):
            ma = mul_a[r]
            mb = mul_b[r]
            for j in range(lo, hi):
                v = np.uint64(codes[r, j]) + np.uint64(1)
                out[j, 0] += v * ma
                out[j, 1] += v * mb
    return out


def distinct_columns(codes: np.ndarray, fingerprints: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(keep, inverse)``: the first column of each group of byte-identical columns (ascending) and, for every column, its row in ``keep``; ``None`` when no column repeats."""
    n, k = codes.shape
    if fingerprints is None:
        rng = np.random.default_rng(0)
        mul_a = rng.integers(1, 2**62, n, dtype=np.uint64) | np.uint64(1)
        mul_b = rng.integers(1, 2**62, n, dtype=np.uint64) | np.uint64(1)
        with parallel_kernel_entry():
            fingerprints = _column_fingerprints(codes, mul_a, mul_b)
    order = np.lexsort((fingerprints[:, 1], fingerprints[:, 0]))
    sorted_fp = fingerprints[order]
    boundary = np.ones(k, dtype=bool)
    boundary[1:] = np.any(sorted_fp[1:] != sorted_fp[:-1], axis=1)
    if boundary.all():
        return None
    first_of = np.arange(k)
    group_start = 0
    for pos in range(1, k + 1):
        if pos == k or boundary[pos]:
            group = order[group_start:pos]
            if group.size > 1:
                leader = int(group.min())
                lead_col = codes[:, leader]
                for member in group:
                    first_of[member] = leader if np.array_equal(codes[:, member], lead_col) else member
                # a fingerprint collision left members unmerged with their own index; each is its own leader
            group_start = pos
    keep = np.flatnonzero(first_of == np.arange(k))
    if keep.size == k:
        return None
    row_of = np.full(k, -1, dtype=np.int64)
    row_of[keep] = np.arange(keep.size)
    return keep, row_of[first_of]


def gate_on_distinct_columns(kernel: Callable[..., np.ndarray], disc_2d: np.ndarray, factors_nbins: Any, **kwargs: Any) -> np.ndarray:
    """``kernel(disc_2d=..., factors_nbins=..., **kwargs)`` evaluated on the distinct columns only, expanded back to every column."""
    k = int(disc_2d.shape[1])
    distinct = distinct_columns(disc_2d) if k >= MIN_COLUMNS_TO_DEDUPE else None
    if distinct is None:
        return np.asarray(kernel(disc_2d=disc_2d, factors_nbins=factors_nbins, **kwargs))
    keep, inverse = distinct
    result = np.asarray(kernel(disc_2d=np.ascontiguousarray(disc_2d[:, keep]), factors_nbins=np.asarray(factors_nbins)[keep], **kwargs))
    expanded: np.ndarray = result[inverse]
    return expanded


__all__ = ["MIN_COLUMNS_TO_DEDUPE", "distinct_columns", "gate_on_distinct_columns"]
