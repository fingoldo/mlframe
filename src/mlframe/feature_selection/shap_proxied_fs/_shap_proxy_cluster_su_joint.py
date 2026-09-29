"""Pairwise joint-count kernels for the SU clustering backend, bounded in memory for high-cardinality columns.

A pair's joint table has ``nb_i * nb_j`` cells. For ordinary quantile-binned columns that is ~100 cells, but a
high-cardinality categorical (an ID-like column with ~n distinct levels) makes it ~n^2 cells: 555k levels means a
2.5 TB table. Such pairs are counted sparsely instead - encode each row as the int64 key ``a * nb_j + b``, sort, and
run-length count - which costs O(n) memory and O(n log n) time regardless of cardinality.

Sorted keys visit the occupied cells in exactly the row-major ``(a, b)`` order the dense sweep does, so the MI
accumulates the same terms in the same order: the sparse path is bit-identical to the dense one, and the choice
between them is purely a memory/speed decision.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

DENSE_JOINT_MIN_CELLS: int = 1 << 16
"""Floor of the dense-table cap. The cap itself is ``max(DENSE_JOINT_MIN_CELLS, DENSE_JOINT_CELLS_PER_ROW * n_samples)``
cells, keeping per-thread scratch O(n_samples) like the sparse path's own key buffer."""

DENSE_JOINT_CELLS_PER_ROW: int = 4
"""Dense counting stays faster than the sort well past ``n`` cells (3.4x at 3.5M cells / n=556k), so the cap is set by
memory, not speed: 4 int64 cells per row is 32 bytes/row per thread."""

SPARSE_CODE_RELABEL_MIN: int = 1024
"""Columns whose largest code exceeds this are relabelled to dense codes before the pair scan; below it the relabel
pass costs more than the empty bins it would remove."""


def dense_relabel_codes(arr: np.ndarray, out: np.ndarray) -> int:
    """Write the monotone dense relabel of non-negative codes ``arr`` into int32 ``out``; return the distinct count.

    Uses a bincount lookup table when the code space is within a small multiple of ``len(arr)`` (O(n + max) time and
    memory), else ``np.unique`` (O(n log n)), so a code space in the billions never allocates a table that large.
    """
    n = int(arr.shape[0])
    if n == 0:
        return 0
    code_max = int(arr.max())
    if code_max <= 4 * n + 1024:
        present = np.bincount(arr, minlength=code_max + 1) > 0
        lut = np.cumsum(present, dtype=np.int64) - 1
        out[:] = lut[arr]
        return int(lut[-1]) + 1
    uniq, inverse = np.unique(arr, return_inverse=True)
    out[:] = inverse.reshape(-1)
    return int(uniq.shape[0])


@njit(nogil=True, cache=True, fastmath=False)
def _dense_cap_cells(n_samples: int, dense_max_cells: int) -> int:
    """Largest ``nb_i * nb_j`` table counted densely; a non-negative ``dense_max_cells`` overrides the default."""
    if dense_max_cells >= 0:
        return dense_max_cells
    cap = DENSE_JOINT_CELLS_PER_ROW * n_samples
    return cap if cap > DENSE_JOINT_MIN_CELLS else DENSE_JOINT_MIN_CELLS


@njit(nogil=True, cache=True, fastmath=False, inline="always")
def _pair_mi_dense(
    row_i: np.ndarray,
    row_j: np.ndarray,
    nb_i: int,
    nb_j: int,
    freqs_packed: np.ndarray,
    off_i: int,
    off_j: int,
    joint: np.ndarray,
    inv_n: float,
) -> float:
    """Plug-in MI of one pair from a flat ``nb_i * nb_j`` joint table held in the caller's reusable ``joint`` buffer."""
    for c in range(nb_i * nb_j):
        joint[c] = 0
    for k in range(row_i.shape[0]):
        joint[row_i[k] * nb_j + row_j[k]] += 1
    mi = 0.0
    for a in range(nb_i):
        px = freqs_packed[off_i + a]
        if px <= 0.0:
            continue
        base = a * nb_j
        for b in range(nb_j):
            jc = joint[base + b]
            if jc == 0:
                continue
            py = freqs_packed[off_j + b]
            if py <= 0.0:
                continue
            jf = jc * inv_n
            mi += jf * math.log(jf / (px * py))
    return mi


@njit(nogil=True, cache=True, fastmath=False, inline="always")
def _pair_mi_sparse(
    row_i: np.ndarray,
    row_j: np.ndarray,
    nb_j: int,
    freqs_packed: np.ndarray,
    off_i: int,
    off_j: int,
    keys: np.ndarray,
    inv_n: float,
) -> float:
    """Plug-in MI of one pair by sorting int64 pair keys; ``keys`` is caller scratch of length ``len(row_i)``."""
    n = row_i.shape[0]
    if n == 0:
        return 0.0
    for k in range(n):
        keys[k] = np.int64(row_i[k]) * nb_j + row_j[k]
    keys.sort()
    mi = 0.0
    cur = keys[0]
    cnt = 1
    for k in range(1, n + 1):
        if k < n and keys[k] == cur:
            cnt += 1
            continue
        a = cur // nb_j
        b = cur - a * nb_j
        px = freqs_packed[off_i + a]
        py = freqs_packed[off_j + b]
        if px > 0.0 and py > 0.0:
            jf = cnt * inv_n
            mi += jf * math.log(jf / (px * py))
        if k < n:
            cur = keys[k]
            cnt = 1
    return mi


@njit(parallel=True, nogil=True, cache=True, fastmath=False)
def _pairwise_su_edges(
    bins_packed: np.ndarray,
    nbins: np.ndarray,
    freqs_packed: np.ndarray,
    freqs_offsets: np.ndarray,
    h_marginals: np.ndarray,
    constant_mask: np.ndarray,
    threshold: float,
    dense_max_cells: int = -1,
) -> np.ndarray:
    """Pairwise SU matrix above ``threshold`` returned as a dense flag matrix.

    ``bins_packed`` is a ``(n_features, n_samples)`` int32 view (each feature's per-sample bin ids occupy a contiguous
    row, so the inner sample-scan reads two contiguous int32 strips); ``nbins[i]`` is column ``i``'s cardinality;
    ``freqs_packed`` is the concatenation of all per-column marginal probability vectors with offsets in
    ``freqs_offsets`` (shape ``(n_features + 1,)``); ``h_marginals[i]`` is column ``i``'s Shannon entropy;
    ``constant_mask[i]`` is ``True`` when column ``i`` has <=1 distinct bin (SU=0 vs anyone).

    A pair whose ``nb_i * nb_j`` exceeds the dense cap (``dense_max_cells``; negative means
    ``max(DENSE_JOINT_MIN_CELLS, DENSE_JOINT_CELLS_PER_ROW * n_samples)``) is counted sparsely, so per-thread scratch is O(n_samples) however
    high the cardinality. Sizing one ``max_nb x max_nb`` buffer for every pair made a single ~555k-level column
    request terabytes: a MemoryError on Windows, and on Linux's omp layer a silently swallowed worker exception
    that returned an all-zero flag matrix.

    Returns an upper-triangle flag matrix (``flag[i, j] = 1`` iff ``SU(i, j) >= threshold`` and ``i < j``).

    bench-attempt-rejected: j-tile block of B consecutive j-columns sharing the i-row L1 read. Every B in
    {2..32} regressed at width=2000 / n_samples=1500 / n_bins=10 (best 0.82x), likely from B strided joint stores
    saturating L1. Do not re-attempt pure j-tiling without changing the inner-loop store pattern.
    """
    n_features, n_samples = bins_packed.shape
    flags = np.zeros((n_features, n_features), dtype=np.uint8)
    if n_samples == 0:
        return flags
    cap = _dense_cap_cells(n_samples, dense_max_cells)
    inv_n = 1.0 / n_samples
    for i in prange(n_features):
        if constant_mask[i]:
            continue
        nb_i = nbins[i]
        off_i = freqs_offsets[i]
        # Size this row's scratch from the pairs it will actually see: the largest dense-eligible table, plus the
        # sparse key buffer only if some partner overflows the cap.
        dense_cells = 1
        need_sparse = False
        for j in range(i + 1, n_features):
            if constant_mask[j]:
                continue
            cells = nb_i * nbins[j]
            if cells > cap:
                need_sparse = True
            elif cells > dense_cells:
                dense_cells = cells
        joint = np.zeros(dense_cells, dtype=np.int64)
        keys = np.empty(n_samples if need_sparse else 0, dtype=np.int64)
        row_i = bins_packed[i]
        h_i = h_marginals[i]
        for j in range(i + 1, n_features):
            if constant_mask[j]:
                continue
            nb_j = nbins[j]
            if nb_i * nb_j > cap:
                mi = _pair_mi_sparse(row_i, bins_packed[j], nb_j, freqs_packed, off_i, freqs_offsets[j], keys, inv_n)
            else:
                mi = _pair_mi_dense(row_i, bins_packed[j], nb_i, nb_j, freqs_packed, off_i, freqs_offsets[j], joint, inv_n)
            denom = h_i + h_marginals[j]
            if denom <= 1e-12:
                continue
            su = 2.0 * mi / denom
            if su >= threshold:
                flags[i, j] = 1
    return flags


@njit(nogil=True, cache=True, fastmath=False)
def su_from_classes_sparse(classes_x: np.ndarray, freqs_x: np.ndarray, classes_y: np.ndarray, freqs_y: np.ndarray) -> float:
    """Sparse-joint twin of ``compute_su_from_classes`` for pairs whose dense ``K_x * K_y`` table would not fit.

    Same formula, the same zero-denominator and negative-MI floors, and the same summation order, so it returns the
    identical float.
    """
    n = classes_x.shape[0]
    if n == 0:
        return 0.0
    k_y = freqs_y.shape[0]
    freqs = np.empty(freqs_x.shape[0] + k_y, dtype=np.float64)
    freqs[: freqs_x.shape[0]] = freqs_x
    freqs[freqs_x.shape[0] :] = freqs_y
    keys = np.empty(n, dtype=np.int64)
    mi = _pair_mi_sparse(classes_x, classes_y, k_y, freqs, 0, freqs_x.shape[0], keys, 1.0 / n)
    h_x = 0.0
    for p in freqs_x:
        if p > 0:
            h_x -= p * math.log(p)
    h_y = 0.0
    for p in freqs_y:
        if p > 0:
            h_y -= p * math.log(p)
    denom = h_x + h_y
    if denom <= 1e-12:
        return 0.0
    if mi < 0.0:
        mi = 0.0
    return float(2.0 * mi / denom)


__all__ = ["DENSE_JOINT_CELLS_PER_ROW", "DENSE_JOINT_MIN_CELLS", "SPARSE_CODE_RELABEL_MIN", "_pairwise_su_edges", "dense_relabel_codes", "su_from_classes_sparse"]
