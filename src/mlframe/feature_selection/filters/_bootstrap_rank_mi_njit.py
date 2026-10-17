"""Rank-binned plug-in MI of bootstrap resamples without gathering or re-sorting the resample.

The bootstrap scorers (modular, decomposition, orthogonal) draw B resamples of the rows (``idx``: row indices with repeats, in draw order) and bin every candidate column into
equi-frequency RANK bins of each resample (``_plugin_mi_classif_njit``): gather ``(sample_n, k)`` values, stable-argsort every column, cut the sorted positions into ``n_bins`` slices of
``m // n_bins`` (+1 for the first ``m % n_bins``), count ``(bin, class)`` cells. Equal values are ordered by their position in the resample (stable sort).

Here each column is ranked ONCE (dense rank of its value: equal values share a rank). A resample's stable sort is then a counting sort of its draws by rank: the position of a draw is the number
of draws with a smaller rank plus the earlier draws of the same rank, which is exactly the position the stable argsort of the gathered values assigns. So the cells are counted
without gathering the values and without sorting a resample, and the MI equals the gather-and-argsort path bit for bit for every column, ties included.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

__all__ = ["bootstrap_rank_mi"]


@njit(cache=True)
def _mi_of_hist(hist: np.ndarray, m: int) -> float:
    """Plug-in MI (nats) of the ``(bins, classes)`` count table ``hist`` of ``m`` rows, summed as ``_plugin_mi_classif_njit`` sums it."""
    n_bins, n_classes = hist.shape
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for bb in range(n_bins):
        for c in range(n_classes):
            hist_x[bb] += hist[bb, c]
            hist_y[c] += hist[bb, c]
    log_n = math.log(m)
    mi = 0.0
    for bb in range(n_bins):
        if hist_x[bb] == 0:
            continue
        log_hx = math.log(hist_x[bb])
        for c in range(n_classes):
            n_xy = hist[bb, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / m) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi


@njit(cache=True, fastmath=True)
def _replicate_mis(rk: np.ndarray, n_blocks: int, y_codes: np.ndarray, n_classes: int, idx: np.ndarray, lens: np.ndarray, binpos: np.ndarray, n_bins: int, out: np.ndarray, col: int) -> None:
    """``out[b, col]`` = MI of replicate ``b`` for a column whose rows have the dense ranks ``rk`` (counting sort of the draws by rank, stable in draw order)."""
    n_rep = idx.shape[0]
    start = np.zeros(n_blocks + 1, dtype=np.int64)
    hist = np.zeros((n_bins, n_classes), dtype=np.int64)
    for b in range(n_rep):
        m = lens[b]
        start[:] = 0
        hist[:, :] = 0
        for t in range(m):
            start[rk[idx[b, t]] + 1] += 1
        for blk in range(n_blocks):
            start[blk + 1] += start[blk]
        for t in range(m):
            r = idx[b, t]
            blk = rk[r]
            q = start[blk]
            start[blk] = q + 1
            hist[binpos[b, q], y_codes[r]] += 1
        out[b, col] = _mi_of_hist(hist, m)


@njit(cache=True, fastmath=True, parallel=True)
def _bootstrap_rank_mi_f32(bits: np.ndarray, y_codes: np.ndarray, n_classes: int, idx: np.ndarray, lens: np.ndarray, binpos: np.ndarray, n_bins: int):
    """``(mi (B, k), ok (k,))`` for a float32 matrix given as its raw ``uint32`` bits ``(n, k)``; ``ok`` is False (and the MI column 0) for a column with a non-finite value.

    The dense ranks come from one sort of packed 64-bit keys per column (order-preserving float bits in the high word, the row index in the low word)."""
    n, k = bits.shape
    n_rep = idx.shape[0]
    out = np.zeros((n_rep, k), dtype=np.float64)
    ok = np.zeros(k, dtype=np.bool_)
    for j in prange(k):
        keys = np.empty(n, dtype=np.uint64)
        finite = True
        for i in range(n):
            u = bits[i, j]
            if (u & np.uint32(0x7F800000)) == np.uint32(0x7F800000):
                finite = False
                break
            if u == np.uint32(0x80000000):
                u = np.uint32(0)  # -0.0 equals +0.0
            if u & np.uint32(0x80000000):
                u = ~u
            else:
                u = u | np.uint32(0x80000000)
            keys[i] = (np.uint64(u) << np.uint64(32)) | np.uint64(i)
        if not finite:
            continue
        ok[j] = True
        keys.sort()
        rk = np.empty(n, dtype=np.int32)
        blk = 0
        for p in range(n):
            if p > 0 and (keys[p] >> np.uint64(32)) != (keys[p - 1] >> np.uint64(32)):
                blk += 1
            rk[np.int64(keys[p] & np.uint64(0xFFFFFFFF))] = blk
        _replicate_mis(rk, blk + 1, y_codes, n_classes, idx, lens, binpos, n_bins, out, j)
    return out, ok


@njit(cache=True, fastmath=True, parallel=True)
def _bootstrap_rank_mi_generic(arr: np.ndarray, y_codes: np.ndarray, n_classes: int, idx: np.ndarray, lens: np.ndarray, binpos: np.ndarray, n_bins: int):
    """:func:`_bootstrap_rank_mi_f32` for any float dtype: the dense ranks come from a stable argsort of the column."""
    n, k = arr.shape
    n_rep = idx.shape[0]
    out = np.zeros((n_rep, k), dtype=np.float64)
    ok = np.zeros(k, dtype=np.bool_)
    for j in prange(k):
        col = np.empty(n, dtype=arr.dtype)
        finite = True
        for i in range(n):
            v = arr[i, j]
            if not np.isfinite(v):
                finite = False
                break
            col[i] = v
        if not finite:
            continue
        ok[j] = True
        order = np.argsort(col, kind="mergesort")
        rk = np.empty(n, dtype=np.int32)
        blk = 0
        for p in range(n):
            if p > 0 and col[order[p]] != col[order[p - 1]]:
                blk += 1
            rk[order[p]] = blk
        _replicate_mis(rk, blk + 1, y_codes, n_classes, idx, lens, binpos, n_bins, out, j)
    return out, ok


def bootstrap_rank_mi(arr: np.ndarray, y: np.ndarray, rep_rows: "list[np.ndarray]", n_bins: int) -> "tuple[np.ndarray, np.ndarray]":
    """Rank-binned plug-in MI of every column of ``arr`` ``(n, k)`` against the class labels ``y`` on every resample in ``rep_rows`` (row-index vectors with repeats, in draw order).

    Returns ``(mi (len(rep_rows), k), ok (k,))``: the MI equals ``_mi_classif_batch(arr[idx], y[idx], nbins=n_bins)`` for each resample ``idx`` and each column with ``ok`` True. A column with a
    non-finite value is not scored (``ok`` False, MI 0): the batch MI masks such a column to its finite rows.
    """
    y_arr = np.asarray(y).ravel()
    y_codes = (y_arr.astype(np.int64) - int(y_arr.min())).astype(np.int64)
    n_classes = int(y_codes.max()) + 1
    n_rep = len(rep_rows)
    lens = np.array([len(r) for r in rep_rows], dtype=np.int64)
    idx = np.zeros((n_rep, int(lens.max())), dtype=np.int32)
    binpos = np.zeros((n_rep, int(lens.max())), dtype=np.int16)
    for b, rows in enumerate(rep_rows):
        m = len(rows)
        idx[b, :m] = rows
        base, rem = divmod(m, n_bins)
        binpos[b, :m] = np.repeat(np.arange(n_bins, dtype=np.int16), [base + (1 if t < rem else 0) for t in range(n_bins)])
    if arr.dtype == np.float32:
        mi, ok = _bootstrap_rank_mi_f32(np.ascontiguousarray(arr).view(np.uint32), y_codes, n_classes, idx, lens, binpos, int(n_bins))
    else:
        mi, ok = _bootstrap_rank_mi_generic(np.ascontiguousarray(arr), y_codes, n_classes, idx, lens, binpos, int(n_bins))
    return mi, np.asarray(ok)
