"""Rank-binned plug-in MI of bootstrap resamples without gathering or re-sorting the resample.

The bootstrap scorers (modular, decomposition, orthogonal) draw B resamples of the rows (``idx``: row indices with repeats, in draw order) and bin every candidate column into
equi-frequency RANK bins of each resample (``_plugin_mi_classif_njit``): gather ``(sample_n, k)`` values, stable-argsort every column, cut the sorted positions into ``n_bins`` slices of
``m // n_bins`` (+1 for the first ``m % n_bins``), count ``(bin, class)`` cells. Equal values are ordered by their position in the resample (stable sort).

Here each column is sorted ONCE. A resample is a multiset of rows, so its sorted order is the full sorted order with every row repeated by its multiplicity, and all replicates advance over
the sorted rows together. A block of equal values is emitted in the order of its draws, which is what the stable argsort of the gathered values does; a column with many repeated values counting-sorts
the draws by dense rank instead. The MI equals the gather-and-argsort path bit for bit for every column, ties included.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

__all__ = ["bootstrap_rank_mi"]

_TILE = 16  # columns a thread ranks together; their buffers are reused
_TIE_FRACTION_MAX = 0.01  # a column with at most this fraction of repeated values takes the sorted walk, the others the counting sort of the draws


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


@njit(cache=True)
def _radix_sort_pairs(vals: np.ndarray, va: np.ndarray, ra: np.ndarray, vb: np.ndarray, rb: np.ndarray) -> None:
    """Stable LSD radix sort (4 passes of 8 bits) of the ``uint32`` order-preserving keys ``vals`` carrying their row ids; the result is in ``(va, ra)``."""
    n = vals.shape[0]
    for i in range(n):
        va[i] = vals[i]
        ra[i] = i
    cnt = np.empty(257, dtype=np.int64)
    for shift in (0, 8, 16, 24):
        cnt[:] = 0
        for i in range(n):
            cnt[((va[i] >> np.uint32(shift)) & np.uint32(0xFF)) + 1] += 1
        for d in range(256):
            cnt[d + 1] += cnt[d]
        for i in range(n):
            d = (va[i] >> np.uint32(shift)) & np.uint32(0xFF)
            q = cnt[d]
            cnt[d] = q + 1
            vb[q] = va[i]
            rb[q] = ra[i]
        for i in range(n):
            va[i] = vb[i]
            ra[i] = rb[i]


@njit(cache=True)
def _emit(hist: np.ndarray, pos: np.ndarray, cur: np.ndarray, cum: np.ndarray, b: int, c: int, take: int) -> None:
    """Add ``take`` rows of class ``c`` to replicate ``b`` at its current position, moving to the next equi-frequency slice whenever the current one is full (``cum[b, s]`` = end of slice ``s``)."""
    r = take
    last = cum.shape[1] - 1
    while r > 0:
        while pos[b] >= cum[b, cur[b]] and cur[b] < last:
            cur[b] += 1
        room = cum[b, cur[b]] - pos[b]
        t = min(r, room) if room > 0 else r
        hist[b, cur[b], c] += t
        pos[b] += t
        r -= t


@njit(cache=True, fastmath=True)
def _walk_sorted(
    ord_rows: np.ndarray,
    sv: np.ndarray,
    y_codes: np.ndarray,
    counts_t: np.ndarray,
    rowptr: np.ndarray,
    poslist: np.ndarray,
    idx: np.ndarray,
    cum: np.ndarray,
    n_classes: int,
    out: np.ndarray,
    col: int,
) -> None:
    """``out[b, col]`` for a column with few repeated values: walk the rows in sorted order, a row emitted ``counts_t[row, b]`` times into replicate ``b``.

    ``cum[b, s]`` is the end position of equi-frequency slice ``s`` of replicate ``b`` (slices of ``m // n_bins`` rows, the first ``m % n_bins`` one longer, as ``_quantile_bin_njit`` cuts them).
    A block of equal values is emitted in the order of its draws (the stable argsort of the resample), found through ``rowptr`` / ``poslist``: the draw positions of every row, per replicate."""
    n = ord_rows.shape[0]
    n_rep = counts_t.shape[1]
    n_bins = cum.shape[1]
    hist = np.zeros((n_rep, n_bins, n_classes), dtype=np.int64)
    pos = np.zeros(n_rep, dtype=np.int64)
    cur = np.zeros(n_rep, dtype=np.int64)
    bound = cum[:, 0].copy()
    buf = np.empty(idx.shape[1], dtype=np.int64)
    p = 0
    while p < n:
        q = p + 1
        while q < n and sv[q] == sv[p]:
            q += 1
        if q - p == 1:
            row = ord_rows[p]
            c = y_codes[row]
            for b in range(n_rep):
                r = counts_t[row, b]
                end = pos[b] + r
                if end > bound[b]:
                    _emit(hist, pos, cur, cum, b, c, r)
                    bound[b] = cum[b, cur[b]]
                else:
                    hist[b, cur[b], c] += r
                    pos[b] = end
        else:
            for b in range(n_rep):
                t = 0
                for z in range(p, q):
                    row = ord_rows[z]
                    for u in range(rowptr[b, row], rowptr[b, row + 1]):
                        buf[t] = poslist[b, u]
                        t += 1
                if t == 0:
                    continue
                sel = np.sort(buf[:t])
                for u in range(t):
                    _emit(hist, pos, cur, cum, b, y_codes[idx[b, sel[u]]], 1)
                bound[b] = cum[b, cur[b]]
        p = q
    for b in range(n_rep):
        out[b, col] = _mi_of_hist(hist[b], pos[b])


@njit(cache=True, fastmath=True, parallel=True)
def _bootstrap_rank_mi_f32(
    bits: np.ndarray,
    y_codes: np.ndarray,
    n_classes: int,
    idx: np.ndarray,
    lens: np.ndarray,
    binpos: np.ndarray,
    counts_t: np.ndarray,
    rowptr: np.ndarray,
    poslist: np.ndarray,
    cum: np.ndarray,
    n_bins: int,
):
    """``(mi (B, k), ok (k,))`` for a float32 matrix given as its raw ``uint32`` bits ``(n, k)``; ``ok`` is False (and the MI column 0) for a column with a non-finite value.

    A thread takes ``_TILE`` neighbouring columns, transposes them into contiguous buffers (reading one column of a wide row-major matrix touches a cache line per row), and radix-sorts the
    order-preserving float bits of each. A column with few repeated values walks its sorted rows once for all replicates (:func:`_walk_sorted`); a column with many repeats takes the counting
    sort of the draws by dense rank (:func:`_replicate_mis`)."""
    n, k = bits.shape
    n_rep = idx.shape[0]
    out = np.zeros((n_rep, k), dtype=np.float64)  # Intentional float64: bootstrap rank-MI output statistic (compared across replicates)
    ok = np.zeros(k, dtype=np.bool_)
    max_ties = int(_TIE_FRACTION_MAX * n)
    n_tiles = (k + _TILE - 1) // _TILE
    for tile in prange(n_tiles):
        j0 = tile * _TILE
        width = min(_TILE, k - j0)
        tb = np.empty((width, n), dtype=np.uint32)
        for i in range(n):
            for t in range(width):
                tb[t, i] = bits[i, j0 + t]
        keys = np.empty(n, dtype=np.uint32)
        va = np.empty(n, dtype=np.uint32)
        vb = np.empty(n, dtype=np.uint32)
        ra = np.empty(n, dtype=np.int32)
        rb = np.empty(n, dtype=np.int32)
        ord_rows = np.empty(n, dtype=np.int64)
        rk = np.empty(n, dtype=np.int32)
        for t in range(width):
            finite = True
            for i in range(n):
                u = tb[t, i]
                if (u & np.uint32(0x7F800000)) == np.uint32(0x7F800000):
                    finite = False
                    break
                if u == np.uint32(0x80000000):
                    u = np.uint32(0)  # -0.0 equals +0.0
                if u & np.uint32(0x80000000):
                    u = ~u
                else:
                    u = u | np.uint32(0x80000000)
                keys[i] = u
            if not finite:
                continue
            ok[j0 + t] = True
            _radix_sort_pairs(keys, va, ra, vb, rb)
            distinct = 1
            for pp in range(1, n):
                if va[pp] != va[pp - 1]:
                    distinct += 1
            for pp in range(n):
                ord_rows[pp] = ra[pp]
            if n - distinct <= max_ties:
                _walk_sorted(ord_rows, va, y_codes, counts_t, rowptr, poslist, idx, cum, n_classes, out, j0 + t)
            else:
                blk = 0
                for pp in range(n):
                    if pp > 0 and va[pp] != va[pp - 1]:
                        blk += 1
                    rk[ra[pp]] = blk
                _replicate_mis(rk, blk + 1, y_codes, n_classes, idx, lens, binpos, n_bins, out, j0 + t)
    return out, ok


@njit(cache=True, fastmath=True, parallel=True)
def _bootstrap_rank_mi_generic(arr: np.ndarray, y_codes: np.ndarray, n_classes: int, idx: np.ndarray, lens: np.ndarray, binpos: np.ndarray, n_bins: int):
    """:func:`_bootstrap_rank_mi_f32` for any float dtype: the dense ranks come from a stable argsort of the column, every column takes the counting sort of the draws."""
    n, k = arr.shape
    n_rep = idx.shape[0]
    out = np.zeros((n_rep, k), dtype=np.float64)  # Intentional float64: bootstrap rank-MI output statistic (compared across replicates)
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
    n = arr.shape[0]
    y_arr = np.asarray(y).ravel()
    y_codes = (y_arr.astype(np.int64) - int(y_arr.min())).astype(np.int64)
    n_classes = int(y_codes.max()) + 1
    n_rep = len(rep_rows)
    lens = np.array([len(r) for r in rep_rows], dtype=np.int64)
    m_max = int(lens.max())
    idx = np.zeros((n_rep, m_max), dtype=np.int32)
    binpos = np.zeros((n_rep, m_max), dtype=np.int16)
    for b, rows in enumerate(rep_rows):
        m = len(rows)
        idx[b, :m] = rows
        base, rem = divmod(m, n_bins)
        binpos[b, :m] = np.repeat(np.arange(n_bins, dtype=np.int16), [base + (1 if t < rem else 0) for t in range(n_bins)])
    if arr.dtype == np.float32:
        counts_t = np.empty((n, n_rep), dtype=np.int32)
        rowptr = np.zeros((n_rep, n + 1), dtype=np.int32)
        poslist = np.zeros((n_rep, m_max), dtype=np.int32)
        for b, rows in enumerate(rep_rows):
            cnt = np.bincount(rows, minlength=n)
            counts_t[:, b] = cnt
            np.cumsum(cnt, out=rowptr[b, 1:])
            poslist[b, : len(rows)] = np.argsort(rows, kind="stable")
        cum = np.zeros((n_rep, n_bins), dtype=np.int64)
        for b, rows in enumerate(rep_rows):
            base, rem = divmod(len(rows), n_bins)
            cum[b] = np.cumsum([base + (1 if t < rem else 0) for t in range(n_bins)])
        mi, ok = _bootstrap_rank_mi_f32(np.ascontiguousarray(arr).view(np.uint32), y_codes, n_classes, idx, lens, binpos, counts_t, rowptr, poslist, cum, int(n_bins))
    else:
        mi, ok = _bootstrap_rank_mi_generic(np.ascontiguousarray(arr), y_codes, n_classes, idx, lens, binpos, int(n_bins))
    return mi, np.asarray(ok)
