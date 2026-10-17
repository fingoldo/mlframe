"""Radix-selected quantile edges for float32 columns (the FE buffer dtype).

:func:`_kernels._quantile_edges_2d_njit` picks the two order statistics each quantile interpolates between with ``multiselect_inplace``.
For the 2000 x 7668 float32 chunk the kernel spends ~50-90 us per column there, which is slower than a plain LSD radix sort of the
column (4 passes of 8 bits over an order-preserving uint32 key, ~40 us). A full sort yields the same order-statistic values, so the edges
are identical; the only difference that can appear is the sign of an exact zero (-0.0 vs +0.0 compare equal, so the bins they define are
identical). Input must be NaN-free, as for the multiselect kernel.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(nogil=True, cache=True)
def _radix_pass_8bit(src: np.ndarray, dst: np.ndarray, cnt: np.ndarray, shift: int, n: int) -> None:
    """One stable LSD counting-sort pass of 8 bits at ``shift``: scatter ``src`` into ``dst`` (``cnt`` is a 256-slot int64 scratch)."""
    cnt[:] = 0
    for i in range(n):
        cnt[(src[i] >> shift) & 0xFF] += 1
    s = 0
    for b in range(256):
        c = cnt[b]
        cnt[b] = s
        s += c
    for i in range(n):
        v = src[i]
        b = (v >> shift) & 0xFF
        dst[cnt[b]] = v
        cnt[b] += 1


@njit(nogil=True, cache=True)
def _radix_sort_f32(col: np.ndarray, keys: np.ndarray, tmp: np.ndarray, cnt: np.ndarray) -> None:
    """Sort the NaN-free float32 vector ``col`` in place (ascending). ``keys``/``tmp`` are uint32 scratch of ``col``'s length, ``cnt`` an
    int64 scratch of 256. The float bits are mapped to order-preserving unsigned keys (flip all bits for negatives, set the sign bit for
    non-negatives), sorted by four LSD passes of 8 bits (an even count, so the result is back in ``keys``), and mapped back."""
    n = col.shape[0]
    u = col.view(np.uint32)
    for i in range(n):
        x = u[i]
        if x & 0x80000000:
            keys[i] = ~x
        else:
            keys[i] = x | 0x80000000
    for shift in range(0, 32, 16):
        _radix_pass_8bit(keys, tmp, cnt, shift, n)
        _radix_pass_8bit(tmp, keys, cnt, shift + 8, n)
    for i in range(n):
        x = keys[i]
        if x & 0x80000000:
            u[i] = x & 0x7FFFFFFF
        else:
            u[i] = ~x


@njit(parallel=True, nogil=True, cache=True)
def _quantile_edges_2d_radix_njit(arr2d: np.ndarray, quantiles: np.ndarray, edges_out: np.ndarray) -> None:
    """Per-column quantile edges of a NaN-free float32 ``arr2d`` (n_rows x n_cols), written into ``edges_out`` (n_q x n_cols, double).

    Same interpolation as :func:`_kernels._quantile_edges_2d_njit` (the ``lo``/``lo + 1`` order statistics with the ``n - 1`` clamp and the
    same lerp), with the order statistics read from a full radix sort of each column. prange over columns: each iteration owns its scratch.
    """
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_q = quantiles.shape[0]
    if n_rows == 0 or n_cols == 0:
        return
    for j in prange(n_cols):
        col = np.empty(n_rows, dtype=np.float32)
        for r in range(n_rows):
            col[r] = arr2d[r, j]
        keys = np.empty(n_rows, dtype=np.uint32)
        tmp = np.empty(n_rows, dtype=np.uint32)
        cnt = np.empty(256, dtype=np.int64)
        _radix_sort_f32(col, keys, tmp, cnt)
        for qi in range(n_q):
            v = (quantiles[qi] / 100.0) * (n_rows - 1)
            lo = math.floor(v)
            if lo >= n_rows - 1:
                edges_out[qi, j] = col[n_rows - 1]
            else:
                a = float(col[lo])
                b = float(col[lo + 1])
                t = v - lo
                diff_b_a = b - a
                if t >= 0.5:
                    edges_out[qi, j] = b - diff_b_a * (1.0 - t)
                else:
                    edges_out[qi, j] = a + diff_b_a * t


_TILE_COLS = 64


@njit(parallel=True, nogil=True, cache=True)
def _quantile_edges_2d_radix_tiled_njit(arr2d: np.ndarray, quantiles: np.ndarray, edges_out: np.ndarray) -> None:
    """Tiled twin of :func:`_quantile_edges_2d_radix_njit`: BYTE-IDENTICAL edges. Each prange tile gathers ``_TILE_COLS`` adjacent columns
    with row-contiguous reads (``arr2d[r, j0:j1]`` is one cache line run), then radix-sorts each column of the tile in place. The
    per-column gather from the C-ordered buffer otherwise costs one cache miss per element."""
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_q = quantiles.shape[0]
    if n_rows == 0 or n_cols == 0:
        return
    n_tiles = (n_cols + _TILE_COLS - 1) // _TILE_COLS
    for t in prange(n_tiles):
        j0 = t * _TILE_COLS
        j1 = min(n_cols, j0 + _TILE_COLS)
        w = j1 - j0
        tile = np.empty((w, n_rows), dtype=np.float32)
        for r in range(n_rows):
            for jj in range(w):
                tile[jj, r] = arr2d[r, j0 + jj]
        keys = np.empty(n_rows, dtype=np.uint32)
        tmp = np.empty(n_rows, dtype=np.uint32)
        cnt = np.empty(256, dtype=np.int64)
        for jj in range(w):
            col = tile[jj]
            _radix_sort_f32(col, keys, tmp, cnt)
            j = j0 + jj
            for qi in range(n_q):
                v = (quantiles[qi] / 100.0) * (n_rows - 1)
                lo = math.floor(v)
                if lo >= n_rows - 1:
                    edges_out[qi, j] = col[n_rows - 1]
                else:
                    a = float(col[lo])
                    b = float(col[lo + 1])
                    tt = v - lo
                    diff_b_a = b - a
                    if tt >= 0.5:
                        edges_out[qi, j] = b - diff_b_a * (1.0 - tt)
                    else:
                        edges_out[qi, j] = a + diff_b_a * tt


__all__ = ["_quantile_edges_2d_radix_njit", "_quantile_edges_2d_radix_tiled_njit", "_radix_sort_f32"]
