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


@njit(nogil=True, cache=True)
def _unique_counts_f32_njit(x: np.ndarray):
    """``(values, counts)`` of the NaN-free float32 vector ``x`` as ``np.unique(x, return_counts=True)`` gives them: one radix sort of a copy and one pass over the sorted run.

    ``nogil`` and allocation-local, so the column pool of the adaptive binner runs it from threads. The values are the same numbers; the sign of an exact zero is the one radix order puts first (-0.0), which compares equal to +0.0.
    """
    n = x.shape[0]
    col = x.copy()
    keys = np.empty(n, dtype=np.uint32)
    tmp = np.empty(n, dtype=np.uint32)
    cnt = np.empty(256, dtype=np.int64)
    _radix_sort_f32(col, keys, tmp, cnt)
    m = 1
    for i in range(1, n):
        if col[i] != col[i - 1]:
            m += 1
    vals = np.empty(m, dtype=col.dtype)
    counts = np.empty(m, dtype=np.int64)
    j = 0
    vals[0] = col[0]
    counts[0] = 1
    for i in range(1, n):
        if col[i] != col[i - 1]:
            j += 1
            vals[j] = col[i]
            counts[j] = 1
        else:
            counts[j] += 1
    return vals, counts


# Below this size ``np.unique`` is as fast as the sort-and-scan (and the JIT dispatch is not worth it): measured 6.8 against 1.1 ms for 100000 continuous float32 values.
_UNIQUE_COUNTS_MIN_SIZE = 4096


def unique_with_counts(values: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """``np.unique(values, return_counts=True)``: the radix sort-and-scan for a large NaN-free float32 vector (6x faster), numpy for anything else."""
    if values.dtype == np.float32 and values.ndim == 1 and values.size >= _UNIQUE_COUNTS_MIN_SIZE:
        radix: "tuple[np.ndarray, np.ndarray]" = _unique_counts_f32_njit(np.ascontiguousarray(values))
        return radix
    return np.unique(values, return_counts=True)


@njit(nogil=True, cache=True)
def _sorted_quantiles_f32_njit(x: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """``np.quantile(x.astype(float64), quantiles)`` (method ``linear``) of a NaN-free float32 vector from one radix sort of a copy.

    The virtual index is ``(n - 1) * q``, the interpolation is numpy's ``_lerp`` in float64 (``a + (b - a) * t``, and ``b - (b - a) * (1 - t)`` from ``t >= 0.5``), so the float64 results are the ones numpy gives bit for bit.
    """
    n = x.shape[0]
    col = x.copy()
    keys = np.empty(n, dtype=np.uint32)
    tmp = np.empty(n, dtype=np.uint32)
    cnt = np.empty(256, dtype=np.int64)
    _radix_sort_f32(col, keys, tmp, cnt)
    out = np.empty(quantiles.shape[0], dtype=np.float64)
    for qi in range(quantiles.shape[0]):
        v = quantiles[qi] * (n - 1)
        lo = math.floor(v)
        if lo >= n - 1:
            out[qi] = np.float64(col[n - 1])  # Intentional float64: the interpolation runs in double, as numpy's on the widened column
        else:
            a = np.float64(col[lo])  # Intentional float64: operands widened exactly before the interpolation
            b = np.float64(col[lo + 1])  # Intentional float64: operands widened exactly before the interpolation
            t = v - lo
            diff = b - a
            if t >= 0.5:
                out[qi] = b - diff * (1.0 - t)
            else:
                out[qi] = a + diff * t
    return out


def sorted_quantiles(values: np.ndarray, quantiles: np.ndarray) -> "np.ndarray | None":
    """Float64 quantiles (fractions in ``[0, 1]``) of a large NaN-free float32 vector by the radix route, or ``None`` when it does not apply (another dtype, a small vector): the caller then uses numpy."""
    if values.dtype == np.float32 and values.ndim == 1 and values.size >= _UNIQUE_COUNTS_MIN_SIZE:
        found: np.ndarray = _sorted_quantiles_f32_njit(np.ascontiguousarray(values), np.ascontiguousarray(quantiles, dtype=np.float64))
        return found
    return None


@njit(nogil=True, cache=True)
def _sort_xy_f32_njit(x: np.ndarray, y: np.ndarray):
    """``(x[order].astype(float64), y[order])`` for the stable ascending order of the NaN-free float32 vector ``x``: a radix argsort (the order-preserving key of each float with its index riding along, four 8-bit LSD passes) and one gather of both arrays.

    Equal ``x`` keep their input order (``np.argsort``'s quicksort leaves them in an unspecified one); the sorted ``x`` is identical, and so are the per-value label counts that the MDLP split search reads.
    """
    n = x.shape[0]
    u = x.view(np.uint32)
    keys = np.empty(n, dtype=np.uint32)
    idx = np.empty(n, dtype=np.int64)
    k2 = np.empty(n, dtype=np.uint32)
    i2 = np.empty(n, dtype=np.int64)
    cnt = np.empty(256, dtype=np.int64)
    for i in range(n):
        v = u[i]
        if v & 0x80000000:
            keys[i] = ~v
        else:
            keys[i] = v | 0x80000000
        idx[i] = i
    for shift in (0, 8, 16, 24):
        cnt[:] = 0
        for i in range(n):
            cnt[(keys[i] >> shift) & 0xFF] += 1
        s = 0
        for b in range(256):
            c = cnt[b]
            cnt[b] = s
            s += c
        for i in range(n):
            k = keys[i]
            b = (k >> shift) & 0xFF
            p = cnt[b]
            k2[p] = k
            i2[p] = idx[i]
            cnt[b] = p + 1
        keys, k2 = k2, keys
        idx, i2 = i2, idx
    xs = np.empty(n, dtype=np.float64)  # Intentional float64: the MDLP cut search runs at the production binning width
    ys = np.empty(n, dtype=y.dtype)
    for i in range(n):
        j = idx[i]
        xs[i] = x[j]
        ys[i] = y[j]
    return xs, ys


def sort_xy(x: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """``(x sorted ascending as float64, y in that order)`` by the radix route for a large NaN-free float32 ``x`` with a 1-D integer ``y`` of the same length, else ``None`` (the caller sorts with numpy)."""
    if x.dtype == np.float32 and x.ndim == 1 and x.size >= _UNIQUE_COUNTS_MIN_SIZE and y.ndim == 1 and y.shape[0] == x.shape[0] and y.dtype.kind in "iu":
        pair: "tuple[np.ndarray, np.ndarray]" = _sort_xy_f32_njit(np.ascontiguousarray(x), np.ascontiguousarray(y))
        return pair
    return None


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


def tile_cols_for(n_cols: int, threads: int) -> int:
    """Tile width for ``n_cols`` columns on ``threads`` threads: ``_TILE_COLS`` unless that leaves threads without a tile, then ``ceil(n_cols / threads)``, never under 16 (one 64-byte line of float32 per row read)."""
    return int(max(16, min(_TILE_COLS, -(-int(n_cols) // max(1, int(threads))))))


@njit(parallel=True, nogil=True, cache=True)
def _quantile_edges_2d_radix_tiled_njit(arr2d: np.ndarray, quantiles: np.ndarray, edges_out: np.ndarray, tile_cols: int = _TILE_COLS) -> None:
    """Tiled twin of :func:`_quantile_edges_2d_radix_njit`: BYTE-IDENTICAL edges. Each prange tile gathers ``_TILE_COLS`` adjacent columns
    with row-contiguous reads (``arr2d[r, j0:j1]`` is one cache line run), then radix-sorts each column of the tile in place. The
    per-column gather from the C-ordered buffer otherwise costs one cache miss per element.

    ``tile_cols`` is the tile width (columns are independent, so it changes the schedule and never an edge): see :func:`tile_cols_for`, which narrows it so that every thread gets a tile."""
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_q = quantiles.shape[0]
    if n_rows == 0 or n_cols == 0:
        return
    n_tiles = (n_cols + tile_cols - 1) // tile_cols
    for t in prange(n_tiles):
        j0 = t * tile_cols
        j1 = min(n_cols, j0 + tile_cols)
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
