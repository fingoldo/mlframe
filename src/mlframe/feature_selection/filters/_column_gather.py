"""Columns of a row-major code matrix gathered into a column-major block, on every core.

The pair kernels read each column of a pair as one contiguous run, and the matrix of the fit is row-major, so the columns a launch touches are gathered and transposed on the host before they
cross the bus. ``matrix[:, used].T`` copied through ``ascontiguousarray`` does that on one thread, walking a stride of ``n_columns`` bytes per element: at 1000000 rows and 770 columns it kept the
card idle for minutes per launch of the all-pairs MI table. Here a tile of rows is read once for every column of the block (the tile stays in the cache), the writes are contiguous, and the row
blocks run on the numba threads.
"""

from __future__ import annotations

import numba
import numpy as np
from numba import prange

from mlframe._numba_parallel_guard import parallel_kernel_entry

#: Rows per parallel task and rows per cache tile (a tile of 64 rows of a 1141-byte row is 73 KB, read again for each column of the block).
_ROW_BLOCK = 8192
_TILE = 64


@numba.njit(cache=True, nogil=True, parallel=True)
def _gather_columns_cm_njit(data, cols, out):
    """``out[j, r] = data[r, cols[j]]`` for every row ``r`` of ``data`` (``out`` has at least as many columns as ``data`` has rows; the tail is left as it is)."""
    n = data.shape[0]
    m = cols.shape[0]
    n_blocks = (n + _ROW_BLOCK - 1) // _ROW_BLOCK
    for b in prange(n_blocks):
        r0 = b * _ROW_BLOCK
        r1 = min(n, r0 + _ROW_BLOCK)
        for t0 in range(r0, r1, _TILE):
            t1 = min(r1, t0 + _TILE)
            for j in range(m):
                c = cols[j]
                for r in range(t0, t1):
                    out[j, r] = data[r, c]


@numba.njit(cache=True, nogil=True, parallel=True)
def _scatter_columns_rm_njit(dst, dst_cols, src):
    """``dst[r, dst_cols[j]] = src[j, r]``: the rows of the column-major block ``src`` go into the columns ``dst_cols`` of the row-major matrix ``dst``."""
    n = dst.shape[0]
    m = src.shape[0]
    n_blocks = (n + _ROW_BLOCK - 1) // _ROW_BLOCK
    for b in prange(n_blocks):
        r0 = b * _ROW_BLOCK
        r1 = min(n, r0 + _ROW_BLOCK)
        for t0 in range(r0, r1, _TILE):
            t1 = min(r1, t0 + _TILE)
            for r in range(t0, t1):
                for j in range(m):
                    dst[r, dst_cols[j]] = src[j, r]


def scatter_columns_rm(dst: np.ndarray, dst_cols: np.ndarray, src: np.ndarray) -> None:
    """Write the ``(m, n)`` column-major block ``src`` into the columns ``dst_cols`` of the ``(n, p)`` row-major matrix ``dst`` (cast to its dtype), tiled on every core.

    ``dst[:, j] = column`` writes one byte of a row every ``p`` bytes: 900 million such writes took 7.7 s for the 1800 columns of a 500000-row symmetric-uncertainty quantisation."""
    cols = np.ascontiguousarray(dst_cols, dtype=np.int64)
    if cols.shape[0] != src.shape[0] or src.shape[1] != dst.shape[0]:
        raise ValueError(f"scatter_columns_rm: block {src.shape} does not fit {cols.shape[0]} columns of a matrix with {dst.shape[0]} rows")
    if src.size:
        with parallel_kernel_entry():
            _scatter_columns_rm_njit(dst, cols, src)


def gather_columns_cm(data: np.ndarray, cols: np.ndarray, *, pad_to: int = 1, dtype: "np.dtype | type" = np.uint8) -> np.ndarray:
    """The columns ``cols`` of the row-major matrix ``data`` as a C-contiguous ``(len(cols), n_rows_padded)`` block of ``dtype``; the rows are padded with zeros up to a multiple of ``pad_to``.

    The codes must fit ``dtype`` (the byte kernels check that before they gather)."""
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    n = int(data.shape[0])
    step = max(1, int(pad_to))
    out = np.zeros((int(cols.shape[0]), -(-n // step) * step), dtype=dtype)
    if n and cols.shape[0]:
        with parallel_kernel_entry():
            _gather_columns_cm_njit(np.asarray(data), cols, out)
    return out


__all__ = ["gather_columns_cm", "scatter_columns_rm"]
