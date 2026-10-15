"""Row-major count-based ``searchsorted(side='right')`` for the FE discretiser (serial and row-block parallel).

:func:`_kernels._count_le_2d_njit` and :func:`_kernels._count_le_2d_njit_parallel` walk one column at a time, so every ``arr2d[r, j]`` read
and every ``out[r, j]`` write strides by ``n_cols`` in the C-ordered buffers. Here each row is handled as a whole: the inner loop runs over
the columns, so the input row, the edge row ``edges_inner[k, :]`` and the output row are all unit-stride, and the compare-and-add vectorises.
The result is the same integer count per element, so the codes are identical; NaN keeps the rightmost-bin contract (``n_edges``).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

_ROW_BLOCK = 256


@njit(nogil=True, cache=True)
def _count_le_2d_rowmajor_njit(edges_inner: np.ndarray, arr2d: np.ndarray, out: np.ndarray) -> None:
    """Serial row-major twin of ``_count_le_2d_njit``: ``out[r, j] = #{k : edges_inner[k, j] <= arr2d[r, j]}``, NaN -> ``n_edges``."""
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_edges = edges_inner.shape[0]
    c = np.empty(n_cols, dtype=np.int64)
    for r in range(n_rows):
        for j in range(n_cols):
            c[j] = 0
        for k in range(n_edges):
            for j in range(n_cols):
                c[j] += edges_inner[k, j] <= arr2d[r, j]
        for j in range(n_cols):
            v = arr2d[r, j]
            out[r, j] = n_edges if v != v else c[j]


@njit(parallel=True, nogil=True, cache=True)
def _count_le_2d_rowmajor_njit_parallel(edges_inner: np.ndarray, arr2d: np.ndarray, out: np.ndarray) -> None:
    """Row-block ``prange`` twin of :func:`_count_le_2d_rowmajor_njit`: each block owns its rows of ``out`` (no shared writes)."""
    n_rows = arr2d.shape[0]
    n_cols = arr2d.shape[1]
    n_edges = edges_inner.shape[0]
    n_blocks = (n_rows + _ROW_BLOCK - 1) // _ROW_BLOCK
    for blk in prange(n_blocks):
        r0 = blk * _ROW_BLOCK
        r1 = min(n_rows, r0 + _ROW_BLOCK)
        c = np.empty(n_cols, dtype=np.int64)
        for r in range(r0, r1):
            for j in range(n_cols):
                c[j] = 0
            for k in range(n_edges):
                for j in range(n_cols):
                    c[j] += edges_inner[k, j] <= arr2d[r, j]
            for j in range(n_cols):
                v = arr2d[r, j]
                out[r, j] = n_edges if v != v else c[j]


__all__ = ["_count_le_2d_rowmajor_njit", "_count_le_2d_rowmajor_njit_parallel"]
