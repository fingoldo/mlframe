"""Stacking the code batches of one pair family into a single (columns, rows) matrix: merged column runs, runs of rows of one parent, and the parallel tiled transpose.

Carved out of ``_family_maxt_gate`` (which re-exports every name here) to keep that module under the line budget.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from mlframe._numba_parallel_guard import parallel_kernel_entry

def _columns_contiguous(a: np.ndarray) -> bool:
    """Whether the columns of the 2-D block ``a`` follow each other in memory within a row; the stride of an axis of length 1 carries no meaning (a single-column view may report any value)."""
    return bool(a.shape[1] <= 1 or a.strides[1] == a.itemsize)


def _column_runs(mats, widths) -> list:
    """``(block, merged)`` pairs: the batches as 2-D blocks, with every run of batches that are adjacent column blocks of one array merged into a single strided view (``merged`` True).

    The pair family passes one single-column view per configuration, all cut from one code matrix in order; copying them one by one reads each column at the row stride of the whole matrix
    and starts a parallel region per column. Two batches merge when they have the same dtype, row count and strides, the columns are contiguous within a row, and the second starts exactly one
    batch width after the first in memory, so the merged view addresses nothing outside the batches' own memory. Anything else stays a block of its own.
    """
    runs: list = []
    for m, w in zip(mats, widths):
        if runs:
            head, width = runs[-1]
            if (
                m.ndim == 2
                and head.ndim == 2
                and m.dtype == head.dtype
                and m.shape[0] == head.shape[0]
                and m.strides[0] == head.strides[0]
                and _columns_contiguous(m)
                and _columns_contiguous(head)
                and m.__array_interface__["data"][0] == head.__array_interface__["data"][0] + width * head.itemsize
            ):
                runs[-1][1] = width + int(w)
                continue
        runs.append([m, int(w)])
    blocks = []
    for head, width in runs:
        if width == head.shape[1]:
            blocks.append((head, False))
        else:
            blocks.append(
                (np.lib.stride_tricks.as_strided(head, shape=(head.shape[0], width), strides=(head.strides[0], head.itemsize), writeable=False), True)
            )
    return blocks


def _stack_codes_transposed(mats, widths) -> np.ndarray:
    """The (K_total, n) code matrix of the batches in the narrowest integer width that holds their codes, built in one pass.

    The family of one pair spans every configuration scored for it (hundreds of batches of ~1500 columns at 30000 rows); upcasting each batch to
    int64 and then stacking and transposing it held that family three times at eight bytes a code, which exhausted host memory on a wide frame.
    Batches that are adjacent column blocks of one array are transposed together (see :func:`_column_runs`).
    """
    # bench-attempt-rejected (2026-10-16): one np.concatenate + transposed copy instead of this loop: 0.27 s vs 0.19 s for 1944 single-column int8 batches at 10000 rows.
    n = int(mats[0].shape[0])
    parts: list = []  # (rows, None) for a run of rows of one transposed parent, (None, blocks) for batches that need the transpose
    for rows, idx in _row_run_segments(mats, widths):
        parts.append((rows, None) if rows is not None else (None, _column_runs([mats[k] for k in idx], [widths[k] for k in idx])))
    top = 0
    for rows, blocks in parts:
        if rows is not None:
            top = max(top, int(rows.max()) if rows.size else 0)
        else:
            top = max(top, max((int(b.max()) for b, _ in blocks if b.size), default=0))
    dt = np.int8 if top <= np.iinfo(np.int8).max else (np.int16 if top <= np.iinfo(np.int16).max else np.int64)
    cols = np.empty((int(sum(widths)), n), dtype=dt)
    at = 0
    for rows, blocks in parts:
        if rows is not None:
            cols[at : at + rows.shape[0]] = rows  # consecutive rows of one parent: already transposed, a plain block copy
            at += int(rows.shape[0])
            continue
        for b, merged in blocks:
            # A parallel blocked transpose into the shared buffer: numpy's strided ``m.T`` copy was the largest host cost of the family gate on a wide pair pool. A merged strided block is read
            # in place (rows of a tile are contiguous in memory); a batch on its own is made contiguous first, as before.
            with parallel_kernel_entry():
                _transpose_into(b if merged else np.ascontiguousarray(b), cols, at)
            at += int(b.shape[1])
    return cols


def _row_run_segments(mats, widths) -> list:
    """The batches in order as ``(rows, None)`` for a run of at least two single-column batches that are consecutive rows of one C-order ``(m, n)`` parent (the layout ``ChunkCodes.columns_t`` hands out),
    seen as one ``(run, n)`` view, and ``(None, indices)`` for every other stretch of batches.

    A single-column view of a row of the parent is contiguous along the rows of the code matrix (its stride is one code) and the next one starts ``n`` codes later; such a run is the transposed block
    already, so it is copied as a block where the generic path would start a parallel transpose per batch (38063 single-column calls, 2.6 s in a whole-fit cProfile).
    """
    segs: list = []
    i, total = 0, len(mats)

    def _row_view(k: int) -> bool:
        """Whether batch ``k`` is a single-column view whose code stride is one code."""
        m = mats[k]
        return bool(m.ndim == 2 and m.shape[1] == 1 and int(widths[k]) == 1 and m.strides[0] == m.itemsize)

    while i < total:
        if _row_view(i):
            m = mats[i]
            n, isz, ptr = int(m.shape[0]), m.itemsize, m.__array_interface__["data"][0]
            j = i + 1
            while (
                j < total
                and _row_view(j)
                and mats[j].dtype == m.dtype
                and mats[j].shape[0] == n
                and mats[j].__array_interface__["data"][0] == ptr + (j - i) * n * isz
            ):
                j += 1
            if j - i >= 2:
                segs.append((np.lib.stride_tricks.as_strided(m, shape=(j - i, n), strides=(n * isz, isz), writeable=False), None))
                i = j
                continue
        if segs and segs[-1][0] is None:
            segs[-1][1].append(i)
        else:
            segs.append((None, [i]))
        i += 1
    return segs


@njit(cache=True, parallel=True, nogil=True)
def _transpose_into(src, dst, at):
    """``dst[at + j, i] = src[i, j]`` for every element of the ``(n, w)`` block ``src``, in cache-sized tiles, parallel over the rows of ``dst``."""
    n, w = src.shape
    tile = 64
    for j0 in prange((w + tile - 1) // tile):
        j_lo = j0 * tile
        j_hi = min(j_lo + tile, w)
        for i0 in range(0, n, tile):
            i_hi = min(i0 + tile, n)
            for j in range(j_lo, j_hi):
                for i in range(i0, i_hi):
                    dst[at + j, i] = src[i, j]
