"""Device gathers of the code columns of a pair family, in one launch each (the comparison of the family's partitions and the matrix of its distinct columns).

A family's code batches are many small device matrices, usually column views of one chunk. ``cp.concatenate`` of a thousand of them is a thousand copy launches (29 ms for 1944 columns of 30000 rows) and the
transpose that follows is the generic strided copy, whose reads or writes are uncoalesced (62.8 ms; nvprof, GTX 1050 Ti, the pair family of the 100000 x 100 all-families fit). Here the batches are read in place
through a table that holds, for every global column, the address of its first element and the row stride of its batch (built on the host, so no kernel searches for a column's batch):

* ``transposed_codes``: the family as ``(columns, rows)``, the layout the partition hash reads. A 32 x 32 tile goes through shared memory padded to 33 columns (no bank conflicts), so the reads (adjacent columns of
  one row) and the writes (adjacent rows of one output row) are both coalesced.
* ``gathered_codes``: the chosen columns as a row-major ``(rows, chosen)`` matrix, the layout the bit-set packing kernels read. Threads walk the chosen columns, so the writes are coalesced.

Win axis: a family of hundreds to thousands of single-column batches (the default pair family); a family of a few wide batches gains little over ``cp.concatenate``. Every launch covers the whole family, whatever the
number of batches.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._batch_mi_noise_gate_bits import _C_TYPES

_SRC = r"""
// coalescing-ok: the tile read takes adjacent columns of one row per warp lane (adjacent addresses when the batches are views of one chunk, the table entries of adjacent columns are adjacent words) and the tile write takes adjacent rows of one output row, both through shared memory
extern "C" __global__ void transpose_multi(const unsigned long long* __restrict__ colptr, const long long* __restrict__ colld, const long long n, const int ntot, {DT}* __restrict__ out) {
    __shared__ {DT} tile[32][33];
    const int g = blockIdx.x * 32 + threadIdx.x;  // the global column this lane reads
    const {DT}* __restrict__ disc = 0;
    long long ldb = 0;
    if (g < ntot) {
        disc = (const {DT}*)colptr[g];
        ldb = colld[g];
    }
    for (long long r0 = (long long)blockIdx.y * 32; r0 < n; r0 += (long long)gridDim.y * 32) {
        #pragma unroll
        for (int j = 0; j < 32; j += 8) {
            const long long row = r0 + threadIdx.y + j;
            if (g < ntot && row < n) tile[threadIdx.y + j][threadIdx.x] = disc[row * ldb];
        }
        __syncthreads();
        const long long row_out = r0 + threadIdx.x;  // the lane writes the row it indexes, the warp's lanes are adjacent rows of one output row
        #pragma unroll
        for (int j = 0; j < 32; j += 8) {
            const int g_out = blockIdx.x * 32 + threadIdx.y + j;
            if (g_out < ntot && row_out < n) out[(long long)g_out * n + row_out] = tile[threadIdx.x][threadIdx.y + j];
        }
        __syncthreads();
    }
}
// coalescing-ok: lanes of a warp write adjacent chosen columns of one output row; each lane reads its own source column of that row, the columns of one chunk row are neighbours
extern "C" __global__ void gather_multi(const unsigned long long* __restrict__ colptr, const long long* __restrict__ colld, const long long n,
                                        const int* __restrict__ chosen, const int kd, {DT}* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= kd) return;
    const int g = chosen[c];
    const {DT}* __restrict__ disc = (const {DT}*)colptr[g];
    const long long ldb = colld[g];
    for (long long row = blockIdx.y; row < n; row += gridDim.y) out[row * kd + c] = disc[row * ldb];
}
"""

_MODULES: dict = {}
_LOCK = threading.Lock()
_MAX_GRID_Y = 65535


def _module(cp: Any, c_type: str) -> Any:
    """The compiled gather kernels for one code type, built once per process."""
    with _LOCK:
        mod = _MODULES.get(c_type)
        if mod is None:
            mod = cp.RawModule(code=_SRC.replace("{DT}", c_type))
            _MODULES[c_type] = mod
    return mod


def _column_table(cp: Any, parts: "list") -> "tuple[Any, Any]":
    """``(address, row stride)`` of every global column of the family: the address of its first element and the row stride, in elements, of its batch (``parts``: row-major batches with a unit column stride, one dtype)."""
    item = int(parts[0].dtype.itemsize)
    widths = np.array([int(d.shape[1]) for d in parts], dtype=np.int64)
    first = np.concatenate([[0], np.cumsum(widths)])[:-1]
    base = np.array([int(d.data.ptr) for d in parts], dtype=np.uint64)
    strides = np.array([int(d.strides[0]) // item for d in parts], dtype=np.int64)
    within = (np.arange(int(widths.sum()), dtype=np.int64) - np.repeat(first, widths)).astype(np.uint64)
    return cp.asarray(np.repeat(base, widths) + within * np.uint64(item)), cp.asarray(np.repeat(strides, widths))


def transposed_codes(cp: Any, parts: "list", n: int) -> Any:
    """The family as a C-contiguous ``(columns, n)`` device matrix of the batches' dtype, read in place (``parts`` as for ``_column_table``)."""
    dtype = parts[0].dtype
    k_total = int(sum(int(d.shape[1]) for d in parts))
    colptr, colld = _column_table(cp, parts)
    out = cp.full((k_total, int(n)), np.iinfo(dtype).max, dtype=dtype)  # a sentinel, not zeros: a cell the kernel misses must not look like a code
    kernel = _module(cp, _C_TYPES[dtype]).get_function("transpose_multi")
    grid = ((k_total + 31) // 32, max(1, min(_MAX_GRID_Y, (int(n) + 31) // 32)))
    kernel(grid, (32, 8), (colptr, colld, np.int64(n), np.int32(k_total), out))  # launch-ok: the 32 x 8 block is the geometry of the 32 x 32 shared tile (each thread moves four rows of it), not an occupancy choice
    return out


def gathered_codes(cp: Any, parts: "list", n: int, chosen: np.ndarray) -> Any:
    """The ``chosen`` global columns of the family as a C-contiguous ``(n, len(chosen))`` device matrix of the batches' dtype."""
    dtype = parts[0].dtype
    kd = int(chosen.shape[0])
    colptr, colld = _column_table(cp, parts)
    out = cp.full((int(n), kd), np.iinfo(dtype).max, dtype=dtype)
    kernel = _module(cp, _C_TYPES[dtype]).get_function("gather_multi")
    block = block_size_for(kernel, work_items=kd)
    grid = ((kd + block - 1) // block, max(1, min(_MAX_GRID_Y, int(n))))
    kernel(grid, (block,), (colptr, colld, np.int64(n), cp.asarray(np.ascontiguousarray(chosen, dtype=np.int32)), np.int32(kd), out))
    return out
