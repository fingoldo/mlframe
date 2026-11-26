"""Device twin of ``_family_columns_dedupe.unique_columns``: the distinct partitions of a family's code columns without bringing the columns to the host.

The family gate compared its columns on the host: the ``(columns, rows)`` codes were copied down (up to 512 MB) and hashed and checked by two numba kernels. In a 300000 x 300 fit the main thread
spent 1.8 s waiting for that copy and 3.0 s in the host comparison. Here both steps run where the codes already are, and only the ``k`` hashes and ``k`` flags come back:

* ``partition_hash`` (one block per column): the row of the first appearance of every code (``atomicMin`` in shared memory), each code's label is the rank of that row among the codes present
  (the first-appearance relabelling of the host kernel, so columns inducing the same partition get the same labels row by row), and the hash is an order-free sum of a 64-bit mix of
  ``(row, label)`` over the rows (equal label sequences, equal sums).
* ``same_partition`` (one block per column with a representative): the code maps between a column and its representative must be one to one, checked with ``atomicCAS`` in shared memory.

The grouping between the two (rows with equal hashes, the first of each group its representative) is the host function's on ``k`` numbers, and a failed check returns ``None`` as the host one does
on a collision, so ``(keep, inverse)`` are those of ``unique_columns``. Codes are non-negative and below ``MAX_CODES``; anything else, or any device failure, returns ``None`` and the caller
takes the host path.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from mlframe.utils.log_throttle import log_throttle

from ._batch_mi_noise_gate_bits import _C_TYPES, typed_module

logger = logging.getLogger(__name__)

#: Codes per column the shared-memory tables hold (two int tables of this size).
MAX_CODES = 2048
_THREADS = 256

_SRC = r"""
__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

extern "C" __global__ void partition_hash(const {DT}* __restrict__ cols, const long long n, const int n_codes, unsigned long long* __restrict__ out) {
    extern __shared__ int sh[];
    int* first = sh;
    int* label = sh + n_codes;
    __shared__ unsigned long long red[256];
    const int tid = threadIdx.x, nt = blockDim.x;
    const {DT}* __restrict__ row = cols + (long long)blockIdx.x * n;
    for (int c = tid; c < n_codes; c += nt) first[c] = 0x7FFFFFFF;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) atomicMin(&first[(int)row[i]], (int)i);
    __syncthreads();
    for (int c = tid; c < n_codes; c += nt) {
        const int f = first[c];
        int r = 0;
        if (f != 0x7FFFFFFF) {
            for (int d = 0; d < n_codes; ++d) r += (first[d] < f) ? 1 : 0;
        }
        label[c] = r;
    }
    __syncthreads();
    unsigned long long acc = 0ull;
    for (long long i = tid; i < n; i += nt) acc += mix64(((unsigned long long)i << 20) ^ (unsigned long long)label[(int)row[i]]);
    red[tid] = acc;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = red[0];
}

extern "C" __global__ void same_partition(const {DT}* __restrict__ cols, const long long* __restrict__ rep, const long long n, const int n_codes, int* __restrict__ bad) {
    extern __shared__ int sh[];
    int* fwd = sh;
    int* back = sh + n_codes;
    __shared__ int flag;
    const int j = blockIdx.x;
    const long long r = rep[j];
    if (r == (long long)j) return;
    const int tid = threadIdx.x, nt = blockDim.x;
    for (int c = tid; c < n_codes; c += nt) { fwd[c] = -1; back[c] = -1; }
    if (tid == 0) flag = 0;
    __syncthreads();
    const {DT}* __restrict__ a_row = cols + (long long)j * n;
    const {DT}* __restrict__ b_row = cols + r * n;
    for (long long i = tid; i < n; i += nt) {
        const int a = (int)a_row[i], b = (int)b_row[i];
        const int fa = atomicCAS(&fwd[a], -1, b);
        const int bb = atomicCAS(&back[b], -1, a);
        if ((fa != -1 && fa != b) || (bb != -1 && bb != a)) flag = 1;
    }
    __syncthreads();
    if (tid == 0 && flag) bad[j] = 1;
}
"""


def unique_columns_device(cp: Any, columns_first: Any) -> "tuple[np.ndarray, np.ndarray] | Literal[False] | None":
    """``unique_columns`` of the device matrix ``columns_first`` ``(k, n)``: ``(keep, inverse)``, ``None`` when every column has its own partition (or a hash collision makes the grouping
    unsafe), ``False`` when the device path does not apply (the caller then takes the host path)."""
    try:
        k, n = int(columns_first.shape[0]), int(columns_first.shape[1])
        if k < 2:
            return None
        if columns_first.dtype not in _C_TYPES or not columns_first.flags.c_contiguous or n >= (1 << 31):
            return False
        lo, hi = int(columns_first.min()), int(columns_first.max())
        if lo < 0 or hi >= MAX_CODES:
            return False
        n_codes = hi + 1
        mod = typed_module(cp, _SRC, _C_TYPES[np.dtype(columns_first.dtype)])
        hashes = cp.empty(k, dtype=cp.uint64)
        mod.get_function("partition_hash")((k,), (_THREADS,), (columns_first, np.int64(n), np.int32(n_codes), hashes), shared_mem=2 * n_codes * 4)  # launch-ok: 256 threads, the size of the static reduction array
        hv = cp.asnumpy(hashes)
        order = np.argsort(hv, kind="stable")
        hs = hv[order]
        new_group = np.empty(k, dtype=np.bool_)
        new_group[0] = True
        new_group[1:] = hs[1:] != hs[:-1]
        if bool(new_group.all()):
            return None
        group_of_sorted = np.cumsum(new_group) - 1
        rep = np.empty(k, dtype=np.int64)
        rep[order] = order[new_group][group_of_sorted]
        bad = cp.zeros(k, dtype=cp.int32)
        mod.get_function("same_partition")((k,), (_THREADS,), (columns_first, cp.asarray(rep), np.int64(n), np.int32(n_codes), bad), shared_mem=2 * n_codes * 4)  # launch-ok: one block per column, a stride loop over its rows
        if bool(cp.asnumpy(bad).any()):
            return None
        keep = np.flatnonzero(rep == np.arange(k))
        slot = np.full(k, -1, dtype=np.int64)
        slot[keep] = np.arange(keep.shape[0])
        return keep, slot[rep]
    except Exception as exc:  # the host comparison answers
        log_throttle(logger, "family_columns_dedupe_cuda.failed", logging.WARNING, "device partition comparison failed (%s: %s); the host compares the columns", type(exc).__name__, exc)
        return False


__all__ = ["MAX_CODES", "unique_columns_device"]
