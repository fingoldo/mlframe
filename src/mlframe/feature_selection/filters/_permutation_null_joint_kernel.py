"""Fused joint counts of the resident pooled-gain-floor permutation null: one launch builds every (candidate, shuffle) contingency table.

``pooled_gain_floor_perms_cupy`` used to build the flat cell index of every sample as cupy expressions: ``x_codes[:, None, :] + yp[None, :, :]`` as an int64 ``(cc, pb, n)`` array, a second int64
array with the (candidate, shuffle) base added, and then ``cupyx.scatter_add`` of ones into the counts: about 40 bytes of traffic per sample and three launches (nvprof of the warm fit,
100000 x 100: ``cupy_scatter_add`` 620 launches of 11 ms, the int64 add 620 launches of 6 ms, copies besides: 5 percent of the fit's GPU time). Here a block owns one (candidate, shuffle)
pair, its threads read the candidate's codes and the shuffle's codes in order (both coalesced, at their own widths), add them into the table and flush it to the ``(cc * pb, max_jc)`` int32
counts; the table lives in shared memory (``max_jc`` is the product of two bin counts, a few hundred cells in practice; a table over 48 KB is left to the caller's scatter-add path). Counts are integers, so the
result is identical to the scatter-add histogram whatever the order of the additions.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

_C_TYPES = {
    np.dtype(np.int8): "signed char",
    np.dtype(np.uint8): "unsigned char",
    np.dtype(np.int16): "short",
    np.dtype(np.uint16): "unsigned short",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "unsigned int",
    np.dtype(np.int64): "long long",
}
_SHARED_TABLE_BYTES = 48 * 1024  # largest per-block table (the default dynamic shared-memory limit); a wider one is left to the caller's scatter-add path: global atomics lost 2x to it at 12000 cells

_SRC = r"""
extern "C" __global__
void perm_joint_counts(const {XT}* __restrict__ x, const {YT}* __restrict__ yp, const long long n, const int pb, const int max_jc,
                       int* __restrict__ counts) {
    // grid (candidates, shuffles): block (c, p) builds the contingency table of candidate c against shuffle p in shared memory. The cell of a sample is x[c, r] + yp[p, r] (x already
    // holds the candidate's block of cells, y the class code), which is below max_jc by construction.
    extern __shared__ int table[];
    const int c = blockIdx.x;
    const int p = blockIdx.y;
    const {XT}* xc = x + (long long)c * n;
    const {YT}* yr = yp + (long long)p * n;
    int* out = counts + ((long long)c * pb + p) * (long long)max_jc;
    for (int i = threadIdx.x; i < max_jc; i += blockDim.x) table[i] = 0;
    __syncthreads();
    for (long long r = threadIdx.x; r < n; r += blockDim.x) atomicAdd(&table[(int)xc[r] + (int)yr[r]], 1);
    __syncthreads();
    for (int i = threadIdx.x; i < max_jc; i += blockDim.x) out[i] = table[i];
}
"""

_KERNELS: dict = {}
_LOCK = threading.Lock()


def _c_type(dtype: Any) -> str:
    """The CUDA type of an integer code array, or ``long long`` for a type this kernel has no spelling for (the caller widens first)."""
    return _C_TYPES.get(np.dtype(dtype), "long long")


def _get_kernel(cp: Any, xt: str, yt: str) -> Any:
    """Compile (once per code type pair) the joint-count kernel."""
    key = (xt, yt)
    with _LOCK:
        kern = _KERNELS.get(key)
        if kern is None:
            src = _SRC.replace("{XT}", xt).replace("{YT}", yt)
            kern = _KERNELS[key] = cp.RawKernel(src, "perm_joint_counts")
    return kern


def perm_joint_counts(cp: Any, x_codes: Any, yp: Any, max_jc: int) -> Any:
    """``(cc * pb, max_jc)`` int32 contingency tables (``None`` when one table does not fit the shared-memory limit: the caller keeps the scatter-add path) of the ``cc`` candidate rows of ``x_codes`` ``(cc, n)`` against the ``pb`` shuffles of ``yp`` ``(pb, n)``; row ``c * pb + p`` is the table
    of candidate ``c`` against shuffle ``p``. Both code arrays are read at their own integer width; a code is non-negative and ``x + y < max_jc``."""
    if int(max_jc) * 4 > _SHARED_TABLE_BYTES:
        return None
    cc, n = int(x_codes.shape[0]), int(x_codes.shape[1])
    pb = int(yp.shape[0])
    x = cp.ascontiguousarray(x_codes)
    y = cp.ascontiguousarray(yp)
    xt = _c_type(x.dtype)
    yt = _c_type(y.dtype)
    if xt == "long long" and x.dtype != cp.int64:
        x = x.astype(cp.int64)
    if yt == "long long" and y.dtype != cp.int64:
        y = y.astype(cp.int64)
    kern = _get_kernel(cp, xt, yt)
    counts = cp.empty(cc * pb * int(max_jc), dtype=cp.int32)
    threads = block_size_for(kern, dyn_smem=int(max_jc) * 4, work_items=n, pow2=True, max_width=1024)
    kern((cc, pb), (threads,), (x, y, np.int64(n), np.int32(pb), np.int32(max_jc), counts), shared_mem=int(max_jc) * 4)
    return counts.reshape(cc * pb, int(max_jc))
