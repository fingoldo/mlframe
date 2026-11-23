"""Content fingerprints of the binned candidate columns of the greedy CMI construction, the whole pool in one launch.

``fp[c] = sum_i (code[c, i] + 1) * w(i)`` over int64 with wrap-around, ``w(i)`` the splitmix64 mix of ``i + 1`` (``_mi_greedy_cmi_fe_steps._fingerprint_weights``): the sum is taken modulo 2**64, so the order of
the additions does not matter and the value equals the one the per-candidate chain (``astype``, ``+ 1``, ``* weights``, ``sum``: four to five launches a candidate) gives.
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np

_SRC = r"""
extern "C" __global__ void bin_fingerprints(const CT_* __restrict__ codes, const long long n, unsigned long long* __restrict__ out) {
    __shared__ unsigned long long red[256];
    const CT_* row = codes + (long long)blockIdx.x * n;
    unsigned long long acc = 0ULL;
    for (long long i = threadIdx.x; i < n; i += blockDim.x)
    {
        unsigned long long z = ((unsigned long long)(i + 1LL)) * 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z = z ^ (z >> 31);
        acc += ((unsigned long long)((long long)row[i] + 1LL)) * z;
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) { if ((int)threadIdx.x < st) red[threadIdx.x] += red[threadIdx.x + st]; __syncthreads(); }
    if (threadIdx.x == 0) out[blockIdx.x] = red[0];
}
"""
_C_TYPES = {"int8": "signed char", "uint8": "unsigned char", "int16": "short", "uint16": "unsigned short", "int32": "int", "uint32": "unsigned int", "int64": "long long"}
_KERNELS: dict = {}
_LOCK = threading.Lock()


def kernel_bin_fingerprints(cp: Any, real: Any = np.int32) -> Any:
    """``bin_fingerprints`` for a code matrix of dtype ``real`` (any integer dtype of ``_C_TYPES``); compile point of :func:`pool_fingerprints` and of the register-pressure registry."""
    key = np.dtype(real).name
    with _LOCK:
        kern = _KERNELS.get(key)
        if kern is None:
            kern = _KERNELS[key] = cp.RawKernel(_SRC.replace("CT_", _C_TYPES[key]), "bin_fingerprints")
    return kern


_STACK_BLOCK_BYTES = 256 * 1024 * 1024  # the most one stacked block of code columns may take; a fraction of the room below it still applies


def _block_fingerprints(cp: Any, columns: Sequence[Any]) -> "list[int]":
    """The fingerprints of ``columns`` (all the same length) from one stack, one launch and one host read."""
    block = cp.stack(list(columns)) if len(columns) > 1 else columns[0][None, :]
    if block.dtype.name not in _C_TYPES:
        block = block.astype(cp.int64)
    block = cp.ascontiguousarray(block)
    k, n = int(block.shape[0]), int(block.shape[1])
    out = cp.empty(k, dtype=cp.uint64)
    kernel_bin_fingerprints(cp, block.dtype)((k,), (256,), (block, np.int64(n), out))  # launch-ok: the block width is the compile-time size of the kernel's shared reduction buffer
    return [int(v) for v in cp.asnumpy(out.view(cp.int64)).tolist()]


def pool_fingerprints(cp: Any, columns: Sequence[Any]) -> "list[int]":
    """The fingerprint of every device code column of ``columns`` (all the same length), as Python ints.

    The columns are stacked in blocks of at most ``_STACK_BLOCK_BYTES`` (and a quarter of the room under the pool limit), one launch and one host read per block. The columns are independent
    (one kernel block each), so a column's fingerprint is the one a single stack of the whole pool gives. One stack of a 3871-column pool at 500000 rows was a second 1.9 GB copy of codes
    that were already resident, which ran the pool out of memory and sent the fit to the host path."""
    cols = list(columns)
    if len(cols) <= 1:
        return _block_fingerprints(cp, cols)
    from ._device_sort_safe import device_room

    n = int(cols[0].shape[0])
    per_col = max(1, n * max(8 if c.dtype.name not in _C_TYPES else int(c.dtype.itemsize) for c in cols))
    budget = max(per_col, min(_STACK_BLOCK_BYTES, int(device_room(cp)) // 4))
    col_block = max(1, budget // per_col)  # columns per stacked block: sized by the device budget, one launch per block
    fps: "list[int]" = []
    for c0 in range(0, len(cols), col_block):
        fps.extend(_block_fingerprints(cp, cols[c0 : c0 + col_block]))
    return fps
