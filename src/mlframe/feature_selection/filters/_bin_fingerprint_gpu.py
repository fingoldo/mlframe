"""Content fingerprints of the binned candidate columns of the greedy CMI construction, the whole pool in one launch.

``fp[c] = sum_i (code[c, i] + 1) * ((i + 1) * 2654435761)`` over int64 with wrap-around: the sum is taken modulo 2**64, so the order of the additions does not matter and the value equals the one the
per-candidate chain (``astype``, ``+ 1``, ``* weights``, ``sum``: four to five launches a candidate) gave.
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
        acc += ((unsigned long long)((long long)row[i] + 1LL)) * (((unsigned long long)(i + 1LL)) * 2654435761ULL);
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


def pool_fingerprints(cp: Any, columns: Sequence[Any]) -> "list[int]":
    """The fingerprint of every device code column of ``columns`` (all the same length), as Python ints, with one stack, one launch and one host read."""
    block = cp.stack(list(columns)) if len(columns) > 1 else columns[0][None, :]
    if block.dtype.name not in _C_TYPES:
        block = block.astype(cp.int64)
    block = cp.ascontiguousarray(block)
    k, n = int(block.shape[0]), int(block.shape[1])
    out = cp.empty(k, dtype=cp.uint64)
    kernel_bin_fingerprints(cp, block.dtype)((k,), (256,), (block, np.int64(n), out))  # launch-ok: the block width is the compile-time size of the kernel's shared reduction buffer
    return [int(v) for v in cp.asnumpy(out.view(cp.int64)).tolist()]
