"""Skinny-matrix kernels for the resident ALS sweep.

cupy's matmul on an (n x d) design with d ~ 6 runs at a small fraction of memory bandwidth (20 ms per matvec and 23 ms per Gram product at n=1M, for 48 MB of
data). These kernels read the design once: ``design_matvec`` is one row per thread, ``weighted_gram`` accumulates ``A'A`` and ``A'y`` for ``A = B * w[:, None]``
in registers, reduces by warp shuffle and a fixed-order block reduction, and sums the per-block partials with a deterministic cupy reduction (no atomics). Results agree with the cupy products to summation-order rounding (~1e-13).
"""

from __future__ import annotations

import threading
from typing import Any

from mlframe.feature_selection.filters._raw_module_cache import RawModuleCache

_LOCK = threading.Lock()

_SRC = r"""
extern "C" {
__global__ void design_matvec(const double* B, const double* c, double* out, const long long n) {
    double cc[D];
    #pragma unroll
    for (int k = 0; k < D; ++k) cc[k] = c[k];
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double* r = B + i * D;
        double s = 0.0;
        #pragma unroll
        for (int k = 0; k < D; ++k) s += r[k] * cc[k];
        out[i] = s;
    }
}
__global__ void weighted_gram(const double* B, const double* w, const double* y, double* part, const long long n, const int use_w) {
    double acc[NACC];
    #pragma unroll
    for (int t = 0; t < NACC; ++t) acc[t] = 0.0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double* r = B + i * D;
        const double wi = use_w ? w[i] : 1.0;
        double a[D];
        #pragma unroll
        for (int k = 0; k < D; ++k) a[k] = r[k] * wi;
        const double yi = y[i];
        int t = 0;
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            #pragma unroll
            for (int k = j; k < D; ++k) { acc[t] += a[j] * a[k]; ++t; }
            acc[t] += a[j] * yi; ++t;
        }
    }
    __shared__ double warp_sums[8][NACC];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int t = 0; t < NACC; ++t) {
        double v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) warp_sums[warp][t] = v;
    }
    __syncthreads();
    // Fixed-order block reduction: no atomics, so the sums are bit-reproducible run to run.
    if (threadIdx.x < NACC) {
        double v = 0.0;
        for (int wv = 0; wv < 8; ++wv) v += warp_sums[wv][threadIdx.x];
        part[(long long)blockIdx.x * NACC + threadIdx.x] = v;
    }
}
}
"""


_MODULES = RawModuleCache(_SRC)


def _grid(cp, n: int) -> int:
    """Block count for a grid-stride launch over ``n`` rows."""
    return int(min(4096, max(1, (n + 255) // 256)))


def design_matvec(cp: Any, B: Any, c: Any) -> Any:
    """``B @ c`` for a C-contiguous float64 ``(n, d)`` design and a ``(d,)`` coefficient vector."""
    n, d = B.shape
    out = cp.empty(n, dtype=cp.float64)
    c = cp.ascontiguousarray(c, dtype=cp.float64)
    _MODULES.get(cp, d).get_function("design_matvec")((_grid(cp, n),), (256,), (B, c, out, cp.int64(n)))
    return out


def weighted_gram(cp: Any, B: Any, w: Any, y: Any) -> "tuple[Any, Any]":
    """``(A'A, A'y)`` for ``A = B * w[:, None]`` (``w=None`` means ``A = B``); ``B`` is C-contiguous float64 ``(n, d)``, ``y`` is ``(n,)``."""
    n, d = B.shape
    nacc = d * (d + 1) // 2 + d
    blocks = _grid(cp, n)
    part = cp.empty((blocks, nacc), dtype=cp.float64)
    y64 = cp.ascontiguousarray(y, dtype=cp.float64)
    w64 = y64 if w is None else cp.ascontiguousarray(w, dtype=cp.float64)
    _MODULES.get(cp, d).get_function("weighted_gram")((blocks,), (256,), (B, w64, y64, part, cp.int64(n), cp.int32(0 if w is None else 1)))
    out = part.sum(axis=0)
    ata_idx, atb_idx = _unpack_index(cp, d)
    return out[ata_idx], out[atb_idx]


_UNPACK: dict = {}


def _unpack_index(cp, d: int):
    """Device gather indices that rebuild the symmetric ``(d, d)`` Gram matrix and the ``(d,)`` right-hand side from the packed accumulator vector."""
    with _LOCK:
        hit = _UNPACK.get(d)
        if hit is None:
            import numpy as np

            pos = {}
            t = 0
            atb = np.empty(d, dtype=np.int64)
            for j in range(d):
                for k in range(j, d):
                    pos[(j, k)] = t
                    t += 1
                atb[j] = t
                t += 1
            ata = np.array([[pos[(min(j, k), max(j, k))] for k in range(d)] for j in range(d)], dtype=np.int64)
            hit = (cp.asarray(ata), cp.asarray(atb))
            _UNPACK[d] = hit
        return hit
