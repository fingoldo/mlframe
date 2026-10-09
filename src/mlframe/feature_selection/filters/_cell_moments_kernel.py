"""Per-cell weighted moment sums in two fused passes, with the histogram kept in shared memory.

The scatter-add form (``resident_bincount``) fires one global-atomic scatter per moment - five launches per (pair, fold), every thread contending for the few dozen cell
addresses. Here pass 1 returns ``(count, sum)`` and pass 2 the centred ``(sum d^2, sum d^3, sum d^4)`` from one kernel each: every block accumulates into a shared-memory
histogram and merges it into the global one once. The arithmetic per row is the same expressions as the scatter form (``w``, ``v*w``, ``d2*w``, ``d2*d*w``, ``d2*d2*w``).
"""

from __future__ import annotations

from typing import Any, Optional

# Shared-memory budget (doubles) per block: 3 accumulators x n_cells must fit in 48 KB.
MAX_CELLS = 2048

_PASS1_SRC = r"""
extern "C" __global__ void cell_sums(const long long* codes, const double* v, const double* w, long long n, int nc, double* cnt_out, double* sum_out) {
    extern __shared__ double sh[];
    for (int i = threadIdx.x; i < 2 * nc; i += blockDim.x) sh[i] = 0.0;
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = (int)codes[i];
        double wi = w[i];
        atomicAdd(&sh[c], wi);
        atomicAdd(&sh[nc + c], v[i] * wi);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nc; i += blockDim.x) {
        atomicAdd(&cnt_out[i], sh[i]);
        atomicAdd(&sum_out[i], sh[nc + i]);
    }
}
"""

_PASS2_SRC = r"""
extern "C" __global__ void cell_centred(const long long* codes, const double* v, const double* w, const double* mean, long long n, int nc,
                                         double* cm2_out, double* cm3_out, double* cm4_out) {
    extern __shared__ double sh[];
    for (int i = threadIdx.x; i < 3 * nc; i += blockDim.x) sh[i] = 0.0;
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = (int)codes[i];
        double wi = w[i];
        double d = v[i] - mean[c];
        double d2 = d * d;
        atomicAdd(&sh[c], d2 * wi);
        atomicAdd(&sh[nc + c], d2 * d * wi);
        atomicAdd(&sh[2 * nc + c], d2 * d2 * wi);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nc; i += blockDim.x) {
        atomicAdd(&cm2_out[i], sh[i]);
        atomicAdd(&cm3_out[i], sh[nc + i]);
        atomicAdd(&cm4_out[i], sh[2 * nc + i]);
    }
}
"""

_KERNELS: Optional[tuple] = None
_UNAVAILABLE = False


def _kernels(cp) -> Optional[tuple]:
    """Compile (once) the two kernels; ``None`` when the device cannot build them (shared-memory double atomics need compute capability 6.0)."""
    global _KERNELS, _UNAVAILABLE
    if _KERNELS is not None or _UNAVAILABLE:
        return _KERNELS
    try:
        _KERNELS = (cp.RawKernel(_PASS1_SRC, "cell_sums"), cp.RawKernel(_PASS2_SRC, "cell_centred"))
        _KERNELS[0].compile()
        _KERNELS[1].compile()
    except Exception:
        _KERNELS = None
        _UNAVAILABLE = True
    return _KERNELS


def masked_cell_moments(cp, codes_g: Any, v_safe: Any, w: Any, n_cells: int) -> Optional[tuple]:
    """``(cnt, mean, cm2, cm3, cm4)`` per cell over the rows weighted by ``w`` (``codes_g`` int64 in ``[0, n_cells)``, ``v_safe`` finite float64), or ``None`` when the
    fused kernels cannot be used (too many cells for shared memory, or no support) and the caller should take the scatter-add form."""
    nc = int(n_cells)
    if nc > MAX_CELLS or nc < 1:
        return None
    kernels = _kernels(cp)
    if kernels is None:
        return None
    k1, k2 = kernels
    n = int(codes_g.shape[0])
    threads = 256
    blocks = max(1, min((n + threads - 1) // threads, 4 * cp.cuda.Device().attributes["MultiProcessorCount"]))
    codes = codes_g if codes_g.dtype == cp.int64 else codes_g.astype(cp.int64)
    cnt = cp.zeros(nc, dtype=cp.float64)
    s1 = cp.zeros(nc, dtype=cp.float64)
    k1((blocks,), (threads,), (codes, v_safe, w, cp.int64(n), cp.int32(nc), cnt, s1), shared_mem=2 * nc * 8)
    mean = s1 / cp.maximum(cnt, 1.0)
    cm2 = cp.zeros(nc, dtype=cp.float64)
    cm3 = cp.zeros(nc, dtype=cp.float64)
    cm4 = cp.zeros(nc, dtype=cp.float64)
    k2((blocks,), (threads,), (codes, v_safe, w, mean, cp.int64(n), cp.int32(nc), cm2, cm3, cm4), shared_mem=3 * nc * 8)
    return cnt, mean, cm2, cm3, cm4
