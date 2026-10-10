"""Per-cell weighted moment sums in two fused passes, with the histogram kept in shared memory.

The scatter-add form (``resident_bincount``) fires one global-atomic scatter per moment - five launches per (pair, fold), every thread contending for the few dozen cell
addresses. Here pass 1 returns ``(count, sum)`` and pass 2 the centred ``(sum d^2, sum d^3, sum d^4)`` from one kernel each: every block accumulates into a shared-memory
histogram and merges it into the global one once. The arithmetic per row is the same expressions as the scatter form (``w``, ``v*w``, ``d2*w``, ``d2*d*w``, ``d2*d2*w``).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

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
    except Exception as exc:
        _KERNELS = None
        _UNAVAILABLE = True
        logger.warning("fused cell-moment kernels unavailable (%s: %s); per-cell moments use the slower scatter-add form for this process", type(exc).__name__, exc)
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


# --- all folds in two launches -------------------------------------------------------------------------------------------------------------------------------------------------
# A fold's train rows are every finite row outside the fold. ``masked_cell_moments`` is launched per fold (two launches, a weight vector and the output arrays each), and the per-fold
# tables then have to be stacked for the OOF gather. Here a thread handles a row for ALL folds at once: the row adds to the histogram of every fold except its own, into a shared
# ``(n_folds, n_cells)`` histogram that is merged into the global ``(n_folds, n_cells)`` tables once per block. Per-row terms are the expressions of the per-fold kernels (``w`` is 1.0 on a
# train row, and a test row would add an exact 0.0), so the tables hold the same moments; only the order of the atomic additions differs, as it already does between runs.
_FOLDS_PASS1_SRC = r"""
extern "C" __global__ void fold_cell_sums(const long long* codes, const double* v, const double* fin, const long long* fold, long long n, int nc, int nf, double* cnt_out, double* sum_out) {
    extern __shared__ double sh[];
    for (int i = threadIdx.x; i < 2 * nf * nc; i += blockDim.x) sh[i] = 0.0;
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = (int)codes[i];
        int fi = (int)fold[i];
        double wi = fin[i];
        double vi = v[i] * wi;
        for (int f = 0; f < nf; ++f) {
            if (f != fi) {
                atomicAdd(&sh[f * nc + c], wi);
                atomicAdd(&sh[nf * nc + f * nc + c], vi);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nf * nc; i += blockDim.x) {
        atomicAdd(&cnt_out[i], sh[i]);
        atomicAdd(&sum_out[i], sh[nf * nc + i]);
    }
}
"""

_FOLDS_PASS2_SRC = r"""
extern "C" __global__ void fold_cell_centred(const long long* codes, const double* v, const double* fin, const long long* fold, const double* mean, long long n, int nc, int nf,
                                              double* cm2_out, double* cm3_out, double* cm4_out) {
    extern __shared__ double sh[];
    for (int i = threadIdx.x; i < 3 * nf * nc; i += blockDim.x) sh[i] = 0.0;
    __syncthreads();
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        int c = (int)codes[i];
        int fi = (int)fold[i];
        double wi = fin[i];
        double vi = v[i];
        for (int f = 0; f < nf; ++f) {
            if (f != fi) {
                double d = vi - mean[f * nc + c];
                double d2 = d * d;
                atomicAdd(&sh[f * nc + c], d2 * wi);
                atomicAdd(&sh[nf * nc + f * nc + c], d2 * d * wi);
                atomicAdd(&sh[2 * nf * nc + f * nc + c], d2 * d2 * wi);
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < nf * nc; i += blockDim.x) {
        atomicAdd(&cm2_out[i], sh[i]);
        atomicAdd(&cm3_out[i], sh[nf * nc + i]);
        atomicAdd(&cm4_out[i], sh[2 * nf * nc + i]);
    }
}
"""

_FOLD_KERNELS: Optional[tuple] = None
_FOLD_UNAVAILABLE = False
_SHARED_DOUBLES = 6144  # 48 KB of shared memory per block


def _fold_kernels(cp) -> Optional[tuple]:
    """Compile (once) the two all-folds kernels; ``None`` when the device cannot build them."""
    global _FOLD_KERNELS, _FOLD_UNAVAILABLE
    if _FOLD_KERNELS is not None or _FOLD_UNAVAILABLE:
        return _FOLD_KERNELS
    try:
        _FOLD_KERNELS = (cp.RawKernel(_FOLDS_PASS1_SRC, "fold_cell_sums"), cp.RawKernel(_FOLDS_PASS2_SRC, "fold_cell_centred"))
        _FOLD_KERNELS[0].compile()
        _FOLD_KERNELS[1].compile()
    except Exception as exc:
        log_throttle(logger, "cell_moments.fold_kernels", logging.WARNING, "cell moments: the all-folds kernels could not be built (%s: %s), the per-fold form is used", type(exc).__name__, exc)
        _FOLD_KERNELS = None
        _FOLD_UNAVAILABLE = True
    return _FOLD_KERNELS


def fold_cell_moments(cp, codes_g: Any, v_safe: Any, finite_f: Any, fold_g: Any, n_folds: int, n_cells: int) -> Optional[tuple]:
    """``(cnt, mean, cm2, cm3, cm4)`` as ``(n_folds, n_cells)`` tables: row ``f`` holds the moments of the TRAIN rows of fold ``f`` (every finite row whose fold is not ``f``).

    Two moment launches for all folds instead of two per fold, and the tables come out stacked. ``None`` when the kernels cannot be used (``3 * n_folds * n_cells`` doubles must fit in
    shared memory) so the caller takes the per-fold form."""
    nf, nc = int(n_folds), int(n_cells)
    if nf < 1 or nc < 1 or 3 * nf * nc > _SHARED_DOUBLES:
        return None
    kernels = _fold_kernels(cp)
    if kernels is None:
        return None
    k1, k2 = kernels
    n = int(codes_g.shape[0])
    threads = 256
    blocks = max(1, min((n + threads - 1) // threads, 4 * cp.cuda.Device().attributes["MultiProcessorCount"]))
    codes = codes_g if codes_g.dtype == cp.int64 else codes_g.astype(cp.int64)
    folds = fold_g if fold_g.dtype == cp.int64 else fold_g.astype(cp.int64)
    cnt = cp.zeros((nf, nc), dtype=cp.float64)
    s1 = cp.zeros((nf, nc), dtype=cp.float64)
    k1((blocks,), (threads,), (codes, v_safe, finite_f, folds, cp.int64(n), cp.int32(nc), cp.int32(nf), cnt, s1), shared_mem=2 * nf * nc * 8)
    mean = s1 / cp.maximum(cnt, 1.0)
    cm2 = cp.zeros((nf, nc), dtype=cp.float64)
    cm3 = cp.zeros((nf, nc), dtype=cp.float64)
    cm4 = cp.zeros((nf, nc), dtype=cp.float64)
    k2((blocks,), (threads,), (codes, v_safe, finite_f, folds, mean, cp.int64(n), cp.int32(nc), cp.int32(nf), cm2, cm3, cm4), shared_mem=3 * nf * nc * 8)
    return cnt, mean, cm2, cm3, cm4
