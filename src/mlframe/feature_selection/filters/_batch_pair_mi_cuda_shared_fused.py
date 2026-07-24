"""Single-launch, full-pass CuPy ``RawKernel`` pair-MI backend using OPT-IN extended dynamic shared memory.

Motivation (2026-07-16, wellbore-100k profiling, quiet machine): the production shape (n_classes_y=20,
max_joint up to 441 for a 21x21 numeric-numeric pair) exceeds ``MAX_JOINT_BINS_CUDA=256`` /
``MAX_Y_BINS_CUDA=16`` -- compile-time caps on :func:`batch_pair_mi_cuda`'s STATIC (``cuda.shared.array``
with a fixed shape) shared-memory histogram, sized to fit the conservative 48KB-per-block budget every
CUDA device guarantees without opt-in. Raising those two constants was already tried and rejected
(2026-07-14, see ``_batch_pair_mi_cuda_kernels.py``'s ``MAX_Y_BINS_CUDA`` docstring): even the max value
that still fits 48KB static shared memory (joint<=256) is smaller than this shape's ``max_joint=441``, so
the full kernel stays rejected regardless of the Y-bins cap. Every rejection falls through to
:func:`batch_pair_mi_cuda_row_chunked`, whose fragmentation-driven per-launch overhead measured 78-92s of
a ~500-585s wellbore-100k fit wall (cProfile, clean/uncontended machine) -- the single largest reproducible
CPU-orchestration-adjacent hotspot in that profile.

The row-chunked kernel's OWN inner histogram kernel is already efficient (shared-memory-staged, see
``_cuda_hist_kernel_shared_factory``) -- the actual cost is the OUTER row/pair-chunking loop, which exists
solely because that kernel accumulates into a device-PERSISTENT GLOBAL buffer across multiple launches (a
requirement of processing ``factors_data`` in row-chunks). At the wellbore-100k scale ``factors_data`` is
~206MB (99401 rows x 518 cols x int32) -- it fits a 4GB card's VRAM whole, so row-chunking is unnecessary
architecturally; only the STATIC 48KB shared-memory cap forced the chunked fallback.

This module removes BOTH constraints in one step: a RawKernel (not ``numba.cuda.jit``, whose shared memory
is always a Python-level, compile-time-constant-shaped ``cuda.shared.array`` and offers no supported way to
opt into >48KB dynamic shared memory in this numba version -- checked directly, no public API found) using
``extern __shared__`` DYNAMIC shared memory, sized at RUNTIME to the actual ``(max_joint, n_classes_y)``
of the call, with the kernel's ``max_dynamic_shared_size_bytes`` property (CuPy-native, verified working)
set to opt into the device's EXTENDED per-block budget wherever the driver allows it (cc>=7.0: 96KB+;
verified live on this host's RTX 500 Ada Laptop GPU, cc 8.9: 48KB static vs 99KB opt-in). One kernel
launch (grid=n_pairs blocks) walks the FULL ``n_samples`` in one pass per pair, accumulates the joint
histogram in shared memory, and reduces straight to the final ``(n_pairs,)`` MI vector -- no global
accumulator, no row-chunking, no pair-subchunking.

Falls back cleanly (returns ``None`` from the gate function) whenever the shape doesn't fit even the
opt-in budget, or CUDA/CuPy/the opt-in probe are unavailable -- the existing ``batch_pair_mi_cuda`` /
``batch_pair_mi_cuda_row_chunked`` / CPU njit chain is untouched and remains the fallback of last resort.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

# GPU_INFRA_D-2 fix (mrmr_audit_2026-07-22): guards the max_dynamic_shared_size_bytes property-set +
# launch pair below so a concurrent call with a different shared_bytes requirement cannot race and
# under-provision another thread's launch (same pattern/fix as _gpu_pairs.py's _SHARED_MEM_SET_LOCK).
_SHARED_MEM_SET_LOCK = threading.Lock()

logger = logging.getLogger(__name__)

try:
    import cupy as _cp
    _CUPY_AVAIL = True
except Exception:
    _cp = None
    _CUPY_AVAIL = False

# Driver-reserved shared memory per block (observed ~1KB on cc 8.9 via ncu "Driver Shared Memory Per
# Block"; mirrors the headroom ``_hist_kernel_shared_fits_budget`` already uses for the row-chunked
# kernel's own shared-memory gate).
_DRIVER_SHARED_MEM_HEADROOM_BYTES = 2048

_SHARED_FUSED_KERNEL: Any = None  # lazy-compiled cp.RawKernel, cached process-wide


_PAIR_MI_SHARED_FUSED_SRC = r"""
extern "C" __global__
void pair_mi_shared_fused(
    const int* __restrict__ factors_data,   // (n_samples, n_features) row-major int32
    const long long* __restrict__ pair_a,   // (n_pairs,)
    const long long* __restrict__ pair_b,   // (n_pairs,)
    const int* __restrict__ nbins,          // (n_features,)
    const int* __restrict__ classes_y,      // (n_samples,)
    const double* __restrict__ freqs_y,     // (n_classes_y,)
    const long long n_samples, const int n_features, const int n_pairs,
    const int max_joint, const int n_classes_y, const double inv_n,
    double* __restrict__ mi_out             // (n_pairs,)
) {
    // Dynamic shared int32 histogram: [0:max_joint*n_classes_y) joint counts, then [.. +max_joint)
    // marginal (fx) counts -- same flat layout as the row-chunked kernel's shared-staged variant.
    extern __shared__ int sh[];
    __shared__ double mi_accum;
    int n_joint_cells = max_joint * n_classes_y;
    int total_cells = n_joint_cells + max_joint;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < total_cells; i += nt) sh[i] = 0;
    if (tid == 0) mi_accum = 0.0;
    __syncthreads();

    int p = blockIdx.x;
    if (p >= n_pairs) return;
    long long a = pair_a[p];
    long long b = pair_b[p];
    int nb_b = nbins[b];
    for (long long i = tid; i < n_samples; i += nt) {
        int va = factors_data[i * (long long)n_features + a];
        int vb = factors_data[i * (long long)n_features + b];
        int cls_x = va * nb_b + vb;
        int cls_y = classes_y[i];
        atomicAdd(&sh[cls_x * n_classes_y + cls_y], 1);
        atomicAdd(&sh[n_joint_cells + cls_x], 1);
    }
    __syncthreads();

    // PARALLEL reduction across all max_joint*n_classes_y joint cells (was a single-thread serial loop
    // over up to ~8800+ cells including a log() call per occupied cell -- the single largest cost in an
    // earlier version of this kernel, measured ~6x slower end-to-end than doing the histogram-accumulate
    // phase alone would suggest; see bench_batch_pair_mi_shared_fused.py). Every thread strides over a
    // disjoint subset of cells and atomically accumulates its partial sum into a shared double -- cells
    // are typically sparse (few threads hit a nonzero cell), so contention on ``mi_accum`` stays low.
    for (int idx = tid; idx < n_joint_cells; idx += nt) {
        int jc = sh[idx];
        if (jc == 0) continue;
        int i = idx / n_classes_y;
        int j = idx - i * n_classes_y;
        int fx = sh[n_joint_cells + i];
        if (fx == 0) continue;
        double prob_y = freqs_y[j];
        if (prob_y <= 0.0) continue;
        double prob_x = (double)fx * inv_n;
        double jf = (double)jc * inv_n;
        atomicAdd(&mi_accum, jf * log(jf / (prob_x * prob_y)));
    }
    __syncthreads();
    if (tid == 0) mi_out[p] = mi_accum;
}
"""


def _opt_in_shared_mem_budget() -> int:
    """The device's extended (opt-in) per-block shared-memory budget in bytes, or the conservative
    49152-byte static default if the probe is unavailable (matches ``_hist_kernel_shared_fits_budget``'s
    own fallback)."""
    try:
        from pyutilz.system.gpu_dispatch import gpu_capability_summary, get_shared_mem_budget_per_block

        summary = gpu_capability_summary(0)
        if summary is None:
            return 49152
        return int(get_shared_mem_budget_per_block(summary["cc_major"], summary["cc_minor"], allow_opt_in=True))
    except Exception as e:
        logger.debug("_opt_in_shared_mem_budget: probe failed (%s); using the 49152B static default", e)
        return 49152


def shared_fused_kernel_fits_budget(max_joint: int, n_classes_y: int) -> int:
    """Bytes needed for :data:`_PAIR_MI_SHARED_FUSED_SRC`'s dynamic shared buffer, or 0 if it would
    exceed the device's per-block shared-memory budget INCLUDING the opt-in extension (caller falls
    back to the row-chunked kernel in that case -- gating the fast path on its safe condition rather
    than risking a launch failure)."""
    needed = (max_joint * n_classes_y + max_joint) * 4  # int32
    budget = _opt_in_shared_mem_budget()
    return needed if needed <= budget - _DRIVER_SHARED_MEM_HEADROOM_BYTES else 0


def _get_shared_fused_kernel() -> Any:
    """Lazy-compile + cache the ``pair_mi_shared_fused`` RawKernel."""
    global _SHARED_FUSED_KERNEL
    if _SHARED_FUSED_KERNEL is None:
        _SHARED_FUSED_KERNEL = _cp.RawKernel(_PAIR_MI_SHARED_FUSED_SRC, "pair_mi_shared_fused")
    return _SHARED_FUSED_KERNEL


def batch_pair_mi_cuda_shared_fused(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    threads_per_block: int = 128,
) -> np.ndarray:
    """Single full-pass CUDA pair-MI kernel: one launch, one block per pair, dynamic (opt-in-extended)
    shared-memory histogram, no row/pair chunking. Requires the FULL ``factors_data`` to fit VRAM (the
    same precondition :func:`batch_pair_mi_cuda` uses) -- callers must check
    :func:`shared_fused_kernel_fits_budget` themselves before calling (this function does not re-probe
    VRAM fit; that's the caller's ``_gpu_upload_fits`` responsibility, matching the sibling backends'
    contract).

    Bit-identical to :func:`batch_pair_mi_cuda_row_chunked`/:func:`batch_pair_mi_njit_prange` (same
    ``sum jf*log(jf/(px*py))`` reduction over occupied cells, same iteration order i-then-j) up to ~1e-15
    ULP floating-point reduction-order noise from the atomic-add accumulation order across threads --
    verified in ``tests/feature_selection/gpu/test_batch_pair_mi_shared_fused.py``.
    """
    if not _CUPY_AVAIL:
        raise RuntimeError("cupy is not available on this host")

    n_pairs = int(pair_a.shape[0])
    if n_pairs == 0:
        return np.empty(0, dtype=np.float64)

    n_samples = int(factors_data.shape[0])
    n_features = int(factors_data.shape[1])
    n_classes_y = int(freqs_y.shape[0])
    if n_samples == 0:
        # GPU_INFRA_A-2 fix (mrmr_audit_2026-07-22): mirrors batch_pair_mi_cupy's explicit
        # `if n_samples == 0: return zeros` guard -- without it, `inv_n = 1.0 / float(n_samples)` below
        # raises ZeroDivisionError for an empty input.
        return np.zeros(n_pairs, dtype=np.float64)

    nbins_i = np.ascontiguousarray(nbins, dtype=np.int32)
    joint_cards = nbins_i[np.ascontiguousarray(pair_a, dtype=np.int64)].astype(np.int64) * nbins_i[np.ascontiguousarray(pair_b, dtype=np.int64)].astype(np.int64)
    if np.any(joint_cards < 1):
        bad = int(np.argmin(joint_cards))
        raise ValueError(f"degenerate pair ({int(pair_a[bad])}, {int(pair_b[bad])}): joint cardinality {int(joint_cards[bad])} < 1")
    max_joint = int(joint_cards.max())

    shared_bytes = shared_fused_kernel_fits_budget(max_joint, n_classes_y)
    if shared_bytes == 0:
        raise RuntimeError(f"batch_pair_mi_cuda_shared_fused: max_joint={max_joint} n_classes_y={n_classes_y} exceeds the opt-in shared-memory budget")

    kernel = _get_shared_fused_kernel()
    static_budget = 49152 - _DRIVER_SHARED_MEM_HEADROOM_BYTES

    d_factors = _cp.asarray(np.ascontiguousarray(factors_data, dtype=np.int32))
    d_pair_a = _cp.asarray(np.ascontiguousarray(pair_a, dtype=np.int64))
    d_pair_b = _cp.asarray(np.ascontiguousarray(pair_b, dtype=np.int64))
    d_nbins = _cp.asarray(nbins_i)
    d_classes_y = _cp.asarray(np.ascontiguousarray(classes_y, dtype=np.int32))
    d_freqs_y = _cp.asarray(np.ascontiguousarray(freqs_y, dtype=np.float64))
    d_mi_out = _cp.zeros(n_pairs, dtype=_cp.float64)

    inv_n = 1.0 / float(n_samples)
    # GPU_INFRA_D-2 fix (mrmr_audit_2026-07-22): the property set + launch must be atomic together, or a
    # concurrent call with a different shared_bytes requirement could race and under-provision this launch
    # (same shared singleton-kernel-attribute pattern as _gpu_pairs.py's identical fix; see its module note).
    with _SHARED_MEM_SET_LOCK:
        if shared_bytes > static_budget:
            # Only touch the opt-in property when actually needed -- CuPy raises if this is set below the
            # device's static default on some driver versions, so avoid the call on the common (small-shape)
            # path where the static budget already suffices.
            kernel.max_dynamic_shared_size_bytes = shared_bytes
        kernel(
            (n_pairs,), (threads_per_block,),
            (
                d_factors, d_pair_a, d_pair_b, d_nbins, d_classes_y, d_freqs_y,
                np.int64(n_samples), np.int32(n_features), np.int32(n_pairs),
                np.int32(max_joint), np.int32(n_classes_y), np.float64(inv_n),
                d_mi_out,
            ),
            shared_mem=shared_bytes,
        )
    return np.asarray(_cp.asnumpy(d_mi_out))


__all__ = ["batch_pair_mi_cuda_shared_fused", "shared_fused_kernel_fits_budget"]
