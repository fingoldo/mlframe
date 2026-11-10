"""Pair-MI CuPy ``RawKernel`` over FEATURE-MAJOR byte codes: one block per pair, coalesced 4-byte loads, int32 shared histogram.

The row-major kernels (`batch_pair_mi_cuda`, `batch_pair_mi_cuda_shared_fused`) read `factors_data[i * n_features + a]`: the 32 lanes of a warp touch 32 different rows, 601 bytes apart at 600
columns, so every request fetches 32 sectors for 32 useful bytes. Measured on 179700 pairs of 100000 rows (GTX 1050 Ti): 105 microseconds per pair, 2.8 GB/s effective against ~112 GB/s of
the card. Here the codes are transposed once on the device to `(n_features, n_pad)` unsigned bytes (rows padded to a multiple of four so every column starts word-aligned), a thread reads four
consecutive rows of a column as one 32-bit word, and the histogram counts stay in a dynamic-shared int32 array exactly as in the shared-fused kernel. The finalising reduction is the
shared-fused kernel's, line for line, so the MI of a pair is the same double (`tests/feature_selection/gpu/test_batch_pair_mi_cuda_fm.py` checks equality with the row-major kernel).

Applies when every code and class fits a byte (`nbins <= 256`, `n_classes_y <= 256`); otherwise the caller keeps the row-major kernels. Counts are exact integers, so the result does not depend
on the order the threads add in.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from ._batch_pair_mi_cuda_shared_fused import shared_fused_kernel_fits_budget, validate_pair_codes
from ._kernel_types import render

logger = logging.getLogger(__name__)

try:
    import cupy as _cp

    _CUPY_AVAIL = True
except ImportError:
    _cp = None
    _CUPY_AVAIL = False

_FM_KERNEL: Any = None
_LAUNCH_LOCK = threading.Lock()
_DRIVER_SHARED_MEM_HEADROOM_BYTES = 2048

_PAIR_MI_FM_SRC = render(r"""
extern "C" __global__
void pair_mi_feature_major(
    const unsigned char* __restrict__ cm,       // (n_features, n_pad) feature-major byte codes
    const unsigned char* __restrict__ ycodes,   // (n_pad,) byte class codes
    const long long* __restrict__ pair_a,
    const long long* __restrict__ pair_b,
    const int* __restrict__ nbins,
    const {ACC}* __restrict__ freqs_y,
    const long long n_samples, const long long n_pad, const int n_pairs,
    const int max_joint, const int n_classes_y, const {ACC} inv_n,
    {ACC}* __restrict__ mi_out
) {
    extern __shared__ int sh[];
    __shared__ {ACC} warp_sums[32];
    int n_joint_cells = max_joint * n_classes_y;
    int total_cells = n_joint_cells + max_joint;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < total_cells; i += nt) sh[i] = 0;
    __syncthreads();

    int p = blockIdx.x;
    if (p >= n_pairs) return;
    long long a = pair_a[p];
    long long b = pair_b[p];
    int nb_b = nbins[b];
    const unsigned int* wa = (const unsigned int*)(cm + a * n_pad);
    const unsigned int* wb = (const unsigned int*)(cm + b * n_pad);
    const unsigned int* wy = (const unsigned int*)ycodes;
    long long n_words = n_samples >> 2;
    for (long long w = tid; w < n_words; w += nt) {
        unsigned int xa = wa[w], xb = wb[w], xy = wy[w];
        #pragma unroll
        for (int s = 0; s < 32; s += 8) {
            int cls_x = (int)((xa >> s) & 255u) * nb_b + (int)((xb >> s) & 255u);
            int cls_y = (int)((xy >> s) & 255u);
            atomicAdd(&sh[cls_x * n_classes_y + cls_y], 1);
            atomicAdd(&sh[n_joint_cells + cls_x], 1);
        }
    }
    for (long long i = (n_words << 2) + tid; i < n_samples; i += nt) {
        int cls_x = (int)cm[a * n_pad + i] * nb_b + (int)cm[b * n_pad + i];
        int cls_y = (int)ycodes[i];
        atomicAdd(&sh[cls_x * n_classes_y + cls_y], 1);
        atomicAdd(&sh[n_joint_cells + cls_x], 1);
    }
    __syncthreads();

    {ACC} loc = 0.0;
    for (int idx = tid; idx < n_joint_cells; idx += nt) {
        int jc = sh[idx];
        if (jc == 0) continue;
        int i = idx / n_classes_y;
        int j = idx - i * n_classes_y;
        int fx = sh[n_joint_cells + i];
        if (fx == 0) continue;
        {ACC} prob_y = freqs_y[j];
        if (prob_y <= 0.0) continue;
        {ACC} prob_x = ({ACC})fx * inv_n;
        {ACC} jf = ({ACC})jc * inv_n;
        loc += jf * log(jf / (prob_x * prob_y));
    }
    for (int off = 16; off > 0; off >>= 1) loc += __shfl_down_sync(0xffffffffu, loc, off);
    if ((tid & 31) == 0) warp_sums[tid >> 5] = loc;
    __syncthreads();
    if (tid == 0) {
        {ACC} total = 0.0;
        for (int w = 0; w < (nt >> 5); ++w) total += warp_sums[w];
        mi_out[p] = total;
    }
}
""")


def codes_fit_bytes(factors_data: np.ndarray, classes_y: np.ndarray, nbins: np.ndarray, freqs_y: np.ndarray) -> bool:
    """Whether every code, class and cardinality fits the byte layout (the kernel's applicability test)."""
    if int(freqs_y.shape[0]) > 256 or (nbins.size and int(np.max(nbins)) > 256):
        return False
    if factors_data.size and (int(factors_data.min()) < 0 or int(factors_data.max()) > 255):
        return False
    return not classes_y.size or (int(classes_y.min()) >= 0 and int(classes_y.max()) <= 255)


def _get_kernel() -> Any:
    """Lazy-compile and cache the RawKernel."""
    global _FM_KERNEL
    if _FM_KERNEL is None:
        _FM_KERNEL = _cp.RawKernel(_PAIR_MI_FM_SRC, "pair_mi_feature_major")
    return _FM_KERNEL


def batch_pair_mi_cuda_feature_major(
    factors_data: np.ndarray, pair_a: np.ndarray, pair_b: np.ndarray, nbins: np.ndarray, classes_y: np.ndarray, freqs_y: np.ndarray, threads_per_block: int = 256,
) -> np.ndarray:
    """MI per pair (nats) from byte codes in feature-major layout; raises when the shape does not fit the kernel (byte codes, shared-memory budget), so the caller falls back to the row-major kernels."""
    if not _CUPY_AVAIL:
        raise RuntimeError("cupy is not available on this host")
    n_pairs = int(pair_a.shape[0])
    if n_pairs == 0:
        return np.empty(0, dtype=np.float64)  # Intentional float64: MI statistic output compared at full precision
    n_samples = int(factors_data.shape[0])
    if n_samples == 0:
        return np.zeros(n_pairs, dtype=np.float64)  # Intentional float64: MI statistic output compared at full precision
    n_classes_y = int(freqs_y.shape[0])
    nbins_i = np.ascontiguousarray(nbins, dtype=np.int32)
    pa = np.ascontiguousarray(pair_a, dtype=np.int64)
    pb = np.ascontiguousarray(pair_b, dtype=np.int64)
    joint = nbins_i[pa].astype(np.int64) * nbins_i[pb].astype(np.int64)
    if np.any(joint < 1):
        raise ValueError("degenerate pair: joint cardinality < 1")
    max_joint = int(joint.max())
    if not codes_fit_bytes(factors_data, classes_y, nbins_i, freqs_y):
        raise ValueError("codes do not fit the byte layout")
    validate_pair_codes(factors_data, pa, pb, nbins_i, classes_y, n_classes_y)
    shared_bytes = shared_fused_kernel_fits_budget(max_joint, n_classes_y)
    if shared_bytes == 0:
        raise RuntimeError("batch_pair_mi_cuda_feature_major: histogram exceeds the shared-memory budget")

    n_pad = (n_samples + 3) & ~3
    n_features = int(factors_data.shape[1])
    # transpose on the device: upload the row-major bytes once, then one (features, rows) copy with the rows padded to whole words
    d_rm = _cp.asarray(np.ascontiguousarray(factors_data, dtype=np.uint8))
    d_cm = _cp.zeros((n_features, n_pad), dtype=_cp.uint8)
    d_cm[:, :n_samples] = d_rm.T
    del d_rm
    d_y = _cp.zeros(n_pad, dtype=_cp.uint8)
    d_y[:n_samples] = _cp.asarray(np.ascontiguousarray(classes_y, dtype=np.uint8))
    d_pa, d_pb = _cp.asarray(pa), _cp.asarray(pb)
    d_nbins = _cp.asarray(nbins_i)
    d_freqs = _cp.asarray(np.ascontiguousarray(freqs_y, dtype=np.float64))  # Intentional float64: the kernel signature declares double
    d_out = _cp.zeros(n_pairs, dtype=_cp.float64)  # Intentional float64: the kernel signature declares double
    kernel = _get_kernel()
    static_budget = 49152 - _DRIVER_SHARED_MEM_HEADROOM_BYTES
    with _LAUNCH_LOCK:
        if shared_bytes > static_budget:
            kernel.max_dynamic_shared_size_bytes = shared_bytes
        kernel(
            (n_pairs,), (int(threads_per_block),),
            (d_cm, d_y, d_pa, d_pb, d_nbins, d_freqs, np.int64(n_samples), np.int64(n_pad), np.int32(n_pairs), np.int32(max_joint), np.int32(n_classes_y), np.float64(1.0 / float(n_samples)), d_out),  # Intentional float64: kernel ABI declares double inv_n
            shared_mem=shared_bytes,
        )
    return np.asarray(_cp.asnumpy(d_out))


__all__ = ["batch_pair_mi_cuda_feature_major", "codes_fit_bytes"]
