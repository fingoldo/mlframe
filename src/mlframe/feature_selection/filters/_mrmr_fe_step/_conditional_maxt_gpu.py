"""Device twin of ``_conditional_mi_columns``: the conditional MI ``CMI(code_j; y_b | z)`` of every candidate column ``j`` for the observed target and every permuted target ``b`` at once.

The host kernel counts ``(z, code, y)`` cells of one target at a time over all columns (about 100 ms for 600 columns of 250000 rows on 8 threads, 40 permutations after one another: 4 s). Here one
block owns one ``(target, column)`` pair, counts the cells in a shared-memory table with integer atomics and evaluates the same formula over the occupied cells, so the whole null is one launch.
The counts are integers (identical to the host's); the sums over cells are double, in a different order, so the values agree with the host's to rounding.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

#: Shared-memory table cells per block (``kz * widest code * ky`` ints, 4 bytes each) the kernel handles; larger tables take the host kernel.
MAX_CELLS = 11_000
_THREADS = 256

_SRC = r"""
extern "C" __global__  // dtype-ok: Intentional float64, out is always the float64 array the launcher allocates, the sums are double on purpose
void cond_mi_columns(const short* __restrict__ codes, const long long* __restrict__ widths, const int* __restrict__ ys, const int* __restrict__ z,
                     const int n, const int p, const int B, const int ky, const int kz, const int max_w, double* __restrict__ out) {
    extern __shared__ int cnt[];
    __shared__ double red[256];
    __shared__ int n_z[64];
    __shared__ int n_zy[64 * 16];
    const int pair = blockIdx.x;
    const int b = pair / p;
    const int j = pair - b * p;
    const int kw = (int)widths[j];
    const int cells = kz * kw * ky;
    const int tid = threadIdx.x;
    for (int u = tid; u < cells; u += blockDim.x) cnt[u] = 0;
    __syncthreads();
    const short* cj = codes + (long long)j * n;
    const int* yb = ys + (long long)b * n;
    for (int i = tid; i < n; i += blockDim.x) atomicAdd(&cnt[(z[i] * kw + (int)cj[i]) * ky + yb[i]], 1);
    __syncthreads();
    for (int u = tid; u < kz * ky; u += blockDim.x) n_zy[u] = 0;
    for (int u = tid; u < kz; u += blockDim.x) n_z[u] = 0;
    __syncthreads();
    for (int u = tid; u < cells; u += blockDim.x) {
        const int c = u % ky;
        const int a = u / (kw * ky);
        atomicAdd(&n_zy[a * ky + c], cnt[u]);
        atomicAdd(&n_z[a], cnt[u]);
    }
    __syncthreads();
    const double inv_n = 1.0 / (double)n;
    double acc = 0.0;
    for (int u = tid; u < kz * kw; u += blockDim.x) {
        const int a = u / kw;
        int n_zw = 0;
        for (int c = 0; c < ky; ++c) n_zw += cnt[u * ky + c];
        if (n_zw == 0) continue;
        for (int c = 0; c < ky; ++c) {
            const int v = cnt[u * ky + c];
            if (v > 0) acc += (double)v * inv_n * log((double)v * (double)n_z[a] / ((double)n_zw * (double)n_zy[a * ky + c]));
        }
    }
    red[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[(long long)b * p + j] = red[0];
}
"""
_KERNEL: Any = None
_FALLBACK_WARNED = False


def _kernel(cp):
    """The compiled kernel."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawModule(code=_SRC).get_function("cond_mi_columns")
    return _KERNEL


def conditional_mi_targets_device(codes: np.ndarray, widths: np.ndarray, targets: np.ndarray, z: np.ndarray, ky: int, kz: int) -> "np.ndarray | None":
    """``(B, p)`` conditional MI of every column of the ``(p, n)`` code matrix for each of the ``B`` target rows of ``targets`` given ``z``, or ``None`` when the device cannot take it
    (no device, a table beyond ``MAX_CELLS`` or 64 strata / 16 classes, an allocation failure): the caller then runs the host kernel."""
    p, n = int(codes.shape[0]), int(codes.shape[1])
    max_w = int(widths.max())
    if kz > 64 or ky > 16 or kz * max_w * ky > MAX_CELLS or int(codes.max()) > 32767:
        return None
    try:
        import cupy as cp

        B = int(targets.shape[0])
        codes_d = cp.asarray(np.ascontiguousarray(codes, dtype=np.int16))
        w_d = cp.asarray(np.ascontiguousarray(widths, dtype=np.int64))
        ys_d = cp.asarray(np.ascontiguousarray(targets, dtype=np.int32))
        z_d = cp.asarray(np.ascontiguousarray(z, dtype=np.int32))
        # Intentional float64: the kernel writes double sums
        out = cp.empty((B, p), dtype=cp.float64)
        _kernel(cp)((p * B,), (_THREADS,), (codes_d, w_d, ys_d, z_d, np.int32(n), np.int32(p), np.int32(B), np.int32(ky), np.int32(kz), np.int32(max_w), out), shared_mem=kz * max_w * ky * 4)  # launch-ok: the kernel's shared reduction array and tree reduction are sized for 256 threads
        return np.asarray(cp.asnumpy(out))
    except Exception as exc:  # no device / out of memory / compile failure: the host kernel answers
        global _FALLBACK_WARNED
        if not _FALLBACK_WARNED:
            _FALLBACK_WARNED = True
            logger.warning("device conditional MI unavailable (%r); using the host kernel (reported once per process)", exc)
        else:
            logger.debug("device conditional MI unavailable (%r); using the host kernel", exc)
        return None


def conditional_mi_null_device(codes: np.ndarray, widths: np.ndarray, y: np.ndarray, z: np.ndarray, ky: int, kz: int, n_permutations: int, seed: int) -> "np.ndarray | None":
    """``(n_permutations + 1, p)``: row 0 the conditional MI of every column for the observed target ``y``, the other rows for ``y`` shuffled within the strata of ``z``, all drawn and counted on the device.

    The host draws these targets with one ``rng.permutation`` per stratum and permutation (about 150 ms for 40 permutations of 100000 rows, five times the kernel); here one flat radix sort of
    ``permutation | stratum | random bits | y`` gives all of them (the within-stratum shuffle of the GPU conditional null). The stream differs from the host's, so the null agrees with it in distribution,
    not bit for bit. ``None`` when the device cannot take it, as in ``conditional_mi_targets_device``."""
    p, n = int(codes.shape[0]), int(codes.shape[1])
    if kz > 64 or ky > 16 or kz * int(widths.max()) * ky > MAX_CELLS or int(codes.max()) > 32767:
        return None
    try:
        import cupy as cp

        from .._fe_cmi_perm_null_gpu import _flat_within_stratum_shuffle

        z_d = cp.asarray(np.ascontiguousarray(z, dtype=np.int32))
        y_d = cp.asarray(np.ascontiguousarray(y, dtype=np.int32))
        order = cp.argsort(z_d)
        z_sorted = z_d[order]
        rank = cp.zeros(n, dtype=cp.int64)
        rank[1:] = cp.cumsum(z_sorted[1:] != z_sorted[:-1])
        flat = _flat_within_stratum_shuffle(cp, cp.random.RandomState(int(seed) & 0x7FFFFFFF), y_d[order], rank[:, None], int(n_permutations), int(ky), int(rank[-1]) + 1)
        if flat is None:
            return None
        targets = cp.empty((int(n_permutations) + 1, n), dtype=cp.int32)
        targets[0] = y_d
        targets[1:, order] = flat
        codes_d = cp.asarray(np.ascontiguousarray(codes, dtype=np.int16))
        w_d = cp.asarray(np.ascontiguousarray(widths, dtype=np.int64))
        # Intentional float64: the kernel writes double sums
        out = cp.empty((int(n_permutations) + 1, p), dtype=cp.float64)
        max_w = int(widths.max())
        _kernel(cp)((p * (int(n_permutations) + 1),), (_THREADS,), (codes_d, w_d, targets, z_d, np.int32(n), np.int32(p), np.int32(int(n_permutations) + 1), np.int32(ky), np.int32(kz), np.int32(max_w), out), shared_mem=kz * max_w * ky * 4)  # launch-ok: the kernel's shared reduction array and tree reduction are sized for 256 threads
        return np.asarray(cp.asnumpy(out))
    except Exception as exc:  # no device / out of memory / compile failure: the host draws and counts
        log_throttle(
            logger, "conditional_maxt_gpu_unavailable", logging.WARNING,
            "device conditional null unavailable (%s: %s); the host draws and counts", type(exc).__name__, exc,
        )
        return None
