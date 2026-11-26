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
# Version 2 (``_SRC_U8``): the same counts and the same epilogue, read from bytes. Version 1 reads a 2-byte code, a 4-byte stratum and a 4-byte target for each row of
# each (target, column) block, and nvprof put it at 72 percent memory-dependency stalls (209.7 ms for 300 columns x 41 targets x 300000 rows, DRAM 47 GB/s, double-precision
# efficiency 1.8 percent: the logarithms are not the cost). The code, the stratum (at most 64) and the class (at most 16) each fit a byte, so a row is 3 bytes, and a thread
# reads 4 rows with one 4-byte load of each array (rows padded to a multiple of 4). Counts are integers and the epilogue is unchanged, so the values are bit-identical:
# 213.6 -> 78.7 ms at that shape (2.7 x), 30.4 -> 15.8 ms at 100003 rows x 120 columns x 33 targets with 64 strata. Win axis: every shape whose codes fit a byte; version 1
# stays for wider codes.
_SRC_U8 = r"""
extern "C" __global__  // dtype-ok: Intentional float64, out is the float64 array the launcher allocates, the sums are double as in _SRC
void cond_mi_columns_u8(const unsigned char* __restrict__ codes, const long long* __restrict__ widths, const unsigned char* __restrict__ ys, const unsigned char* __restrict__ z,
                        const int n, const int stride, const int p, const int B, const int ky, const int kz, const int max_w, double* __restrict__ out) {
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
    const unsigned char* cj = codes + (long long)j * stride;
    const unsigned char* yb = ys + (long long)b * stride;
    const unsigned int* cj4 = (const unsigned int*)cj;
    const unsigned int* yb4 = (const unsigned int*)yb;
    const unsigned int* z4 = (const unsigned int*)z;
    const int n4 = n >> 2;
    for (int g = tid; g < n4; g += blockDim.x) {
        const unsigned int wc = cj4[g], wy = yb4[g], wz = z4[g];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int zz = (int)((wz >> (8 * u)) & 0xffu), cc = (int)((wc >> (8 * u)) & 0xffu), yy = (int)((wy >> (8 * u)) & 0xffu);
            atomicAdd(&cnt[(zz * kw + cc) * ky + yy], 1);
        }
    }
    for (int i = (n4 << 2) + tid; i < n; i += blockDim.x) atomicAdd(&cnt[((int)z[i] * kw + (int)cj[i]) * ky + (int)yb[i]], 1);
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
_KERNEL_U8: Any = None
#: Rows of the byte arrays are padded to this many bytes (the 4-row loads).
_ROW_ALIGN = 4


def _kernel_u8(cp):
    """The compiled byte kernel."""
    global _KERNEL_U8
    if _KERNEL_U8 is None:
        _KERNEL_U8 = cp.RawModule(code=_SRC_U8).get_function("cond_mi_columns_u8")
    return _KERNEL_U8


def _padded_bytes(cp, a2d, stride):
    """``a2d`` (host or device, integer values below 256) as a device uint8 matrix whose rows are ``stride`` bytes (the pad is zero and never counted)."""
    out = cp.zeros((int(a2d.shape[0]), int(stride)), dtype=cp.uint8)
    out[:, : int(a2d.shape[1])] = cp.asarray(a2d).astype(cp.uint8, copy=False)
    return out


def _launch(cp: Any, codes: np.ndarray, widths: np.ndarray, targets_d: Any, z: Any, n: int, p: int, B: int, ky: int, kz: int, max_w: int) -> np.ndarray:
    """One launch over every (target, column) pair: the byte kernel when the codes fit a byte, version 1 otherwise; ``(B, p)`` double sums on the host."""
    w_d = cp.asarray(np.ascontiguousarray(widths, dtype=np.int64))
    # Intentional float64: the kernel writes double sums
    out = cp.empty((B, p), dtype=cp.float64)
    shared = kz * max_w * ky * 4
    if max_w <= 256 and int(codes.max()) <= 255:
        stride = -(-n // _ROW_ALIGN) * _ROW_ALIGN
        codes_d = _padded_bytes(cp, np.ascontiguousarray(codes, dtype=np.uint8), stride)
        ys_d = _padded_bytes(cp, targets_d, stride)
        z_d = _padded_bytes(cp, cp.asarray(z).reshape(1, -1), stride)[0]
        _kernel_u8(cp)((p * B,), (_THREADS,), (codes_d, w_d, ys_d, z_d, np.int32(n), np.int32(stride), np.int32(p), np.int32(B), np.int32(ky), np.int32(kz), np.int32(max_w), out), shared_mem=shared)  # launch-ok: 256 threads, the size of the static reduction array of the kernel
    else:
        codes_d = cp.asarray(np.ascontiguousarray(codes, dtype=np.int16))
        _kernel(cp)((p * B,), (_THREADS,), (codes_d, w_d, cp.asarray(targets_d, dtype=cp.int32), cp.asarray(z, dtype=cp.int32), np.int32(n), np.int32(p), np.int32(B), np.int32(ky), np.int32(kz), np.int32(max_w), out), shared_mem=shared)  # launch-ok: 256 threads, the size of the static reduction array of the kernel
    return np.asarray(cp.asnumpy(out))


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
        return _launch(cp, codes, widths, cp.asarray(np.ascontiguousarray(targets, dtype=np.int32)), np.ascontiguousarray(z, dtype=np.int32), n, p, B, ky, kz, max_w)
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
        return _launch(cp, codes, widths, targets, z_d, n, p, int(n_permutations) + 1, ky, kz, int(widths.max()))
    except Exception as exc:  # no device / out of memory / compile failure: the host draws and counts
        log_throttle(
            logger, "conditional_maxt_gpu_unavailable", logging.WARNING,
            "device conditional null unavailable (%s: %s); the host draws and counts", type(exc).__name__, exc,
        )
        return None
