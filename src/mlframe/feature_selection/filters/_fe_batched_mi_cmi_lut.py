"""Lookup-table row entropy of the device CMI count matrices, carved out of ``_fe_batched_mi_cmi`` to keep that module under its size budget.

``_fe_batched_mi_cmi`` re-exports every name here, so existing imports keep working.
"""

from __future__ import annotations

import os
import threading

import numpy as np
from pyutilz.performance.gpu_hw_launch import block_size_for

from ._kernel_types import render

# LOOKUP-TABLE variant (profiled on a GTX 1050 Ti, 1500 candidates x 200000 / 400000 int32 cells per row, n = 20000): rows_ent_nnz_i32 was 66 percent of the device
# CMI batch (nvprof: occupancy 0.99, SM efficiency 98 percent, DRAM utilisation low, IPC 0.31, 55 to 68 percent of the stalls on execution dependency). The counts are
# sparse (at most n of the M cells are occupied) but a warp always holds an occupied lane, so every warp ran the double precision log (1/32 rate on Pascal) for all cells.
# Counts are integers <= n, so c*ln(c) comes from a table built once per n, and with p = c/n the row entropy is (ln(n) * sum(c) - sum(c ln c)) / n. It is the same entropy
# up to rounding (max difference 8.9e-15, occupied-cell counts identical); 19x (M = 200000) and 3x (M = 400000) faster on this card. rows_ent_nnz_i32 stays as the fallback.
_ROWS_ENT_NNZ_LUT_SRC = render(r"""
extern "C" __global__
void rows_ent_nnz_i32_lut4(const int* __restrict__ counts, const {ACC}* __restrict__ xlogx, const int nmax,
                           const {ACC} inv_n, const {ACC} log_n, const int K, const long long M,
                           {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    int row = blockIdx.x;
    if (row >= K) return;
    const int4* cr4 = reinterpret_cast<const int4*>(counts + (long long)row * M);
    const long long M4 = M >> 2;
    int tid = threadIdx.x;
    {ACC} sloc = 0.0; long long cloc = 0; int kloc = 0;
    for (long long i = tid; i < M4; i += blockDim.x) {
        int4 v = cr4[i];
        if ((v.x | v.y | v.z | v.w) != 0) {
            if (v.x > 0) { sloc += (v.x <= nmax) ? xlogx[v.x] : ({ACC})v.x * log(({ACC})v.x); cloc += v.x; kloc++; }
            if (v.y > 0) { sloc += (v.y <= nmax) ? xlogx[v.y] : ({ACC})v.y * log(({ACC})v.y); cloc += v.y; kloc++; }
            if (v.z > 0) { sloc += (v.z <= nmax) ? xlogx[v.z] : ({ACC})v.z * log(({ACC})v.z); cloc += v.z; kloc++; }
            if (v.w > 0) { sloc += (v.w <= nmax) ? xlogx[v.w] : ({ACC})v.w * log(({ACC})v.w); cloc += v.w; kloc++; }
        }
    }
    __shared__ {ACC} sh_s[256];
    __shared__ long long sh_c[256];
    __shared__ int sh_k[256];
    sh_s[tid] = sloc; sh_c[tid] = cloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_s[tid] += sh_s[tid + s]; sh_c[tid] += sh_c[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out_h[row] = (log_n * ({ACC})sh_c[0] - sh_s[0]) * inv_n; out_k[row] = (long long)sh_k[0]; }
}
extern "C" __global__
void rows_ent_nnz_i32_lut1(const int* __restrict__ counts, const {ACC}* __restrict__ xlogx, const int nmax,
                           const {ACC} inv_n, const {ACC} log_n, const int K, const long long M,
                           {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    int row = blockIdx.x;
    if (row >= K) return;
    const int* cr = counts + (long long)row * M;
    int tid = threadIdx.x;
    {ACC} sloc = 0.0; long long cloc = 0; int kloc = 0;
    for (long long i = tid; i < M; i += blockDim.x) {
        int c = cr[i];
        if (c > 0) { sloc += (c <= nmax) ? xlogx[c] : ({ACC})c * log(({ACC})c); cloc += c; kloc++; }
    }
    __shared__ {ACC} sh_s[256];
    __shared__ long long sh_c[256];
    __shared__ int sh_k[256];
    sh_s[tid] = sloc; sh_c[tid] = cloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_s[tid] += sh_s[tid + s]; sh_c[tid] += sh_c[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out_h[row] = (log_n * ({ACC})sh_c[0] - sh_s[0]) * inv_n; out_k[row] = (long long)sh_k[0]; }
}
// SPLIT-K lookup-table variant: the narrow-chunk, huge-row case (K < 48 rows of 2^20 or more cells) runs one block per (row, segment) so the SMs are not left idle (the split kernel
// of _fe_batched_mi_cmi), and each cell takes c * ln(c) from the table instead of a 64-bit log (1/32 rate on Pascal). Each block writes its partial (sum of c ln c, sum of c,
// occupied cells) to a (K, S) scratch; the finish kernel folds the S partials of a row in segment order and applies the same closing formula as the one-block kernels.
extern "C" __global__
void rows_ent_nnz_i32_lut_split(const int* __restrict__ counts, const {ACC}* __restrict__ xlogx, const int nmax,
                                const int K, const long long M, const int S,
                                {ACC}* __restrict__ part_s, {ACC}* __restrict__ part_c, {ACC}* __restrict__ part_k) {
    int row = blockIdx.x;
    int seg = blockIdx.y;
    if (row >= K || seg >= S) return;
    long long seg_len = (M + S - 1) / S;
    long long lo = (long long)seg * seg_len;
    long long hi = lo + seg_len; if (hi > M) hi = M;
    const int* cr = counts + (long long)row * M;
    int tid = threadIdx.x;
    {ACC} sloc = 0.0; long long cloc = 0; int kloc = 0;
    for (long long i = lo + tid; i < hi; i += blockDim.x) {
        int c = cr[i];
        if (c > 0) { sloc += (c <= nmax) ? xlogx[c] : ({ACC})c * log(({ACC})c); cloc += c; kloc++; }
    }
    __shared__ {ACC} sh_s[256];
    __shared__ long long sh_c[256];
    __shared__ int sh_k[256];
    sh_s[tid] = sloc; sh_c[tid] = cloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_s[tid] += sh_s[tid + s]; sh_c[tid] += sh_c[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { part_s[row * S + seg] = sh_s[0]; part_c[row * S + seg] = ({ACC})sh_c[0]; part_k[row * S + seg] = ({ACC})sh_k[0]; }
}
// coalescing-ok: a finishing pass: one thread per row sums its S block partials (K * S values, read once) against the K * M cells the counting pass read.
extern "C" __global__
void rows_ent_nnz_lut_finish(const {ACC}* __restrict__ part_s, const {ACC}* __restrict__ part_c, const {ACC}* __restrict__ part_k,
                             const int K, const int S, const {ACC} inv_n, const {ACC} log_n,
                             {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    int row = (int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= K) return;
    {ACC} s = 0.0, c = 0.0, k = 0.0;
    for (int q = 0; q < S; ++q) { s += part_s[row * S + q]; c += part_c[row * S + q]; k += part_k[row * S + q]; }
    out_h[row] = (log_n * c - s) * inv_n; out_k[row] = (long long)(k + 0.5);
}
""")
_ROWS_ENT_NNZ_LUT_KERNELS = None
_ROWS_ENT_NNZ_LUT_SPLIT_KERNELS = None
_XLOGX_TABLES: dict = {}
_XLOGX_TABLES_LOCK = threading.Lock()
_XLOGX_TABLES_MAX = 4


def _get_rows_ent_nnz_lut_kernels(cp):
    """Lazily compile and cache the vectorised (``lut4``, M % 4 == 0) and scalar (``lut1``) lookup-table entropy kernels."""
    global _ROWS_ENT_NNZ_LUT_KERNELS
    if _ROWS_ENT_NNZ_LUT_KERNELS is None:
        mod = cp.RawModule(code=_ROWS_ENT_NNZ_LUT_SRC)
        _ROWS_ENT_NNZ_LUT_KERNELS = (mod.get_function("rows_ent_nnz_i32_lut4"), mod.get_function("rows_ent_nnz_i32_lut1"))
    return _ROWS_ENT_NNZ_LUT_KERNELS


def _get_rows_ent_nnz_lut_split_kernels(cp):
    """Lazily compile and cache the split-K lookup-table pair: the per-(row, segment) partial kernel and the finishing fold."""
    global _ROWS_ENT_NNZ_LUT_SPLIT_KERNELS
    if _ROWS_ENT_NNZ_LUT_SPLIT_KERNELS is None:
        mod = cp.RawModule(code=_ROWS_ENT_NNZ_LUT_SRC)
        _ROWS_ENT_NNZ_LUT_SPLIT_KERNELS = (mod.get_function("rows_ent_nnz_i32_lut_split"), mod.get_function("rows_ent_nnz_lut_finish"))
    return _ROWS_ENT_NNZ_LUT_SPLIT_KERNELS


def _rows_entropy_lut_split(cp, c, inv_n: float, segments: int):
    """Split-K row entropy + occupied-cell count through the lookup table, or ``None`` when the row count is not a whole number or the table switch is off (the caller then runs the
    double precision split kernel). ``segments`` blocks share each row."""
    if os.environ.get("MLFRAME_ENT_LUT", "1") == "0" or inv_n <= 0.0:
        return None
    n_f = 1.0 / float(inv_n)
    n_int = round(n_f)
    if n_int < 1 or abs(n_f - n_int) > 1e-9 * n_int:
        return None
    K = int(c.shape[0])
    M = int(c.shape[1]) if c.ndim > 1 else int(c.size)
    split, finish = _get_rows_ent_nnz_lut_split_kernels(cp)
    tab = _xlogx_table(cp, n_int)
    S = int(segments)
    part_s = cp.empty(K * S, dtype=cp.float64)  # Intentional float64: partial sums of the entropy table, as the one-block kernels hold them
    part_c = cp.empty(K * S, dtype=cp.float64)  # Intentional float64: partial count sums (exact up to 2^53)
    part_k = cp.empty(K * S, dtype=cp.float64)  # Intentional float64: partial occupied-cell counts (exact up to 2^53)
    out_h = cp.empty(K, dtype=cp.float64)  # Intentional float64: entropy output contract
    out_k = cp.empty(K, dtype=cp.int64)
    threads = block_size_for(split, work_items=M, pow2=True, max_width=256)
    split((K, S), (threads,), (c, tab, np.int32(n_int), np.int32(K), np.int64(M), np.int32(S), part_s, part_c, part_k))
    finish_threads = block_size_for(finish, work_items=K)
    finish(((K + finish_threads - 1) // finish_threads,), (finish_threads,), (part_s, part_c, part_k, np.int32(K), np.int32(S), float(inv_n), float(np.log(n_int)), out_h, out_k))
    return out_h, out_k


def _xlogx_table(cp, nmax: int):
    """Device table ``t[c] = c * ln(c)`` for ``c = 0 .. nmax`` (``t[0] = 0``), built on the host in double precision and kept per ``nmax`` (a few distinct row counts per fit)."""
    with _XLOGX_TABLES_LOCK:
        tab = _XLOGX_TABLES.get(int(nmax))
        if tab is None:
            # Intentional float64: the entropy table is built in double precision, as the kernel reads it
            c = np.arange(1, int(nmax) + 1, dtype=np.float64)
            host = np.zeros(int(nmax) + 1, dtype=np.float64)  # Intentional float64: entropy table kept at double precision
            host[1:] = c * np.log(c)
            if len(_XLOGX_TABLES) >= _XLOGX_TABLES_MAX:
                _XLOGX_TABLES.pop(next(iter(_XLOGX_TABLES)))  # evict-ok: a miss rebuilds the table
            tab = _XLOGX_TABLES[int(nmax)] = cp.asarray(host)
    return tab


def _rows_entropy_lut(cp, c, inv_n: float):
    """Row entropy + occupied-cell count of the int32 count matrix ``c`` through the lookup-table kernels, or ``None`` when the row count is not a whole number or the table switch is off (the caller then runs the log kernel)."""
    if os.environ.get("MLFRAME_ENT_LUT", "1") == "0" or inv_n <= 0.0:
        return None
    n_f = 1.0 / float(inv_n)
    n_int = round(n_f)
    if n_int < 1 or abs(n_f - n_int) > 1e-9 * n_int:
        return None
    K = int(c.shape[0])
    M = int(c.shape[1]) if c.ndim > 1 else int(c.size)
    lut4, lut1 = _get_rows_ent_nnz_lut_kernels(cp)
    tab = _xlogx_table(cp, n_int)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    out_h = cp.empty(K, dtype=cp.float64)
    out_k = cp.empty(K, dtype=cp.int64)
    ker = lut4 if (M % 4 == 0) else lut1
    # the block reduction halves blockDim.x over 256-wide shared scratch: a power of two, at most 256
    threads = block_size_for(ker, work_items=M, pow2=True, max_width=256)
    ker((K,), (threads,), (c, tab, np.int32(n_int), float(inv_n), float(np.log(n_int)), np.int32(K), np.int64(M), out_h, out_k))
    return out_h, out_k


__all__ = [
    "_ROWS_ENT_NNZ_LUT_SRC",
    "_XLOGX_TABLES",
    "_get_rows_ent_nnz_lut_kernels",
    "_get_rows_ent_nnz_lut_split_kernels",
    "_rows_entropy_lut",
    "_rows_entropy_lut_split",
    "_xlogx_table",
]
