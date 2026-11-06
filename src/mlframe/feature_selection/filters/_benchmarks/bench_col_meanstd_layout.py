"""Row-major versus column-major read for the col_meanstd reduction (sum pass only, same access pattern)."""

import time

import cupy as cp
import numpy as np

ROW = r"""
extern "C" __global__
void col_sum_row(const double* __restrict__ M, const long long n, const int K, double* out) {
    int col = blockIdx.x; if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    double s = 0.0;
    for (long long i = tid; i < n; i += nt) s += M[i * (long long)K + col];
    __shared__ double sh[256];
    sh[tid] = s; __syncthreads();
    for (int j = nt >> 1; j > 0; j >>= 1) { if (tid < j) sh[tid] += sh[tid + j]; __syncthreads(); }
    if (tid == 0) out[col] = sh[0];
}"""
COL = r"""
extern "C" __global__
void col_sum_col(const double* __restrict__ MT, const long long n, const int K, double* out) {
    int col = blockIdx.x; if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    const double* c = MT + (long long)col * n;
    double s = 0.0;
    for (long long i = tid; i < n; i += nt) s += c[i];
    __shared__ double sh[256];
    sh[tid] = s; __syncthreads();
    for (int j = nt >> 1; j > 0; j >>= 1) { if (tid < j) sh[tid] += sh[tid + j]; __syncthreads(); }
    if (tid == 0) out[col] = sh[0];
}"""
mod = cp.RawModule(code=ROW + COL)
kr = mod.get_function("col_sum_row")
kc = mod.get_function("col_sum_col")
K = 16


def timed(fn, reps=7):
    fn()
    cp.cuda.Device().synchronize()
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        best = min(best, time.perf_counter() - s)
    return best


for n in (500, 10_000, 100_000, 1_000_000):
    M = cp.random.random((n, K))
    MT = cp.ascontiguousarray(M.T)
    out = cp.empty(K)
    t_row = timed(lambda: kr((K,), (256,), (M, np.int64(n), np.int32(K), out)))
    t_col = timed(lambda: kc((K,), (256,), (MT, np.int64(n), np.int32(K), out)))
    print(f"n={n:>9} row={t_row * 1e3:9.3f}ms  col(kernel only)={t_col * 1e3:9.3f}ms")
