"""Row-major versus column-major read for the heavy_tail_stats reduction (same thread-strided pattern, per-column median/scale)."""

import time

import cupy as cp
import numpy as np

ROW = r"""
extern "C" __global__
void tail_row(const double* __restrict__ M, const double* __restrict__ med, const double* __restrict__ scale,
              const double outerK, const long long n, const int K, double* out) {
    int col = blockIdx.x; if (col >= K) return;
    double mc = med[col], thr = outerK * scale[col];
    int tid = threadIdx.x;
    double bulk = -1e308;
    for (long long i = tid; i < n; i += blockDim.x) {
        double dev = fabs(M[i * (long long)K + col] - mc);
        if (dev <= thr && dev > bulk) bulk = dev;
    }
    __shared__ double sh[256];
    sh[tid] = bulk; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s && sh[tid + s] > sh[tid]) sh[tid] = sh[tid + s]; __syncthreads(); }
    if (tid == 0) out[col] = sh[0];
}"""
COL = r"""
extern "C" __global__
void tail_col(const double* __restrict__ MT, const double* __restrict__ med, const double* __restrict__ scale,
              const double outerK, const long long n, const int K, double* out) {
    int col = blockIdx.x; if (col >= K) return;
    double mc = med[col], thr = outerK * scale[col];
    int tid = threadIdx.x;
    const double* c = MT + (long long)col * n;
    double bulk = -1e308;
    for (long long i = tid; i < n; i += blockDim.x) {
        double dev = fabs(c[i] - mc);
        if (dev <= thr && dev > bulk) bulk = dev;
    }
    __shared__ double sh[256];
    sh[tid] = bulk; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s && sh[tid + s] > sh[tid]) sh[tid] = sh[tid + s]; __syncthreads(); }
    if (tid == 0) out[col] = sh[0];
}"""
mod = cp.RawModule(code=ROW + COL)
kr = mod.get_function("tail_row")
kc = mod.get_function("tail_col")
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
    med = cp.full(K, 0.5)
    scale = cp.full(K, 0.2)
    out = cp.empty(K)
    t_row = timed(lambda: kr((K,), (256,), (M, med, scale, np.float64(3.0), np.int64(n), np.int32(K), out)))
    t_col = timed(lambda: kc((K,), (256,), (MT, med, scale, np.float64(3.0), np.int64(n), np.int32(K), out)))
    print(f"n={n:>9} row={t_row * 1e3:9.3f}ms  col(kernel only)={t_col * 1e3:9.3f}ms")
