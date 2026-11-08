import time
import numpy as np
import cupy as cp

ROW = r'''
extern "C" __global__  // dtype-ok: benchmark: every buffer is allocated float64 in this file
void col_minmax_row(const double* __restrict__ M, const long long n, const int K, double* out_min, double* out_max) {
    int col = blockIdx.x; if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    double lmin = 1e308, lmax = -1e308;
    for (long long i = tid; i < n; i += nt) { double v = M[i * (long long)K + col]; if (v < lmin) lmin = v; if (v > lmax) lmax = v; }
    __shared__ double sh_min[256]; __shared__ double sh_max[256];
    sh_min[tid] = lmin; sh_max[tid] = lmax; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { if (sh_min[tid+s] < sh_min[tid]) sh_min[tid] = sh_min[tid+s]; if (sh_max[tid+s] > sh_max[tid]) sh_max[tid] = sh_max[tid+s]; } __syncthreads(); }
    if (tid == 0) { out_min[col] = sh_min[0]; out_max[col] = sh_max[0]; }
}'''
COL = r'''
extern "C" __global__  // dtype-ok: benchmark: every buffer is allocated float64 in this file
void col_minmax_col(const double* __restrict__ MT, const long long n, const int K, double* out_min, double* out_max) {
    int col = blockIdx.x; if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    double lmin = 1e308, lmax = -1e308;
    const double* c = MT + (long long)col * n;
    for (long long i = tid; i < n; i += nt) { double v = c[i]; if (v < lmin) lmin = v; if (v > lmax) lmax = v; }
    __shared__ double sh_min[256]; __shared__ double sh_max[256];
    sh_min[tid] = lmin; sh_max[tid] = lmax; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { if (sh_min[tid+s] < sh_min[tid]) sh_min[tid] = sh_min[tid+s]; if (sh_max[tid+s] > sh_max[tid]) sh_max[tid] = sh_max[tid+s]; } __syncthreads(); }
    if (tid == 0) { out_min[col] = sh_min[0]; out_max[col] = sh_max[0]; }
}'''
mod = cp.RawModule(code=ROW + COL)  # dtype-ok: benchmark: every buffer is allocated float64 in this file
kr = mod.get_function("col_minmax_row")  # dtype-ok: benchmark: every buffer is allocated float64 in this file
kc = mod.get_function("col_minmax_col")  # dtype-ok: benchmark: every buffer is allocated float64 in this file
K = 16

def timed(fn, reps=7):
    fn(); cp.cuda.Device().synchronize()
    ts = []
    for _ in range(reps):
        s = time.perf_counter(); fn(); cp.cuda.Device().synchronize(); ts.append(time.perf_counter() - s)
    return min(ts)

for n in (500, 10_000, 100_000, 1_000_000):
    M = cp.random.random((n, K))
    omin = cp.empty(K); omax = cp.empty(K)
    def row():
        kr((K,), (256,), (M, np.int64(n), np.int32(K), omin, omax))  # launch-ok: layout benchmark, the block size is part of the measured variant
    def col_resident():
        MT = cp.ascontiguousarray(M.T)
        kc((K,), (256,), (MT, np.int64(n), np.int32(K), omin, omax))  # launch-ok: layout benchmark, the block size is part of the measured variant
    def col_kernel_only():
        kc((K,), (256,), (MT0, np.int64(n), np.int32(K), omin, omax))  # launch-ok: layout benchmark, the block size is part of the measured variant
    MT0 = cp.ascontiguousarray(M.T)
    print(f"n={n:>9} row={timed(row)*1e3:9.3f}ms  col(transpose+kernel)={timed(col_resident)*1e3:9.3f}ms  col(kernel only)={timed(col_kernel_only)*1e3:9.3f}ms")
