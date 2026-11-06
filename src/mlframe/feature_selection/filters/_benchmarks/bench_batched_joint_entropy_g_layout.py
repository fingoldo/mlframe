"""Row-major versus column-major X read for batched_joint_entropy2_g (one block per column group, thread-strided rows)."""

import time

import cupy as cp
import numpy as np

BODY = r"""
extern "C" __global__
void jh_{tag}(const long long* __restrict__ X, const long long* __restrict__ b, const int Kb, const long long n, const int K, const int M,
              unsigned int* __restrict__ scratch, double* __restrict__ out) {{
    unsigned int* hist = scratch + (long long)blockIdx.x * M;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int col = blockIdx.x; col < K; col += gridDim.x) {{
        for (int i = tid; i < M; i += nt) hist[i] = 0u;
        __syncthreads();
        for (long long i = tid; i < n; i += nt) {{
            long long xv = {READ};
            atomicAdd(&hist[(unsigned int)(xv * Kb + b[i])], 1u);
        }}
        __syncthreads();
        if (tid == 0) {{ double s = 0.0; for (int q = 0; q < M; ++q) s += hist[q]; out[col] = s; }}
        __syncthreads();
    }}
}}"""
ROW = BODY.format(tag="row", READ="X[i * (long long)K + col]")
COL = BODY.format(tag="col", READ="X[(long long)col * n + i]")
mod = cp.RawModule(code=ROW + COL)
k_row = mod.get_function("jh_row")
k_col = mod.get_function("jh_col")
K, KB, KX = 16, 8, 8
M = KX * KB
GRID = 64


def timed(fn, reps=5):
    fn()
    cp.cuda.Device().synchronize()
    best = float("inf")
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        best = min(best, time.perf_counter() - s)
    return best


for n in (10_000, 100_000, 1_000_000):
    X = cp.random.randint(0, KX, size=(n, K), dtype=cp.int64)
    XT = cp.ascontiguousarray(X.T)
    b = cp.random.randint(0, KB, size=n, dtype=cp.int64)
    scratch = cp.zeros(GRID * M, dtype=cp.uint32)
    out = cp.empty(K, dtype=cp.float64)
    t_row = timed(lambda: k_row((GRID,), (256,), (X, b, np.int32(KB), np.int64(n), np.int32(K), np.int32(M), scratch, out)))
    t_col = timed(lambda: k_col((GRID,), (256,), (XT, b, np.int32(KB), np.int64(n), np.int32(K), np.int32(M), scratch, out)))
    t_tr = timed(lambda: cp.ascontiguousarray(X.T))
    print(f"n={n:>9} row={t_row * 1e3:9.3f}ms  col(kernel only)={t_col * 1e3:9.3f}ms  transpose={t_tr * 1e3:8.3f}ms")
