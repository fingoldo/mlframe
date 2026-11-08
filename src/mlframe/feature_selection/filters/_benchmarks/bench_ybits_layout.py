"""Label-bitset read of null_mi_perms_bits: lanes are permutations, one warp per word.

``(P, W)`` row-per-permutation layout (current) against ``(W, P)`` word-major layout: the same XOR-popcount work per load.
"""

import time

import cupy as cp
import numpy as np

SRC = r"""
extern "C" __global__
void read_pw(const unsigned int* __restrict__ yb, const int W, const int P, unsigned int* __restrict__ out) {
    // one warp per block; lane l handles permutations l, l+32, ...; accumulates popcount of the label words.
    const int lane = threadIdx.x & 31;
    unsigned int acc = 0;
    for (int w = blockIdx.x; w < W; w += gridDim.x) {
        for (int q = 0; q < (P + 31) / 32; ++q) {
            const int p = lane + 32 * q;
            if (p < P) acc += __popc(yb[(long long)p * W + w]);
        }
    }
    if (acc == 0xFFFFFFFFu) out[0] = acc;
}
extern "C" __global__
void read_wp(const unsigned int* __restrict__ yb, const int W, const int P, unsigned int* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    unsigned int acc = 0;
    for (int w = blockIdx.x; w < W; w += gridDim.x) {
        for (int q = 0; q < (P + 31) / 32; ++q) {
            const int p = lane + 32 * q;
            if (p < P) acc += __popc(yb[(long long)w * P + p]);
        }
    }
    if (acc == 0xFFFFFFFFu) out[0] = acc;
}
"""
mod = cp.RawModule(code=SRC)
k_pw = mod.get_function("read_pw")
k_wp = mod.get_function("read_wp")


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


for n in (100_000, 1_000_000):
    for P in (32, 128):
        W = (n + 31) // 32
        yb_pw = cp.random.randint(0, 2**31, size=(P, W), dtype=cp.uint32)
        yb_wp = cp.ascontiguousarray(yb_pw.T)
        out = cp.zeros(1, dtype=cp.uint32)
        blocks = 256
        t_pw = timed(lambda: k_pw((blocks,), (32,), (yb_pw, np.int32(W), np.int32(P), out)))  # launch-ok: layout benchmark, the block size is part of the measured variant
        t_wp = timed(lambda: k_wp((blocks,), (32,), (yb_wp, np.int32(W), np.int32(P), out)))  # launch-ok: layout benchmark, the block size is part of the measured variant
        print(f"n={n:>9} P={P:>4}  (P,W) row-per-perm={t_pw * 1e3:9.3f}ms  (W,P) word-major={t_wp * 1e3:9.3f}ms  speedup={t_pw / t_wp:5.2f}x")
