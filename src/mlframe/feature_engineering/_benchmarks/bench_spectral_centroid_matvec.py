"""Bench: spectral centroid/bandwidth power-weighted-mean reduction.

OLD: (spec * k[None, :]).sum(axis=1)  -> materializes a (rows, n_freq) temporary, then reduces.
NEW: spec @ k                          -> BLAS gemv, no temporary.

Both compute Sum_j spec[r, j] * k[j]. Bit-identity is NOT guaranteed (BLAS uses a
different/pairwise summation order than np.sum), but the divergence is FP reduction-order
(~1e-12 relative), well below any selection-altering threshold for an FE feature.

Run: CUDA_VISIBLE_DEVICES="" python bench_spectral_centroid_matvec.py
"""
import time
import numpy as np

RNG = np.random.default_rng(0)


def old_reduce(spec, k):
    return (spec * k[None, :]).sum(axis=1)


def new_reduce(spec, k):
    return spec @ k


def bench(rows, n_freq, iters=200):
    spec = RNG.random((rows, n_freq)) ** 2  # power spectrum-like, non-negative
    k = np.arange(n_freq, dtype=np.float64)

    o = old_reduce(spec, k)
    n = new_reduce(spec, k)
    max_rel = np.max(np.abs(o - n) / (np.abs(o) + 1e-30))

    # warm
    for _ in range(5):
        old_reduce(spec, k); new_reduce(spec, k)

    best_old = min(_t(old_reduce, spec, k, iters) for _ in range(5))
    best_new = min(_t(new_reduce, spec, k, iters) for _ in range(5))
    print(f"rows={rows:>7} n_freq={n_freq:>4} | OLD {best_old*1e3:8.3f}ms  NEW {best_new*1e3:8.3f}ms"
          f"  speedup {best_old/best_new:5.2f}x  max_rel_diff {max_rel:.2e}")


def _t(fn, spec, k, iters):
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(spec, k)
    return (time.perf_counter() - t0) / iters


if __name__ == "__main__":
    # n_freq fixed by window_K (=K//2+1). rows = windows per group segment.
    for K in (100, 256):
        nf = K // 2 + 1
        for rows in (1_000, 10_000, 100_000):
            bench(rows, nf)
        print()
