"""P-4 bench: is the y-marginal (ry) recompute inside mi_from_codes's x-loop worth hoisting?

The ``mi_from_codes`` RawKernel recomputes ``ry = sum_x sh[x*Ky+yy]`` for every (xx, yy) cell with a nonzero
count -- it is xx-invariant, so it is O(Kx^2 * Ky) redundant work. BUT it runs on a SINGLE thread (tid==0)
per column, in the kernel's tail AFTER the n-row atomicAdd histogram, which is the real cost. For nbins-scale
Kx/Ky (<= ~20) the tail is a few thousand serial ops per column vs an n=1M histogram -> immeasurable.

Measured (RTX 500 Ada, cupy 14.1): whole binned_mi_from_codes_gpu call, warm, median of 20:
  n=200k K=200 Kx=Ky=10  -> 18.7 ms
  n=1M   K=100 Kx=Ky=10  -> 52.1 ms
  n=200k K=200 Kx=Ky=20  -> 19.4 ms   (Kx/Ky doubled -> tail work 8x, wall +0.7 ms == histogram-size noise)

Doubling Kx/Ky (which 8x's the ry-recompute tail) moves the wall by <4%, and that delta is the larger
histogram, not the reduce. A bit-identical hoist would need an 8-byte-aligned shared scratch region for the
Ky y-marginals (or a bounded static array) for no measurable end-to-end win. Verdict: REJECT (negligible).

Run: PYTHONPATH=src python -m mlframe.feature_selection.filters._benchmarks.bench_mi_from_codes_ymarginal_hoist
"""
from __future__ import annotations

import time

import numpy as np


def main() -> None:
    import cupy as cp

    from .._fe_batched_mi import binned_mi_from_codes_gpu

    rng = np.random.default_rng(0)
    for n, K, Kx, Ky in [(200000, 200, 10, 10), (1000000, 100, 10, 10), (200000, 200, 20, 20)]:
        C = cp.asarray(rng.integers(0, Kx, size=(n, K), dtype=np.int64))
        y = cp.asarray(rng.integers(0, Ky, size=n, dtype=np.int64))
        for _ in range(3):
            binned_mi_from_codes_gpu(C, y, ky=Ky, codes_trusted=True)
        cp.cuda.Stream.null.synchronize()
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            binned_mi_from_codes_gpu(C, y, ky=Ky, codes_trusted=True)
            cp.cuda.Stream.null.synchronize()
            ts.append(time.perf_counter() - t)
        print(f"n={n} K={K} Kx={Kx} Ky={Ky}: median {np.median(ts) * 1e3:.3f} ms  min {min(ts) * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
