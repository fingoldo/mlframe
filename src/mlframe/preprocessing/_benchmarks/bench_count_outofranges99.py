"""iter99 bench: row-parallel prange for count_num_outofranges at n=10M.

Run:
    PYTHONPATH=src CUDA_VISIBLE_DEVICES="" NUMBA_DISABLE_CUDA=1 NUMBA_NUM_THREADS=8 python src/mlframe/preprocessing/_benchmarks/bench_count_outofranges99.py

The per-row out-of-range count is an order-invariant integer reduction, so parallelising over rows is bit-identical to the serial loop by construction.
Measured 2026-06-14 (8 threads): old (serial) vs new (prange), interleaved best-of-7 + separate-process.
    N=10M D= 8  old 0.0758  new 0.0249  3.04x   (separate-process: 0.1260 -> 0.0311, 4.05x)
    N=10M D= 4  old 0.0600  new 0.0158  3.79x
    N=10M D=30  old 0.2652  new 0.0840  3.16x
    N= 1M D= 8  old 0.0092  new 0.0026  3.56x
Checksums identical on every size. RESOLVED.
"""

import sys

sys.modules["cupy"] = None
import time

import numpy as np
import scipy.stats  # noqa: F401  (segfault-avoidance import-order on py3.14)
from numba import njit, prange


@njit(cache=True)
def _serial(X, mins, maxs):
    n, d = X.shape
    out = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c = 0
        for j in range(d):
            v = X[i, j]
            if v < mins[j] or v > maxs[j]:
                c += 1
        out[i] = c
    return out


@njit(cache=True, parallel=True)
def _parallel(X, mins, maxs):
    n, d = X.shape
    out = np.zeros(n, dtype=np.int64)
    for i in prange(n):
        c = 0
        for j in range(d):
            v = X[i, j]
            if v < mins[j] or v > maxs[j]:
                c += 1
        out[i] = c
    return out


def main():
    rng = np.random.default_rng(0)
    for N, D in [(10_000_000, 8), (10_000_000, 4), (10_000_000, 30), (1_000_000, 8)]:
        X = rng.standard_normal((N, D))
        mins = X[:1000].min(0)
        maxs = X[:1000].max(0)
        _serial(X[:1000], mins, maxs)
        _parallel(X[:1000], mins, maxs)
        ro = _serial(X, mins, maxs)
        rn = _parallel(X, mins, maxs)
        bo = bn = 1e9
        for _ in range(7):
            t = time.perf_counter()
            _serial(X, mins, maxs)
            bo = min(bo, time.perf_counter() - t)
            t = time.perf_counter()
            _parallel(X, mins, maxs)
            bn = min(bn, time.perf_counter() - t)
        print("N=%9d D=%2d  old %.4f  new %.4f  %.2fx  identical=%s" % (N, D, bo, bn, bo / bn, np.array_equal(ro, rn)))


if __name__ == "__main__":
    main()
