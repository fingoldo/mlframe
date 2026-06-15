"""Bench: max-error reduction serial njit vs prange twin (iter130, @10M).

``fast_max_error`` dispatched only to ``_fast_max_error_seq`` while every sibling regression metric (MAE/MSE/R2)
already had a ``parallel=True`` twin + size dispatch. Max-of-abs-diff is an order-invariant comparison reduction, so a
prange twin is BIT-IDENTICAL (no FP reorder -- only > comparisons). This bench measures the crossover.

Run: PYTHONPATH=src python -m mlframe.metrics._benchmarks.bench_fast_max_error_par_iter130
"""
import sys
sys.modules.setdefault("cupy", None)
import time
import numpy as np
import numba

NP = dict(cache=True, fastmath=True)


@numba.njit(**NP)
def seq(yt, yp):
    n = len(yt)
    m = 0.0
    for i in range(n):
        d = abs(yt[i] - yp[i])
        if d > m:
            m = d
    return m


@numba.njit(**NP, parallel=True)
def par(yt, yp):
    n = len(yt)
    m = 0.0
    for i in numba.prange(n):
        m = max(m, abs(yt[i] - yp[i]))
    return m


def main():
    np.random.seed(1)
    for N in (100_000, 300_000, 500_000, 750_000, 1_000_000, 5_000_000, 10_000_000):
        yt = np.random.randn(N)
        yp = np.random.randn(N)
        seq(yt, yp)
        par(yt, yp)

        def b(f, n=9):
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                r = f(yt, yp)
                ts.append(time.perf_counter() - t)
            return min(ts), r

        ms, rs = b(seq)
        mp, rp = b(par)
        print(f"N={N:>9}: seq {ms*1000:8.3f}ms  par {mp*1000:8.3f}ms  speedup {ms/mp:5.2f}x  identical={rs == rp}")


if __name__ == "__main__":
    main()
