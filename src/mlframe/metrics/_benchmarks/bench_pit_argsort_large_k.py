"""Benchmark (gated): _fast_pit O(K^2) insertion sort vs an njit argsort dispatch
at large K. Ships only if measurably faster AND bit-identical. PIT diagrams use
K<=20 typically, but this probes K in {50, 100, 200} to see if a threshold pays.
"""

from __future__ import annotations

import time

import numpy as np
import numba

from mlframe.metrics.quantile import _fast_pit, _NJIT_KW


@numba.njit(**_NJIT_KW)
def _fast_pit_argsort(P, y, a_arr):
    n = P.shape[0]
    k = P.shape[1]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        row = P[i]
        order = np.argsort(row)
        sq = row[order]
        sa = a_arr[order]
        yi = y[i]
        if yi <= sq[0]:
            out[i] = sa[0]
        elif yi >= sq[k - 1]:
            out[i] = sa[k - 1]
        else:
            t = 0
            while t < k - 1 and yi > sq[t + 1]:
                t += 1
            x0 = sq[t]
            x1 = sq[t + 1]
            if x1 == x0:
                out[i] = sa[t]
            else:
                slope = (sa[t + 1] - sa[t]) / (x1 - x0)
                out[i] = slope * (yi - x0) + sa[t]
    return out


def _time(fn, *a, repeat=5):
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*a)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(1)
    # warm up
    P0 = np.ascontiguousarray(np.sort(rng.standard_normal((100, 10)), axis=1))
    y0 = rng.standard_normal(100)
    a0 = np.linspace(0.05, 0.95, 10)
    _fast_pit(P0, y0, a0)
    _fast_pit_argsort(P0, y0, a0)

    print(f"{'N':>8} {'K':>5} {'insort(ms)':>12} {'argsort(ms)':>12} {'speedup':>9} {'identical':>10}")
    for n, k in [(50_000, 20), (50_000, 50), (50_000, 100), (20_000, 200)]:
        P = np.ascontiguousarray(np.sort(rng.standard_normal((n, k)), axis=1))
        y = rng.standard_normal(n)
        a = np.linspace(0.02, 0.98, k)
        r1 = _fast_pit(P, y, a)
        r2 = _fast_pit_argsort(P, y, a)
        identical = np.array_equal(r1, r2)
        t1 = _time(_fast_pit, P, y, a) * 1e3
        t2 = _time(_fast_pit_argsort, P, y, a) * 1e3
        print(f"{n:>8} {k:>5} {t1:>12.3f} {t2:>12.3f} {t1 / t2:>8.2f}x {str(identical):>10}")


if __name__ == "__main__":
    main()
