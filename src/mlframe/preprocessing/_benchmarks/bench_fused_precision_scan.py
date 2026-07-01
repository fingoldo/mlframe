"""REJECTED bench (2026-07): fuse all rounding precisions of ``is_variable_truly_continuous`` into one kernel pass.

Idea: instead of calling the per-precision distinct-count njit kernel once per precision (max_fract_digits sweeps over the sorted fractional
array), walk the array a single time keeping a running (prev, count, have_prev) per precision. Verdict: SLOWER at every max_fract_digits tested
(-6% .. -14% at n=100k continuous). The per-precision kernel wins via its simpler inner loop + the caller's early break on distinct-saturation;
the fused pass recomputes np.rint for ALL precisions per element even after a precision saturates. Not shipped; note left at cleaning.py call site.

Run: python -m mlframe.preprocessing._benchmarks.bench_fused_precision_scan
"""
import time

import numba
import numpy as np
from numba import njit

from mlframe.preprocessing._cleaning_kernels import _get_count_distinct_rounded_njit


@njit(cache=True)
def _count_distinct_rounded_all(sorted_vals, max_digits, skip0, skip1):
    n = sorted_vals.shape[0]
    counts = np.zeros(max_digits, dtype=np.int64)
    scales = np.empty(max_digits, dtype=np.float64)
    prev = np.empty(max_digits, dtype=np.float64)
    have_prev = np.zeros(max_digits, dtype=np.bool_)
    for d in range(max_digits):
        scales[d] = 10.0 ** (d + 1)
    for i in range(n):
        v = sorted_vals[i]
        if v != v:
            continue
        for d in range(max_digits):
            r = np.rint(v * scales[d]) / scales[d]
            if r == skip0 or (skip1 == skip1 and r == skip1):
                continue
            if (not have_prev[d]) or r != prev[d]:
                counts[d] += 1
                prev[d] = r
                have_prev[d] = True
    return counts


def main():
    per = _get_count_distinct_rounded_njit()
    np.random.seed(0)
    frac = np.sort(np.random.rand(100_000))
    per(frac, 3, 0.0, 1.0)
    _count_distinct_rounded_all(frac, 8, 0.0, 1.0)
    for maxd in (5, 8, 12):
        ca = _count_distinct_rounded_all(frac, maxd, 0.0, 1.0)
        for d in range(1, maxd + 1):
            assert per(frac, d, 0.0, 1.0) == ca[d - 1]
    print("identity OK")

    def bench_old(maxd):
        best = 1e9
        for _ in range(30):
            t = time.perf_counter()
            for d in range(1, maxd):
                per(frac, d, 0.0, 1.0)
            best = min(best, time.perf_counter() - t)
        return best * 1000

    def bench_new(maxd):
        best = 1e9
        for _ in range(30):
            t = time.perf_counter()
            _count_distinct_rounded_all(frac, maxd - 1, 0.0, 1.0)
            best = min(best, time.perf_counter() - t)
        return best * 1000

    for maxd in (5, 8, 12, 16):
        o = bench_old(maxd)
        nw = bench_new(maxd)
        print(f"max_fract_digits={maxd}: per-precision {o:.3f}ms fused {nw:.3f}ms ({100*(1-nw/o):.1f}% faster)")


if __name__ == "__main__":
    main()
