"""Microbench for CPX18 (count-distinct via _get_nunique fast path) and CPX19
(fused span-mask + fence counts) in preprocessing/cleaning.py.

Run:  CUDA_VISIBLE_DEVICES="" python bench_cleaning_cpx18_cpx19.py

OLD side = the actual prior numpy expressions (faithfully reproduced inline).
NEW side = the shipped helpers. Both warmed (njit JIT) before timing; best-of-N.
Identity is asserted inside (counts / n_outliers / values_in_span equal).
"""
import sys
sys.modules.setdefault("cupy", None)  # avoid pre-existing cupy native-AV at import on this box
import time
import numpy as np

from mlframe.preprocessing.cleaning import _get_nunique, _get_span_fence_njit


def best_of(fn, n=7):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


def make_data(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float64)
    # inject outliers + NaN
    x[rng.integers(0, n, n // 100)] *= 50.0
    x[rng.integers(0, n, n // 50)] = np.nan
    return x


def cpx18():
    print("=== CPX18: count distinct of values_in_span ===")
    for n in (100_000, 1_000_000):
        # values_in_span as it appears downstream: a span-filtered (no-NaN) subset
        x = make_data(n)
        q0, q1 = np.nanquantile(x, (0.1, 0.9))
        vis = x[(x >= q0) & (x <= q1)]  # no NaN
        # also the non-quantile branch where values_in_span == values (has NaN)
        old = lambda: len(np.unique(vis))
        new = lambda: _get_nunique(vis, skip_nan=False)
        assert old() == new(), (old(), new())
        # NaN-containing case
        old_n = len(np.unique(x))
        new_n = _get_nunique(x, skip_nan=False)
        assert old_n == new_n, (old_n, new_n)
        to = best_of(old)
        tn = best_of(new)
        print(f"  n={n:>9} span: OLD {to*1e3:7.3f}ms  NEW {tn*1e3:7.3f}ms  speedup {to/tn:5.2f}x  (count={old()})")


def cpx19():
    print("=== CPX19: fused span-mask + fence counts ===")
    kern = _get_span_fence_njit()
    # warm
    _ = kern(np.array([0.0, 1.0, np.nan]), 0.0, 1.0, -5.0, 5.0)
    for n in (100_000, 1_000_000):
        x = make_data(n)
        q0, q1 = np.nanquantile(x, (0.1, 0.9))
        iqr = q1 - q0
        m = 1.5
        lo = q0 - m * iqr
        hi = q1 + m * iqr

        def old():
            vis = x[(x >= q0) & (x <= q1)]
            no = (x < lo).sum() + (x > hi).sum()
            return vis, no

        def new():
            mask, nb, na = kern(x, q0, q1, lo, hi)
            return x[mask], nb + na

        vo, no = old()
        vn, nn = new()
        assert no == nn, (no, nn)
        assert np.array_equal(vo, vn), "values_in_span differ"
        to = best_of(old)
        tn = best_of(new)
        print(f"  n={n:>9}: OLD {to*1e3:7.3f}ms  NEW {tn*1e3:7.3f}ms  speedup {to/tn:5.2f}x  (n_outliers={no})")


if __name__ == "__main__":
    cpx18()
    cpx19()
