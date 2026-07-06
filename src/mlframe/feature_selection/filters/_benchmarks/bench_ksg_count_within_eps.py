"""Bench for _ksg._count_within_eps vectorisation (CPX7).

The pre-fix implementation looped in pure Python issuing 2*N np.searchsorted
calls per MI pair; this was the dominant cost in mixed_ksg_mi (cProfile: 0.551s
of 0.687s for 5 calls at n=5000, vs 0.133s for the joint KDTree). The fix
issues two array-valued searchsorted calls (O(N log N) in C). Bit-identical.

Run: python bench_ksg_count_within_eps.py
"""
from __future__ import annotations

import time

import numpy as np


def _old_count_within_eps(arr_1d, eps):
    arr = arr_1d.astype(np.float64).ravel()
    sorter = np.argsort(arr)
    sorted_arr = arr[sorter]
    counts = np.empty(arr.size, dtype=np.int64)
    n = arr.size
    for i in range(n):
        lo = arr[i] - eps[i] + 1e-12
        hi = arr[i] + eps[i] - 1e-12
        lo_idx = np.searchsorted(sorted_arr, lo, side="left")
        hi_idx = np.searchsorted(sorted_arr, hi, side="right")
        counts[i] = max(0, (hi_idx - lo_idx) - 1)
    return counts


def _best(fn, x, eps, reps=7):
    times = []
    for _ in range(reps):
        s = time.perf_counter()
        fn(x, eps)
        times.append(time.perf_counter() - s)
    return min(times)


def main():
    from mlframe.feature_selection.filters._ksg import _count_within_eps

    rng = np.random.default_rng(0)
    for n in (5000, 20000):
        x = rng.standard_normal(n)
        eps = np.abs(rng.standard_normal(n)) * 0.1 + 0.01
        old = _old_count_within_eps(x, eps)
        new = _count_within_eps(x, eps)
        assert np.array_equal(old, new), f"identity mismatch at n={n}"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input
        t_old = _best(_old_count_within_eps, x, eps)
        t_new = _best(_count_within_eps, x, eps)
        print(f"n={n}: OLD {t_old*1e3:.2f}ms -> NEW {t_new*1e3:.2f}ms " f"({t_old/t_new:.1f}x) identity OK")


if __name__ == "__main__":
    main()
