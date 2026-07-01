"""cProfile + serial/parallel backend bench for Nadaraya-Watson smoothing (PZAD traffic).

Run:  python -m mlframe.feature_engineering._benchmarks.profile_nadaraya_watson

NW is O(n_query * n_train); the prange-over-queries kernel is the optimization. This A/Bs
serial vs parallel across sizes to justify the dispatch threshold ``_NW_PARALLEL_MIN_QUERIES``.
cProfile mis-attributes njit body time to the caller frame; use the wall A/B, not tottime.
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.feature_engineering.nadaraya_watson import _nw_parallel, _nw_serial, nadaraya_watson_smooth


def _best_of(fn, *args, n=5):
    fn(*args)
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - t)
    return best


def bench_backends():
    print("\n=== NW backend A/B (serial vs prange over queries), gaussian ===")
    print(f"{'n':>8} {'serial_ms':>12} {'parallel_ms':>14} {'speedup':>9}")
    for n in (500, 2000, 10000, 50000):
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 100, size=n))
        y = np.sin(x) + rng.normal(0, 0.2, size=n)
        w = np.ones(1)
        args = (x, x, y, w, 0, 1.0, False)
        s = _best_of(_nw_serial, *args)
        p = _best_of(_nw_parallel, *args)
        print(f"{n:>8} {s * 1e3:>12.3f} {p * 1e3:>14.3f} {s / p:>8.2f}x")


def profile_full():
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 100, size=10000))
    y = np.sin(x) + rng.normal(0, 0.2, size=10000)
    nadaraya_watson_smooth(x, y, bandwidth=1.0)  # warm
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        nadaraya_watson_smooth(x, y, bandwidth=1.0)
    pr.disable()
    print("\n=== cProfile (n=10000 in-sample, x3) ===")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(8)


if __name__ == "__main__":
    bench_backends()
    profile_full()
