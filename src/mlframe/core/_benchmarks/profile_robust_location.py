"""cProfile + scaling bench for robust location estimators (PZAD probweights).

Run:  python -m mlframe.core._benchmarks.profile_robust_location

robust_mean_mestimator is O(iters * n); geometric_median is O(iters * n * d). Both are njit.
This harness times them across n to decide whether a prange variant is warranted (per the
kernel ladder's skip clause). cProfile mis-attributes njit body time to the caller; use the wall numbers.
"""

from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np

from mlframe.core.robust_location import geometric_median, robust_mean_mestimator


def _best_of(fn, *a, n=7, **k):
    fn(*a, **k)  # warm JIT
    best = float("inf")
    for _ in range(n):
        t = time.perf_counter()
        fn(*a, **k)
        best = min(best, time.perf_counter() - t)
    return best


def bench():
    print("\n=== robust_mean_mestimator (tukey) vs geometric_median, wall ms ===")
    print(f"{'n':>10} {'robust_mean_ms':>16} {'geomedian2d_ms':>16}")
    rng = np.random.default_rng(0)
    for n in (1_000, 10_000, 100_000, 1_000_000):
        x = rng.normal(0, 1, size=n)
        X = rng.normal(0, 1, size=(n, 2))
        rm = _best_of(robust_mean_mestimator, x, n=5)
        gm = _best_of(geometric_median, X, n=5)
        print(f"{n:>10} {rm * 1e3:>16.3f} {gm * 1e3:>16.3f}")


def profile():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=100_000)
    X = rng.normal(0, 1, size=(100_000, 3))
    robust_mean_mestimator(x)  # warm
    geometric_median(X)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        robust_mean_mestimator(x, weight="meshalkin")
        geometric_median(X)
    pr.disable()
    print("\n=== cProfile (n=100k, x5) ===")
    pstats.Stats(pr).sort_stats("cumulative").print_stats(8)


if __name__ == "__main__":
    bench()
    profile()


def bench_parallel():
    from mlframe.core.robust_location import _robust_mean_irls, _robust_mean_irls_parallel
    from mlframe.core.robust_location import WEIGHTS
    print("\n=== robust_mean serial vs prange (tukey), wall ms ===")
    print(f"{'n':>10} {'serial_ms':>12} {'parallel_ms':>14} {'speedup':>9}")
    rng = np.random.default_rng(0)
    wc = WEIGHTS.index("tukey")
    for n in (10_000, 100_000, 1_000_000):
        x = np.ascontiguousarray(rng.normal(0, 1, size=n))
        s = _best_of(_robust_mean_irls, x, wc, 4.685, -1.0, 50, 1e-8, n=4)
        p = _best_of(_robust_mean_irls_parallel, x, wc, 4.685, -1.0, 50, 1e-8, n=4)
        print(f"{n:>10} {s*1e3:>12.2f} {p*1e3:>14.2f} {s/p:>8.2f}x")

if __name__ == "__main__" and "PARALLEL" in __import__("os").environ:
    bench_parallel()
