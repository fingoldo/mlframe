"""cProfile harness for ``calibration.ensembling.odds_ratio_combine``.

Run: ``python -m mlframe.calibration._benchmarks.bench_odds_ratio_combine``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.calibration.ensembling import odds_ratio_combine


def _run(n: int, k: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, size=(n, k))
    w = rng.uniform(0.5, 1.5, size=k)
    for _ in range(n_calls):
        odds_ratio_combine(p)
        odds_ratio_combine(p, weights=w)


if __name__ == "__main__":
    _run(50, 3, 1)  # warm every njit variant (single-thread + parallel, weighted + unweighted) before timing
    _run(30_000, 3, 1)

    for n, k, n_calls in [(1_000, 10, 200), (100_000, 20, 20), (1_000_000, 5, 5)]:
        t0 = time.perf_counter()
        _run(n, k, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} k={k:>2} n_calls={n_calls:>4} -> {wall * 1000:8.2f} ms total, {wall / n_calls * 1e6:8.2f} us/call-pair")

    n, k, n_calls = 100_000, 20, 50
    profiler = cProfile.Profile()
    profiler.enable()
    _run(n, k, n_calls)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
