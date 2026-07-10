"""cProfile harness for ``feature_engineering.select_significant_lags``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_acf_lag_selection``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.acf_lag_selection import select_significant_lags


def _run(n_points: int, max_lag: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    series = rng.normal(0, 1, n_points)
    for _ in range(n_calls):
        select_significant_lags(series, max_lag=max_lag)


if __name__ == "__main__":
    for n_points, max_lag, n_calls in [(10_000, 50, 100), (1_000_000, 50, 20)]:
        t0 = time.perf_counter()
        _run(n_points, max_lag, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_points={n_points:>9,} max_lag={max_lag:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:9.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000, 50, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
