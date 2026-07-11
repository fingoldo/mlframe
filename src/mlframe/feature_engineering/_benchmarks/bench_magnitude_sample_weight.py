"""cProfile harness for ``feature_engineering.magnitude_sample_weight.magnitude_sample_weight``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_magnitude_sample_weight``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.magnitude_sample_weight import magnitude_sample_weight


def _run(n: int, n_targets: int, n_calls: int, robust: bool = False) -> None:
    rng = np.random.default_rng(0)
    y_multi = rng.normal(size=(n, n_targets))
    for _ in range(n_calls):
        magnitude_sample_weight(y_multi, norm="mean_abs", robust=robust)


if __name__ == "__main__":
    for n, n_targets, n_calls in [(10000, 4, 1000), (1000000, 4, 100), (1000000, 20, 100)]:
        t0 = time.perf_counter()
        _run(n, n_targets, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>9} n_targets={n_targets:>3} n_calls={n_calls:>5} robust=False -> {wall * 1000:9.2f} ms")

        t0 = time.perf_counter()
        _run(n, n_targets, n_calls, robust=True)
        wall = time.perf_counter() - t0
        print(f"n={n:>9} n_targets={n_targets:>3} n_calls={n_calls:>5} robust=True  -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 20, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_robust = cProfile.Profile()
    profiler_robust.enable()
    _run(1000000, 20, 200, robust=True)
    profiler_robust.disable()
    buf_robust = StringIO()
    stats_robust = pstats.Stats(profiler_robust, stream=buf_robust).sort_stats("cumulative")
    stats_robust.print_stats(15)
    print(buf_robust.getvalue())
