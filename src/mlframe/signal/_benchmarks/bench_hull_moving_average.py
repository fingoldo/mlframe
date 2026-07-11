"""cProfile harness for ``signal.hull_moving_average``.

Run: ``python -m mlframe.signal._benchmarks.bench_hull_moving_average``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.signal.hull_moving_average import hull_ma_deviation, hull_moving_average


def _run(n: int, window: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=n)) + 100
    for _ in range(n_calls):
        hull_moving_average(x, window)
        hull_ma_deviation(x, window)


if __name__ == "__main__":
    for n, window, n_calls in [(1000, 20, 200), (100000, 20, 200), (100000, 50, 500)]:
        t0 = time.perf_counter()
        _run(n, window, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} window={window:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100000, 50, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
