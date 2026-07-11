"""cProfile harness for ``signal.hull_moving_average``.

Run: ``python -m mlframe.signal._benchmarks.bench_hull_moving_average``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.signal.hull_moving_average import hull_ma_deviation, hull_moving_average, hull_moving_average_multi


def _run(n: int, window: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=n)) + 100
    for _ in range(n_calls):
        hull_moving_average(x, window)
        hull_ma_deviation(x, window)


def _run_multi_looped(n: int, windows: list[int], n_calls: int) -> None:
    """Baseline: what a caller does WITHOUT ``hull_moving_average_multi`` -- one ``hull_moving_average``
    call per window, each recomputing the input series' cumsum from scratch."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=n)) + 100
    for _ in range(n_calls):
        for window in windows:
            hull_moving_average(x, window)


def _run_multi_shared(n: int, windows: list[int], n_calls: int) -> None:
    """``hull_moving_average_multi``: shares one cumsum of the input series across all windows' fast/slow
    SMA passes instead of recomputing it once per window."""
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=n)) + 100
    for _ in range(n_calls):
        hull_moving_average_multi(x, windows)


if __name__ == "__main__":
    for n, window, n_calls in [(1000, 20, 200), (100000, 20, 200), (100000, 50, 500)]:
        t0 = time.perf_counter()
        _run(n, window, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} window={window:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    windows = [10, 20, 50, 100, 200]
    for n, n_calls in [(1000, 200), (100000, 200)]:
        t0 = time.perf_counter()
        _run_multi_looped(n, windows, n_calls)
        wall_looped = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_multi_shared(n, windows, n_calls)
        wall_shared = time.perf_counter() - t0

        speedup = wall_looped / wall_shared if wall_shared > 0 else float("nan")
        print(f"n={n:>7} windows={windows} n_calls={n_calls:>4} -> looped={wall_looped * 1000:9.2f} ms  multi={wall_shared * 1000:9.2f} ms  speedup={speedup:.2f}x")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100000, 50, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_multi_shared(100000, windows, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
