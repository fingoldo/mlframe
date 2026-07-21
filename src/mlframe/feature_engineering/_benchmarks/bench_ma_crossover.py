"""cProfile harness for ``feature_engineering.ma_crossover.ma_crossover_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_ma_crossover``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.ma_crossover import ma_crossover_features


def _run(n: int, n_windows: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=n).cumsum() + 100)
    windows = [3 * (i + 1) for i in range(n_windows)]
    mas = {w: x.rolling(w).mean() for w in windows}
    group_ids = np.repeat(np.arange(max(n // 1000, 1)), 1000)[:n]
    for _ in range(n_calls):
        ma_crossover_features(mas)
        ma_crossover_features(mas, group_ids=group_ids)
        ma_crossover_features(mas, short_window_weight_power=2.0)


if __name__ == "__main__":
    for n, n_windows, n_calls in [(2000, 6, 50), (200000, 6, 50), (200000, 15, 50)]:
        t0 = time.perf_counter()
        _run(n, n_windows, n_calls)
        wall = time.perf_counter() - t0
        print(f"n={n:>7} n_windows={n_windows:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(200000, 15, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
