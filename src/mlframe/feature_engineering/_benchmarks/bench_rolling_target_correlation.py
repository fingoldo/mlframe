"""cProfile harness for ``feature_engineering.rolling_target_correlation_tracker``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_rolling_target_correlation``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering import rolling_target_correlation_tracker


def _make_dataset(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = X["f0"].to_numpy() + rng.normal(scale=0.3, size=n)
    return X, y


def _run(n: int, d: int, window: int) -> None:
    X, y = _make_dataset(n, d, seed=0)
    rolling_target_correlation_tracker(X, y, window=window, min_periods=max(10, window // 10))


if __name__ == "__main__":
    for n, d, window in [(5_000, 10, 200), (20_000, 10, 200), (50_000, 20, 500)]:
        t0 = time.perf_counter()
        _run(n, d, window)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} d={d:>3} window={window:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 20, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
