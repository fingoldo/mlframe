"""cProfile harness for ``training.composite.ChainedWindowForecaster``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_chained_window_forecast``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import ChainedWindowForecaster


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    z_prev = rng.normal(size=n)
    z_curr = 0.9 * z_prev + rng.normal(scale=0.3, size=n)
    z_target = 0.9 * z_curr + rng.normal(scale=0.3, size=n)

    def make_features(z: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"f1": np.sin(z * 2), "f2": z**2, "f3": rng.normal(size=len(z))})

    return make_features(z_prev), make_features(z_curr), z_curr, z_target


def _run(n: int) -> None:
    X_prev, X_curr, y_curr, y_target = _make_dataset(n, seed=0)
    chained = ChainedWindowForecaster(stage1_estimator=LinearRegression(), stage2_estimator=LinearRegression())
    chained.fit(X_prev, X_curr, y_curr, y_target)
    chained.predict(X_curr)


if __name__ == "__main__":
    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
