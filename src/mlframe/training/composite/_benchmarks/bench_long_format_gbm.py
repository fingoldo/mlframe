"""cProfile harness for ``training.composite.melt_to_long_gbm_features``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_long_format_gbm``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import melt_to_long_gbm_features


def _make_dataset(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{j}" for j in range(d)])
    y = rng.normal(size=n)
    return X, y


def _run(n: int, d: int) -> None:
    X, y = _make_dataset(n, d, seed=0)
    melt_to_long_gbm_features(X, y, model_factory=lambda: LinearRegression(), n_splits=5)


if __name__ == "__main__":
    for n, d in [(500, 20), (2_000, 50), (5_000, 100)]:
        t0 = time.perf_counter()
        _run(n, d)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} d={d:>4} (long rows={n*d:>9,}) -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
