"""cProfile harness for ``feature_engineering.randomize_as_of_lag``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_random_lag_augmentation``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering import randomize_as_of_lag


def _make_as_of(n: int) -> pd.DataFrame:
    return pd.DataFrame({"entity": np.arange(n), "as_of": 100.0})


def _run(n: int) -> None:
    randomize_as_of_lag(_make_as_of(n), "as_of", max_lag=15.0, min_lag=0.0, random_state=0)


if __name__ == "__main__":
    for n in [10_000, 100_000, 1_000_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>9,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
