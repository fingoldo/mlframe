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


# Empirical staleness histogram skewed toward short lags with a long tail, e.g. observed serving logs.
_HIST_EDGES = [0.0, 1.0, 2.0, 5.0, 15.0]
_HIST_COUNTS = [70.0, 15.0, 10.0, 5.0]


def _run_histogram(n: int) -> None:
    randomize_as_of_lag(
        _make_as_of(n),
        "as_of",
        max_lag=15.0,
        min_lag=0.0,
        random_state=0,
        lag_histogram_edges=_HIST_EDGES,
        lag_histogram_counts=_HIST_COUNTS,
    )


if __name__ == "__main__":
    for n in [10_000, 100_000, 1_000_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"uniform      n={n:>9,} -> {wall * 1000:9.2f} ms")

    for n in [10_000, 100_000, 1_000_000]:
        t0 = time.perf_counter()
        _run_histogram(n)
        wall = time.perf_counter() - t0
        print(f"histogram    n={n:>9,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("uniform path:")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_histogram(1_000_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("histogram path:")
    print(buf.getvalue())
