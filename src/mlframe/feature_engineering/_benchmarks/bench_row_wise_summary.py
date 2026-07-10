"""cProfile harness for ``feature_engineering.row_wise_summary_stats``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_row_wise_summary``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering import row_wise_summary_stats


def _make_dataset(n: int, d: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])


def _run(n: int, d: int) -> None:
    X = _make_dataset(n, d, seed=0)
    row_wise_summary_stats(X, stats=("mean", "std", "q10", "q50", "q90"))


if __name__ == "__main__":
    for n, d in [(10_000, 30), (100_000, 30), (100_000, 200)]:
        t0 = time.perf_counter()
        _run(n, d)
        wall = time.perf_counter() - t0
        print(f"n={n:>8,} d={d:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
