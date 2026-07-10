"""cProfile harness for ``feature_engineering.boolean_pair_interactions.boolean_pair_interactions``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_boolean_pair_interactions``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.boolean_pair_interactions import boolean_pair_interactions


def _make_dataset(n_rows: int, n_binary_cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"b{i}": rng.integers(0, 2, n_rows) for i in range(n_binary_cols)})


def _run(n_rows: int, n_binary_cols: int) -> None:
    df = _make_dataset(n_rows, n_binary_cols, seed=0)
    boolean_pair_interactions(df)


if __name__ == "__main__":
    for n_rows, n_binary_cols in [(5000, 20), (5000, 80), (50000, 80)]:
        t0 = time.perf_counter()
        _run(n_rows, n_binary_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>6} n_binary_cols={n_binary_cols:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 80)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
