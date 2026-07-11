"""cProfile harness for ``feature_engineering.variance_gated_pairwise_diff.variance_gated_pairwise_diff``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_variance_gated_pairwise_diff``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.variance_gated_pairwise_diff import variance_gated_pairwise_diff


def _make_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])


def _run(n_rows: int, n_cols: int) -> None:
    df = _make_dataset(n_rows, n_cols, seed=0)
    variance_gated_pairwise_diff(df, list(df.columns), min_variance=0.5)


if __name__ == "__main__":
    for n_rows, n_cols in [(2000, 50), (2000, 100), (5000, 150)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        n_pairs = n_cols * (n_cols - 1) // 2
        print(f"n_rows={n_rows:>5} n_cols={n_cols:>4} n_pairs={n_pairs:>6} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5000, 150)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
