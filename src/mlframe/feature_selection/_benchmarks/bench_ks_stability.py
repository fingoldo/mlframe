"""cProfile harness for ``feature_selection.filters.ks_stability_filter``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_ks_stability``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._ks_stability import ks_stability_filter


def _run(n_rows: int, n_cols: int) -> None:
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_cols)})
    test_df = pd.DataFrame({f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_cols)})
    ks_stability_filter(train_df, test_df)


def _run_multi_split(n_rows: int, n_cols: int, n_splits: int) -> None:
    rng = np.random.default_rng(0)
    train_df = pd.DataFrame({f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_cols)})
    test_df = pd.DataFrame({f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_cols)})
    ks_stability_filter(train_df, test_df, n_splits=n_splits)


if __name__ == "__main__":
    for n_rows, n_cols in [(20_000, 20), (200_000, 50)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    for n_rows, n_cols, n_splits in [(20_000, 20, 9), (200_000, 50, 9)]:
        t0 = time.perf_counter()
        _run_multi_split(n_rows, n_cols, n_splits)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>9,} n_cols={n_cols:>3} n_splits={n_splits} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_multi_split(20_000, 20, 9)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("multi-split (n_splits=9):")
    print(buf.getvalue())
