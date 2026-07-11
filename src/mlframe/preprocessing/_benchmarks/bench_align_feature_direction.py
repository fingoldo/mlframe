"""cProfile harness for ``preprocessing.align_feature_direction.align_feature_direction``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_align_feature_direction``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.align_feature_direction import align_feature_direction, check_feature_direction_stability


def _make_dataset(n_rows: int, n_cols: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n_rows)
    df = pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])
    return df, y


def _run(n_rows: int, n_cols: int) -> None:
    df, y = _make_dataset(n_rows, n_cols, seed=0)
    align_feature_direction(df, y)


def _run_stability(n_rows: int, n_cols: int, n_folds: int) -> None:
    df, y = _make_dataset(n_rows, n_cols, seed=0)
    check_feature_direction_stability(df, y, n_folds=n_folds)


if __name__ == "__main__":
    for n_rows, n_cols in [(5000, 50), (50000, 50), (50000, 500)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>6} n_cols={n_cols:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    print("--- check_feature_direction_stability (opt-in, K-fold sign-stability check) ---")
    for n_rows, n_cols, n_folds in [(5000, 50, 5), (50000, 50, 5), (50000, 500, 5)]:
        t0 = time.perf_counter()
        _run_stability(n_rows, n_cols, n_folds)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>6} n_cols={n_cols:>4} n_folds={n_folds} -> {wall * 1000:9.2f} ms")

    profiler2 = cProfile.Profile()
    profiler2.enable()
    _run_stability(50000, 500, 5)
    profiler2.disable()
    buf2 = StringIO()
    stats2 = pstats.Stats(profiler2, stream=buf2).sort_stats("cumulative")
    stats2.print_stats(15)
    print(buf2.getvalue())
