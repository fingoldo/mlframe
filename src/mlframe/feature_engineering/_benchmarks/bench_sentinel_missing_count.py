"""cProfile harness for ``feature_engineering.sentinel_missing_count.add_sentinel_missing_count_feature``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_sentinel_missing_count``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.sentinel_missing_count import add_sentinel_missing_count_feature, detect_sentinel_values


def _make_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])


def _make_per_column_sentinel_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    """Half the columns sentinel-coded -1, half -999 -- exercises the per-column/auto-detect path."""
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_rows, n_cols))
    half = n_cols // 2
    mask = rng.random((n_rows, n_cols)) < 0.1
    values[:, :half] = np.where(mask[:, :half], -1.0, values[:, :half])
    values[:, half:] = np.where(mask[:, half:], -999.0, values[:, half:])
    return pd.DataFrame(values, columns=[f"f{i}" for i in range(n_cols)])


def _run(n_rows: int, n_cols: int) -> None:
    df = _make_dataset(n_rows, n_cols, seed=0)
    add_sentinel_missing_count_feature(df, sentinel=-1.0)


def _run_per_column(n_rows: int, n_cols: int) -> None:
    df = _make_per_column_sentinel_dataset(n_rows, n_cols, seed=0)
    half = n_cols // 2
    per_column_sentinels = {f"f{i}": -1.0 for i in range(half)} | {f"f{i}": -999.0 for i in range(half, n_cols)}
    add_sentinel_missing_count_feature(df, per_column_sentinels=per_column_sentinels)


def _run_auto_detect(n_rows: int, n_cols: int) -> None:
    df = _make_per_column_sentinel_dataset(n_rows, n_cols, seed=0)
    add_sentinel_missing_count_feature(df, auto_detect_sentinels=True)


if __name__ == "__main__":
    for n_rows, n_cols in [(50000, 50), (1000000, 50), (1000000, 200)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"global-sentinel     n_rows={n_rows:>8} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    for n_rows, n_cols in [(50000, 50), (1000000, 50), (1000000, 200)]:
        t0 = time.perf_counter()
        _run_per_column(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"per-column-sentinel n_rows={n_rows:>8} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    for n_rows, n_cols in [(50000, 50), (1000000, 50), (1000000, 200)]:
        t0 = time.perf_counter()
        _run_auto_detect(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"auto-detect         n_rows={n_rows:>8} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("=== global sentinel (unchanged default path) ===")
    print(buf.getvalue())

    profiler2 = cProfile.Profile()
    profiler2.enable()
    _run_per_column(1000000, 200)
    profiler2.disable()
    buf2 = StringIO()
    stats2 = pstats.Stats(profiler2, stream=buf2).sort_stats("cumulative")
    stats2.print_stats(15)
    print("=== per-column sentinels (new path) ===")
    print(buf2.getvalue())

    profiler3 = cProfile.Profile()
    profiler3.enable()
    _run_auto_detect(1000000, 200)
    profiler3.disable()
    buf3 = StringIO()
    stats3 = pstats.Stats(profiler3, stream=buf3).sort_stats("cumulative")
    stats3.print_stats(15)
    print("=== auto-detect sentinels (new path) ===")
    print(buf3.getvalue())
