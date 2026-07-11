"""cProfile harness for ``feature_engineering.multi_decomposition_bank.multi_decomposition_feature_bank``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_multi_decomposition_bank``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.multi_decomposition_bank import multi_decomposition_feature_bank


def _make_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n_rows, n_cols)), columns=[f"f{i}" for i in range(n_cols)])


def _run(n_rows: int, n_cols: int, n_components: int) -> None:
    df = _make_dataset(n_rows, n_cols, seed=0)
    multi_decomposition_feature_bank(df, n_components=n_components, methods=("svd", "pca", "grp", "srp"))


def _run_pruned(n_rows: int, n_cols: int, n_components: int) -> None:
    df = _make_dataset(n_rows, n_cols, seed=0)
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=n_rows)
    multi_decomposition_feature_bank(
        df, n_components=n_components, methods=("svd", "pca", "grp", "srp"), y=y, prune_uninformative_methods=True
    )


if __name__ == "__main__":
    for n_rows, n_cols, n_components in [(2000, 50, 10), (2000, 200, 10), (20000, 200, 10)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols, n_components)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>6} n_cols={n_cols:>4} n_components={n_components:>3} -> {wall * 1000:9.2f} ms")

    for n_rows, n_cols, n_components in [(2000, 50, 10), (2000, 200, 10), (20000, 200, 10)]:
        t0 = time.perf_counter()
        _run_pruned(n_rows, n_cols, n_components)
        wall = time.perf_counter() - t0
        print(f"[pruned] n_rows={n_rows:>6} n_cols={n_cols:>4} n_components={n_components:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20000, 200, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_pruned(20000, 200, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("[pruned]")
    print(buf.getvalue())
