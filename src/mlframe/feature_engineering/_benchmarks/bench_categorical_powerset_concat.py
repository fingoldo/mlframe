"""cProfile harness for ``feature_engineering.categorical_powerset_concat.categorical_powerset_concat``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_categorical_powerset_concat``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.categorical_powerset_concat import categorical_powerset_concat


def _make_dataset(n_rows: int, n_columns: int, n_categories: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({chr(ord("a") + c): rng.integers(0, n_categories, n_rows).astype(str) for c in range(n_columns)})


def _run(n_rows: int, n_columns: int, n_categories: int, max_order=None) -> None:
    df = _make_dataset(n_rows, n_columns, n_categories, seed=0)
    columns = list(df.columns)
    categorical_powerset_concat(df, columns, max_order=max_order)


if __name__ == "__main__":
    for n_rows, n_columns, n_categories in [(50000, 3, 20), (1000000, 3, 20), (200000, 5, 20)]:
        t0 = time.perf_counter()
        _run(n_rows, n_columns, n_categories)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>8} n_columns={n_columns:>2} n_categories={n_categories:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(200000, 5, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
