"""cProfile harness for ``feature_engineering.row_wise_extremality.row_wise_extremality_index``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_row_wise_extremality``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.row_wise_extremality import row_wise_extremality_index, row_wise_top_k_extreme_columns


def _make_frame(n: int, n_cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, n_cols)), columns=[f"f{i}" for i in range(n_cols)])


def _run(n: int, n_cols: int, n_calls: int) -> None:
    df = _make_frame(n, n_cols)
    for _ in range(n_calls):
        row_wise_extremality_index(df)


def _run_top_k(n: int, n_cols: int, n_calls: int, k: int = 3) -> None:
    df = _make_frame(n, n_cols)
    for _ in range(n_calls):
        row_wise_top_k_extreme_columns(df, k=k)


if __name__ == "__main__":
    for n, n_cols, n_calls in [(2000, 20, 20), (200000, 20, 10), (200000, 200, 5)]:
        t0 = time.perf_counter()
        _run(n, n_cols, n_calls)
        wall = time.perf_counter() - t0
        print(f"row_wise_extremality_index    n={n:>7} n_cols={n_cols:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms ({wall * 1000 / n_calls:9.2f} ms/call)")

    for n, n_cols, n_calls in [(2000, 20, 20), (200000, 20, 10), (200000, 200, 5)]:
        t0 = time.perf_counter()
        _run_top_k(n, n_cols, n_calls)
        wall = time.perf_counter() - t0
        print(f"row_wise_top_k_extreme_columns n={n:>7} n_cols={n_cols:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms ({wall * 1000 / n_calls:9.2f} ms/call)")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_top_k(200000, 200, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
