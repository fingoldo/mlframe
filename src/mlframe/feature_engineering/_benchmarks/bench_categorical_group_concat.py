"""cProfile harness for ``feature_engineering.categorical_group_concat.concat_categorical_group``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_categorical_group_concat``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.categorical_group_concat import concat_categorical_group


def _make_dataset(n_rows: int, n_categories: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.integers(0, n_categories, n_rows).astype(str),
            "b": rng.integers(0, n_categories, n_rows).astype(str),
            "c": rng.integers(0, n_categories, n_rows).astype(str),
        }
    )


def _run(n_rows: int, n_categories: int) -> None:
    df = _make_dataset(n_rows, n_categories, seed=0)
    concat_categorical_group(df, ["a", "b", "c"])


if __name__ == "__main__":
    for n_rows, n_categories in [(50000, 20), (1000000, 20), (1000000, 200)]:
        t0 = time.perf_counter()
        _run(n_rows, n_categories)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>8} n_categories={n_categories:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
