"""cProfile harness for ``preprocessing.select_column_transforms``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_auto_transform_select``

Cost is dominated by ``n_columns * n_candidate_transforms * n_splits`` probe-model fits -- an offline
preprocessing-decision tool run once per dataset, not a hot inner-loop kernel.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.auto_transform_select import select_column_transforms


def _make_data(n_rows: int, n_cols: int, seed: int):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n_rows)
    y = (z + rng.normal(0, 0.5, n_rows) > 0).astype(int)
    cols = {f"col_{i}": z * rng.uniform(0.5, 2.0) + rng.normal(0, 1, n_rows) for i in range(n_cols)}
    return pd.DataFrame(cols), y


def _run(n_rows: int, n_cols: int) -> None:
    df, y = _make_data(n_rows, n_cols, seed=0)
    select_column_transforms(df, y, task="classification", n_splits=3, random_state=0)


if __name__ == "__main__":
    for n_rows, n_cols in [(2_000, 5), (20_000, 10)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7,} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
