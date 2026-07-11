"""cProfile harness for ``preprocessing.gaussian_power_transform_search.gaussian_power_transform_search``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_gaussian_power_transform_search``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.gaussian_power_transform_search import gaussian_power_transform_search


def _make_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {f"f{c}": np.exp(rng.normal(size=n_rows)) for c in range(n_cols)}
    return pd.DataFrame(data)


def _run(n_rows: int, n_cols: int) -> None:
    df = _make_dataset(n_rows, n_cols, seed=0)
    gaussian_power_transform_search(df)


if __name__ == "__main__":
    for n_rows, n_cols in [(5000, 10), (50000, 10), (50000, 50)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>6} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
