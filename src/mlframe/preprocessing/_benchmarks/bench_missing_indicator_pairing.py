"""cProfile harness for ``preprocessing.missing_indicator_pairing.impute_with_missing_indicator``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_missing_indicator_pairing``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.missing_indicator_pairing import impute_with_missing_indicator


def _make_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for c in range(n_cols):
        col = rng.normal(size=n_rows)
        mask = rng.random(n_rows) < 0.3
        col[mask] = np.nan
        data[f"f{c}"] = col
    return pd.DataFrame(data)


def _run(n_rows: int, n_cols: int) -> None:
    df = _make_dataset(n_rows, n_cols, seed=0)
    impute_with_missing_indicator(df)


if __name__ == "__main__":
    for n_rows, n_cols in [(50000, 10), (500000, 10), (500000, 50)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>7} n_cols={n_cols:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
