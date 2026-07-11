"""cProfile harness for ``preprocessing.outlier_capping_or_missing.outlier_cap_or_missing``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_outlier_capping_or_missing``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.outlier_capping_or_missing import outlier_cap_or_missing


def _make_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {}
    for c in range(n_cols):
        col = rng.normal(size=n_rows)
        n_outliers = int(n_rows * 0.01)
        idx = rng.choice(n_rows, size=n_outliers, replace=False)
        col[idx] = rng.uniform(20, 40, size=n_outliers) * rng.choice([-1, 1], size=n_outliers)
        data[f"f{c}"] = col
    return pd.DataFrame(data)


def _run(n_rows: int, n_cols: int, mode: str = "cap", rule: str = "auto") -> None:
    df = _make_dataset(n_rows, n_cols, seed=0)
    outlier_cap_or_missing(df, mode=mode, rule=rule)


if __name__ == "__main__":
    for n_rows, n_cols in [(50000, 10), (500000, 10), (500000, 50)]:
        for rule in ("auto", "mad"):
            t0 = time.perf_counter()
            _run(n_rows, n_cols, rule=rule)
            wall = time.perf_counter() - t0
            print(f"n_rows={n_rows:>7} n_cols={n_cols:>3} rule={rule:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
