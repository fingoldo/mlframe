"""cProfile harness for ``preprocessing.rare_count_pruning``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_rare_count_pruning``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.rare_count_pruning import collapse_rare_categories, drop_rare_features


def _make_dataset(n_rows: int, n_cols: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cat_cols = {f"cat{i}": rng.choice([f"v{j}" for j in range(n_rows // 3)], size=n_rows) for i in range(n_cols // 2)}
    ind_cols = {f"ind{i}": rng.integers(0, 2, n_rows) for i in range(n_cols // 2)}
    return pd.DataFrame({**cat_cols, **ind_cols})


def _run(n_rows: int, n_cols: int) -> None:
    df = _make_dataset(n_rows, n_cols, seed=0)
    cat_cols = [c for c in df.columns if c.startswith("cat")]
    collapse_rare_categories(df, cat_cols, min_count=10)
    drop_rare_features(df, min_total_count=20)


if __name__ == "__main__":
    for n_rows, n_cols in [(5000, 40), (50000, 40), (50000, 200)]:
        t0 = time.perf_counter()
        _run(n_rows, n_cols)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>6} n_cols={n_cols:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
