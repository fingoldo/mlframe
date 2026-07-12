"""cProfile benchmark for ``mlframe.competition.leak_scan``.

COMPETITION/EXPLORATORY USE ONLY -- see ``mlframe.competition`` package docstring.

Run directly: ``python -m mlframe.competition._benchmarks.bench_leak_scan``
"""
from __future__ import annotations

import cProfile
import pstats
import time

import numpy as np
import pandas as pd

from mlframe.competition.leak_scan import find_shifted_column_groups, sort_by_density_leak_scan


def _make_sparse_frame(rng: np.random.Generator, n_rows: int, n_cols: int, density: float) -> pd.DataFrame:
    values = rng.uniform(1, 100, size=(n_rows, n_cols))
    mask = rng.random(size=(n_rows, n_cols)) < density
    values[~mask] = np.nan
    return pd.DataFrame(values, columns=[f"c{i}" for i in range(n_cols)])


def _run_once() -> None:
    rng = np.random.default_rng(42)

    for n_rows, n_cols in [(500, 50), (2_000, 100), (5_000, 150)]:
        df = _make_sparse_frame(rng, n_rows=n_rows, n_cols=n_cols, density=0.2)
        sort_by_density_leak_scan(df)

    small_df = _make_sparse_frame(rng, n_rows=300, n_cols=25, density=0.5)
    find_shifted_column_groups(small_df, max_lag=2)


def main() -> None:
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    _run_once()
    profiler.disable()
    wall = time.perf_counter() - t0

    stats = pstats.Stats(profiler).sort_stats("cumulative")
    print(f"wall time: {wall:.4f}s")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
