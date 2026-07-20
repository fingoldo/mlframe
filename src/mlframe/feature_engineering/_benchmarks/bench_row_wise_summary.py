"""cProfile harness for ``feature_engineering.row_wise_summary_stats``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_row_wise_summary``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering import row_wise_summary_stats


def _make_dataset(n: int, d: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])


def _make_dataset_with_nan(n: int, d: int, seed: int, nan_rate: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n, d))
    arr[rng.random((n, d)) < nan_rate] = np.nan
    return pd.DataFrame(arr, columns=[f"f{i}" for i in range(d)])


def _run(n: int, d: int) -> None:
    X = _make_dataset(n, d, seed=0)
    row_wise_summary_stats(X, stats=("mean", "std", "q10", "q50", "q90"))


def _run_with_nan(n: int, d: int) -> None:
    X = _make_dataset_with_nan(n, d, seed=0)
    row_wise_summary_stats(X, stats=("mean", "std", "median", "q10", "q50", "q90"))


def _make_grouped_dataset(n: int, n_groups: int, d_per_group: int, seed: int) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    rng = np.random.default_rng(seed)
    groups: dict[str, list[str]] = {}
    blocks = []
    for g in range(n_groups):
        cols = [f"g{g}_f{i}" for i in range(d_per_group)]
        groups[f"g{g}"] = cols
        blocks.append(rng.normal(size=(n, d_per_group)))
    all_cols = [c for cols in groups.values() for c in cols]
    X = pd.DataFrame(np.hstack(blocks), columns=all_cols)
    return X, groups


def _run_grouped(n: int, n_groups: int, d_per_group: int) -> None:
    X, groups = _make_grouped_dataset(n, n_groups, d_per_group, seed=0)
    row_wise_summary_stats(X, stats=("mean", "std", "q10", "q50", "q90"), groups=groups)


def _run_grouped_manual(n: int, n_groups: int, d_per_group: int) -> None:
    X, groups = _make_grouped_dataset(n, n_groups, d_per_group, seed=0)
    parts = [row_wise_summary_stats(X, columns=cols, stats=("mean", "std", "q10", "q50", "q90"), column_prefix=f"row_summary_{name}") for name, cols in groups.items()]
    pd.concat(parts, axis=1)


if __name__ == "__main__":
    for n, d in [(10_000, 30), (100_000, 30), (100_000, 200)]:
        t0 = time.perf_counter()
        _run(n, d)
        wall = time.perf_counter() - t0
        print(f"n={n:>8,} d={d:>4} -> {wall * 1000:9.2f} ms")

    print("\nNaN-present path (njit per-row nanquantile kernel vs np.nanquantile apply_along_axis):")
    for n, d in [(200_000, 30)]:
        _run_with_nan(1_000, d)  # warm the njit kernel outside the timed block
        t0 = time.perf_counter()
        _run_with_nan(n, d)
        wall = time.perf_counter() - t0
        print(f"n={n:>8,} d={d:>4} (5% NaN) -> {wall * 1000:9.2f} ms")

    print("\ngrouped path (groups=) vs manual per-group call + concat:")
    for n, n_groups, d_per_group in [(10_000, 5, 6), (100_000, 10, 20)]:
        t0 = time.perf_counter()
        _run_grouped(n, n_groups, d_per_group)
        wall_grouped = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_grouped_manual(n, n_groups, d_per_group)
        wall_manual = time.perf_counter() - t0

        print(
            f"n={n:>8,} groups={n_groups:>2} d/group={d_per_group:>3} -> "
            f"grouped={wall_grouped * 1000:9.2f} ms, manual={wall_manual * 1000:9.2f} ms, "
            f"speedup={wall_manual / wall_grouped:5.2f}x"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
