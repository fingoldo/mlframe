"""cProfile harness for ``preprocessing.sibling_group_cold_start_fill.sibling_group_cold_start_fill``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_sibling_group_cold_start_fill``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.sibling_group_cold_start_fill import sibling_group_cold_start_fill


def _make_frame(n_groups: int, rows_per_group: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    levels = np.cumsum(rng.normal(scale=0.5, size=n_groups)) + 50
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    order_vals = np.repeat(np.arange(n_groups), rows_per_group)
    values = np.repeat(levels, rows_per_group) + rng.normal(scale=0.3, size=n_groups * rows_per_group)
    cold_start = rng.choice(n_groups, size=max(1, n_groups // 10), replace=False)
    values[np.isin(group_ids, cold_start)] = np.nan
    return pd.DataFrame({"group": group_ids, "order": order_vals, "value": values})


def _run(n_groups: int, rows_per_group: int, n_calls: int, interpolate: bool = False) -> None:
    df = _make_frame(n_groups, rows_per_group)
    for _ in range(n_calls):
        sibling_group_cold_start_fill(df, "group", "order", "value", interpolate=interpolate)


if __name__ == "__main__":
    for n_groups, rows_per_group, n_calls in [(2000, 5, 20), (50000, 5, 20), (50000, 20, 20)]:
        for interpolate in (False, True):
            t0 = time.perf_counter()
            _run(n_groups, rows_per_group, n_calls, interpolate=interpolate)
            wall = time.perf_counter() - t0
            print(f"n_groups={n_groups:>7} rows_per_group={rows_per_group:>3} n_calls={n_calls:>4} interpolate={interpolate!s:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 20, 20, interpolate=True)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
