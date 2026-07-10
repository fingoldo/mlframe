"""cProfile harness for ``feature_selection.filters.monotonic_deviation_stability_filter``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_monotonic_stability``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._monotonic_stability import monotonic_deviation_stability_filter


def _make_data(n_groups: int, rows_per_group: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    n = len(group_ids)
    cols = {f"f{i}": rng.normal(0, 1, n) for i in range(n_features)}
    y = rng.integers(0, 2, n)
    df = pd.DataFrame({"group": group_ids, **cols})
    return df, y


def _run(n_groups: int, rows_per_group: int, n_features: int, n_subsamples: int) -> None:
    df, y = _make_data(n_groups, rows_per_group, n_features, seed=0)
    monotonic_deviation_stability_filter(df, y, group_col="group", n_subsamples=n_subsamples, random_state=0)


if __name__ == "__main__":
    for n_groups, rows_per_group, n_features, n_subsamples in [(500, 6, 10, 30), (5_000, 6, 10, 30)]:
        t0 = time.perf_counter()
        _run(n_groups, rows_per_group, n_features, n_subsamples)
        wall = time.perf_counter() - t0
        print(f"n_groups={n_groups:>6,} rows/grp={rows_per_group} n_features={n_features:>3} n_subsamples={n_subsamples:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500, 6, 10, 30)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
