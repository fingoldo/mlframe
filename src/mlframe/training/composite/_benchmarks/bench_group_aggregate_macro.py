"""cProfile harness for ``training.composite.predicted_group_aggregate_feature``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_group_aggregate_macro``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import predicted_group_aggregate_feature


def _make_dataset(n_groups: int, n_per_group: int, seed: int):
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_groups), n_per_group)
    macro = rng.normal(scale=2.0, size=n_groups)
    n = n_groups * n_per_group
    x_row = macro[group_ids] + rng.normal(scale=3.0, size=n)
    y_row = macro[group_ids] + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x": x_row}), y_row, group_ids


def _run(n_groups: int, n_per_group: int) -> None:
    X, y, group_ids = _make_dataset(n_groups, n_per_group, seed=0)
    predicted_group_aggregate_feature(X, y, group_ids, macro_estimator_factory=lambda: LinearRegression(), n_splits=5)


if __name__ == "__main__":
    for n_groups, n_per_group in [(100, 20), (1_000, 20), (5_000, 20)]:
        t0 = time.perf_counter()
        _run(n_groups, n_per_group)
        wall = time.perf_counter() - t0
        print(f"n_groups={n_groups:>6,} n_per_group={n_per_group:>3} (n_rows={n_groups*n_per_group:>7,}) -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
