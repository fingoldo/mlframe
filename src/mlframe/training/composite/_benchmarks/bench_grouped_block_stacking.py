"""cProfile harness for ``training.composite.GroupedBlockStacker``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_grouped_block_stacking``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import GroupedBlockStacker


def _make_dataset(n: int, n_groups: int, cols_per_group: int, seed: int):
    rng = np.random.default_rng(seed)
    n_cols = n_groups * cols_per_group
    group_id = rng.integers(0, n_groups, n)
    X = np.zeros((n, n_cols))
    y = np.zeros(n)
    true_w = {g: rng.normal(size=cols_per_group) for g in range(n_groups)}
    for g in range(n_groups):
        mask = group_id == g
        vals = rng.normal(size=(mask.sum(), cols_per_group))
        X[np.ix_(mask, range(g * cols_per_group, (g + 1) * cols_per_group))] = vals
        y[mask] = vals @ true_w[g] + rng.normal(scale=0.3, size=mask.sum())
    col_names = [f"g{g}_c{c}" for g in range(n_groups) for c in range(cols_per_group)]
    X_df = pd.DataFrame(X, columns=col_names)
    feature_groups = {f"group{g}": [f"g{g}_c{c}" for c in range(cols_per_group)] for g in range(n_groups)}
    return X_df, y, feature_groups


def _run(n: int) -> None:
    X, y, feature_groups = _make_dataset(n, n_groups=6, cols_per_group=6, seed=0)
    stacker = GroupedBlockStacker(feature_groups=feature_groups, submodel_factory=lambda: LinearRegression(), meta_estimator=LinearRegression(), n_splits=5)
    stacker.fit(X, y)
    stacker.predict(X)


def _run_auto_discover(n: int) -> None:
    X, y, _ = _make_dataset(n, n_groups=6, cols_per_group=6, seed=0)
    stacker = GroupedBlockStacker(auto_discover_blocks=True, block_corr_threshold=0.6, submodel_factory=lambda: LinearRegression(), meta_estimator=LinearRegression(), n_splits=5)
    stacker.fit(X, y)
    stacker.predict(X)


if __name__ == "__main__":
    print("-- manual feature_groups --")
    for n in [1_000, 10_000, 50_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms")

    print("-- auto_discover_blocks=True --")
    for n in [1_000, 10_000, 50_000]:
        t0 = time.perf_counter()
        _run_auto_discover(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("-- manual feature_groups, n=50,000 --")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_auto_discover(50_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("-- auto_discover_blocks=True, n=50,000 --")
    print(buf.getvalue())
