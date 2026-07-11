"""cProfile harness for ``training.DirectHorizonBucketForecaster``.

Run: ``python -m mlframe.training._benchmarks.bench_direct_horizon_bucket_forecaster``

Cost is dominated by ``n_groups * n_buckets`` independent model fits, inherent to the direct-per-bucket
design (the whole point is avoiding a single shared recursive model).
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training._direct_horizon_bucket_forecaster import DirectHorizonBucketForecaster


def _make_data(n_groups: int, n_rows_per_group: int, seed: int):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n_groups):
        for _ in range(n_rows_per_group):
            rows.append({"x": rng.normal(0, 1), "grp": g})
    X = pd.DataFrame(rows)
    y = 2.0 * X["x"].to_numpy() + rng.normal(0, 0.1, len(X))
    horizon_day = rng.integers(1, 29, len(X))
    return X, y, horizon_day


def _run(n_groups: int, n_rows_per_group: int, edge_blend_width: int = 0) -> None:
    X, y, horizon_day = _make_data(n_groups, n_rows_per_group, seed=0)
    buckets = [(1, 7), (8, 14), (15, 21), (22, 28)]
    forecaster = DirectHorizonBucketForecaster(buckets, model_factory=lambda: LinearRegression(), group_col="grp")
    forecaster.fit(X, y, horizon_day)
    forecaster.predict(X, horizon_day, edge_blend_width=edge_blend_width)


if __name__ == "__main__":
    for n_groups, n_rows_per_group in [(10, 500), (50, 2_000)]:
        t0 = time.perf_counter()
        _run(n_groups, n_rows_per_group)
        wall = time.perf_counter() - t0
        print(f"n_groups={n_groups:>3} rows/group={n_rows_per_group:>6,} edge_blend_width=0 -> {wall * 1000:9.2f} ms")

        t0 = time.perf_counter()
        _run(n_groups, n_rows_per_group, edge_blend_width=2)
        wall = time.perf_counter() - t0
        print(f"n_groups={n_groups:>3} rows/group={n_rows_per_group:>6,} edge_blend_width=2 -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("hard-boundary predict (edge_blend_width=0):")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10, 500, edge_blend_width=2)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("edge-blended predict (edge_blend_width=2):")
    print(buf.getvalue())
