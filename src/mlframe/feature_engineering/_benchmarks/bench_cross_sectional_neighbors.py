"""cProfile harness for ``feature_engineering.compute_cross_sectional_neighbor_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_cross_sectional_neighbors``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering import compute_cross_sectional_neighbor_features


def _make_dataset(n_snapshots: int, rows_per_snap: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_snapshots):
        base = rng.normal(scale=5, size=3)
        for _ in range(rows_per_snap):
            f = base + rng.normal(scale=1.0, size=3)
            rows.append({"snap": s, "f0": f[0], "f1": f[1], "f2": f[2]})
    return pd.DataFrame(rows)


def _run(n_snapshots: int, rows_per_snap: int) -> None:
    df = _make_dataset(n_snapshots, rows_per_snap, seed=0)
    compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1", "f2"], k=10, agg_stats=("mean", "std"))


if __name__ == "__main__":
    for n_snapshots, rows_per_snap in [(500, 10), (2_000, 10), (5_000, 20)]:
        t0 = time.perf_counter()
        _run(n_snapshots, rows_per_snap)
        wall = time.perf_counter() - t0
        print(f"n_snapshots={n_snapshots:>6,} rows_per_snap={rows_per_snap:>3} (n_rows={n_snapshots*rows_per_snap:>8,}) -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
