"""cProfile harness for ``feature_engineering.transformer.compute_neighbor_aggregate_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_neighbor_aggregate_features``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import compute_neighbor_aggregate_features


def _make_dataset(n: int, n_clusters: int, seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5, size=(n_clusters, 2))
    cluster_means = rng.uniform(10, 100, n_clusters)
    cluster_id = rng.integers(0, n_clusters, n)
    X = centers[cluster_id] + rng.normal(scale=0.5, size=(n, 2))
    y = cluster_means[cluster_id] + rng.normal(scale=2, size=n)
    return X, y


def _run(n: int) -> None:
    X, y = _make_dataset(n, n_clusters=max(5, n // 100), seed=0)
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    compute_neighbor_aggregate_features(X, {"y": y}, X_query=None, splitter=splitter, seed=0, k_values=(10, 20, 40))


if __name__ == "__main__":
    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
