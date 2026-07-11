"""cProfile harness for ``training.composite.feature_subset_bagging.FeatureSubsetBaggingEnsemble``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_feature_subset_bagging``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.training.composite.feature_subset_bagging import FeatureSubsetBaggingEnsemble


def _make_dataset(n: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = X[:, :5].sum(axis=1) + rng.normal(scale=0.5, size=n)
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)]), y


def _run(n: int, n_features: int, n_subsets: int, aggregation: str = "mean") -> None:
    df, y = _make_dataset(n, n_features, seed=0)
    ens = FeatureSubsetBaggingEnsemble(lambda: Ridge(alpha=0.1), n_subsets=n_subsets, subset_size=10, n_clusters=10, random_state=0, aggregation=aggregation)
    ens.fit(df, y)
    ens.predict(df)


if __name__ == "__main__":
    for n, n_features, n_subsets in [(500, 50, 10), (2000, 50, 10), (2000, 200, 20)]:
        t0 = time.perf_counter()
        _run(n, n_features, n_subsets)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} n_features={n_features:>4} n_subsets={n_subsets:>3} aggregation=mean     -> {wall * 1000:9.2f} ms")

        t0 = time.perf_counter()
        _run(n, n_features, n_subsets, aggregation="weighted")
        wall = time.perf_counter() - t0
        print(f"n={n:>5} n_features={n_features:>4} n_subsets={n_subsets:>3} aggregation=weighted -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 200, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 200, 20, aggregation="weighted")
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
