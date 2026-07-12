"""cProfile harness for ``training.composite.GatedRegressionMixture``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_gated_regression_mixture``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.training.composite import GatedRegressionMixture


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    is_outlier = rng.random(n) < 0.15
    x = rng.normal(size=n)
    severity = np.where(is_outlier, rng.uniform(0, 1, n), 0.0)
    evidence = severity + rng.normal(scale=0.3, size=n)
    y = x * 1.0 + np.where(is_outlier, 5.0 * severity, 0.0) + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"x": x, "evidence": evidence}), y, is_outlier


def _run(n: int, soft_routing: bool = False) -> None:
    X, y, is_outlier = _make_dataset(n, seed=0)
    mixture = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=500), low_regressor=LinearRegression(), high_regressor=LinearRegression(),
        n_splits=5, soft_routing=soft_routing, soft_routing_bandwidth=0.15,
    )
    mixture.fit(X, y, is_outlier)
    mixture.predict(X)


if __name__ == "__main__":
    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms (hard routing)")

        t0 = time.perf_counter()
        _run(n, soft_routing=True)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms (soft routing)")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("-- hard routing --")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, soft_routing=True)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("-- soft routing --")
    print(buf.getvalue())
