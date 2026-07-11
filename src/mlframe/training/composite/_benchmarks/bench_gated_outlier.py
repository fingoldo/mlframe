"""cProfile harness for ``training.composite.GatedOutlierEstimator``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_gated_outlier``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.training.composite import GatedOutlierEstimator


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    is_purchase = rng.random(n) < (0.1 + 0.8 * (X[:, 0] > 0))
    y = np.zeros(n)
    y[is_purchase] = np.clip(100 + 20 * X[is_purchase, 1] + rng.normal(0, 5, is_purchase.sum()), 1, None)
    return X, y


def _run(n: int) -> None:
    X, y = _make_dataset(n, seed=0)
    est = GatedOutlierEstimator(regressor=LinearRegression(), classifier=LogisticRegression(max_iter=1000))
    est.fit(X, y)
    est.predict(X)


def _run_calibrated(n: int) -> None:
    X, y = _make_dataset(n, seed=0)
    est = GatedOutlierEstimator(
        regressor=LinearRegression(),
        classifier=RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0),
        calibrate_classifier=True,
        calibration_cv=3,
    )
    est.fit(X, y)
    est.predict(X)


if __name__ == "__main__":
    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms")

    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        _run_calibrated(n)
        wall = time.perf_counter() - t0
        print(f"calibrated n={n:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_calibrated(100_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
