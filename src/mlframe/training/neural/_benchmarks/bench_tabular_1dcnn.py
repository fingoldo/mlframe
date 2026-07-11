"""cProfile harness for ``training.neural.tabular_1dcnn``.

Run: ``python -m mlframe.training.neural._benchmarks.bench_tabular_1dcnn``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.neural.tabular_1dcnn import Tabular1DCNNClassifier, Tabular1DCNNRegressor, correlation_order_features


def _make_data(n: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (X[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n)).astype(np.float32)
    return X, y


def _make_classification_data(n: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    y = (X[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n) > 0).astype(np.int64)
    return X, y


def _run_order(n: int, n_features: int, n_calls: int) -> None:
    X, _ = _make_data(n, n_features)
    for _ in range(n_calls):
        correlation_order_features(X)


def _run_fit(n: int, n_features: int, n_epochs: int) -> None:
    X, y = _make_data(n, n_features)
    Tabular1DCNNRegressor(n_channels=8, kernel_size=3, n_epochs=n_epochs, random_state=0).fit(X, y)


def _run_fit_classifier(n: int, n_features: int, n_epochs: int) -> None:
    X, y = _make_classification_data(n, n_features)
    Tabular1DCNNClassifier(n_channels=8, kernel_size=3, n_epochs=n_epochs, random_state=0).fit(X, y)


if __name__ == "__main__":
    for n, n_features, n_calls in [(500, 30, 20), (2000, 30, 20), (2000, 200, 5)]:
        t0 = time.perf_counter()
        _run_order(n, n_features, n_calls)
        wall = time.perf_counter() - t0
        print(f"correlation_order_features n={n:>5} n_features={n_features:>3} n_calls={n_calls:>3} -> {wall * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run_fit(300, 30, 200)
    print(f"fit n=300 n_features=30 n_epochs=200 -> {(time.perf_counter() - t0) * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run_fit_classifier(300, 30, 200)
    print(f"fit_classifier n=300 n_features=30 n_epochs=200 -> {(time.perf_counter() - t0) * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run_order(2000, 200, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_fit_classifier(2000, 200, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
