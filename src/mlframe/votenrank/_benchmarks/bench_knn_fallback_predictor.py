"""cProfile harness for ``votenrank.knn_fallback_predictor.KNNFallbackPredictor``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_knn_fallback_predictor``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.votenrank.knn_fallback_predictor import KNNFallbackPredictor


def _run(n_train: int, n_query: int, n_features: int) -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    y_train = rng.integers(0, 2, n_train).astype(np.float64)
    X_query = rng.normal(size=(n_query, n_features)).astype(np.float32)

    knn = KNNFallbackPredictor(k=15).fit(X_train, y_train)
    knn.predict(X_query)


def _run_blend(n_train: int, n_query: int, n_features: int) -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(n_train, n_features)).astype(np.float32)
    y_train = rng.integers(0, 2, n_train).astype(np.float64)
    X_query = rng.normal(size=(n_query, n_features)).astype(np.float32)
    main_pred = rng.uniform(size=n_query)

    knn = KNNFallbackPredictor(k=15).fit(X_train, y_train)
    knn.predict_blend(X_query, main_pred)


if __name__ == "__main__":
    for n_train, n_query, n_features in [(5000, 2000, 10), (50000, 2000, 10), (50000, 20000, 10)]:
        t0 = time.perf_counter()
        _run(n_train, n_query, n_features)
        wall = time.perf_counter() - t0
        print(f"predict          n_train={n_train:>6} n_query={n_query:>6} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    for n_train, n_query, n_features in [(5000, 2000, 10), (50000, 2000, 10), (50000, 20000, 10)]:
        t0 = time.perf_counter()
        _run_blend(n_train, n_query, n_features)
        wall = time.perf_counter() - t0
        print(f"predict_blend    n_train={n_train:>6} n_query={n_query:>6} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 20000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_blend(50000, 20000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    print("predict_blend profile:")
    stats.print_stats(15)
    print(buf.getvalue())
