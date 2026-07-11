"""cProfile harness for ``training.composite.SegmentRoutedEstimator``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_segment_routed``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.linear_model import Ridge

from mlframe.training.composite import SegmentRoutedEstimator


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    n_reliable = 2
    n_noise = 30
    reliable = rng.normal(size=(n, n_reliable))
    noise_cols = rng.normal(size=(n, n_noise))
    hist_len = rng.integers(1, 20, n)
    X = np.concatenate([reliable, noise_cols, hist_len.reshape(-1, 1)], axis=1)
    is_sparse = hist_len <= 2
    y = np.zeros(n)
    y[~is_sparse] = reliable[~is_sparse, 0] + noise_cols[~is_sparse].sum(axis=1) + rng.normal(scale=0.3, size=(~is_sparse).sum())
    y[is_sparse] = reliable[is_sparse, 0] + rng.normal(scale=0.3, size=is_sparse.sum())
    return X, y


def _segment_predicate(X: np.ndarray) -> np.ndarray:
    return X[:, -1] <= 2


def _run(n: int) -> None:
    X, y = _make_dataset(n, seed=0)
    est = SegmentRoutedEstimator(main_estimator=Ridge(alpha=1.0), specialist_estimator=Ridge(alpha=1.0), segment_predicate=_segment_predicate, specialist_features=[0, 1])
    est.fit(X, y)
    est.predict(X)


def _run_auto(n: int) -> None:
    X, y = _make_dataset(n, seed=0)
    est = SegmentRoutedEstimator(
        main_estimator=Ridge(alpha=1.0),
        specialist_estimator=Ridge(alpha=1.0),
        auto_segment_column=-1,
        auto_segment_quantile=0.1,
        auto_segment_direction="low",
        specialist_features=[0, 1],
    )
    est.fit(X, y)
    est.predict(X)


if __name__ == "__main__":
    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"manual predicate: n={n:>7,} -> {wall * 1000:9.2f} ms")

    for n in [1_000, 10_000, 100_000]:
        t0 = time.perf_counter()
        _run_auto(n)
        wall = time.perf_counter() - t0
        print(f"auto_segment_column: n={n:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_auto = cProfile.Profile()
    profiler_auto.enable()
    _run_auto(100_000)
    profiler_auto.disable()
    buf_auto = StringIO()
    stats_auto = pstats.Stats(profiler_auto, stream=buf_auto).sort_stats("cumulative")
    stats_auto.print_stats(15)
    print(buf_auto.getvalue())
