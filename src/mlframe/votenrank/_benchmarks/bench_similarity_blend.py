"""cProfile harness for ``votenrank.SimilarityBlendEnsemble``.

Run: ``python -m mlframe.votenrank._benchmarks.bench_similarity_blend``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.linear_model import LinearRegression

from mlframe.votenrank import SimilarityBlendEnsemble


def _make_dataset(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=1.0, size=(n, 8))
    y = X @ rng.normal(size=8) + rng.normal(scale=0.2, size=n)
    return X, y


def _run(n: int) -> None:
    X, y = _make_dataset(n, seed=0)
    blend = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=10)
    blend.fit(X, y)
    blend.predict(X)


def _run_n_specialist(n: int, n_regions: int = 4) -> None:
    """Profile the opt-in N-specialist path (fit_multi_region/predict_multi_region) against the same
    total row count as ``_run``, split evenly across regions.
    """
    n_per_region = n // n_regions
    region_X, region_y = [], []
    for i in range(n_regions):
        X, y = _make_dataset(n_per_region, seed=i)
        region_X.append(X)
        region_y.append(y)
    region_estimators = [LinearRegression() for _ in range(n_regions)]
    blend = SimilarityBlendEnsemble(in_dist_estimator=LinearRegression(), out_dist_estimator=LinearRegression(), k=10, region_estimators=region_estimators)
    blend.fit_multi_region(region_X, region_y)
    blend.predict_multi_region(np.vstack(region_X))


if __name__ == "__main__":
    for n in [1_000, 10_000, 60_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"2-specialist   n={n:>7,} -> {wall * 1000:9.2f} ms")

    for n in [1_000, 10_000, 60_000]:
        t0 = time.perf_counter()
        _run_n_specialist(n, n_regions=4)
        wall = time.perf_counter() - t0
        print(f"4-specialist   n={n:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(60_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("2-specialist profile:")
    print(buf.getvalue())

    profiler_n = cProfile.Profile()
    profiler_n.enable()
    _run_n_specialist(60_000, n_regions=4)
    profiler_n.disable()
    buf_n = StringIO()
    stats_n = pstats.Stats(profiler_n, stream=buf_n).sort_stats("cumulative")
    stats_n.print_stats(15)
    print("N-specialist profile:")
    print(buf_n.getvalue())
