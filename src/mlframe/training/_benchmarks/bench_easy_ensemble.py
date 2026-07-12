"""cProfile harness for ``training.easy_ensemble_fit_predict``.

Run: ``python -m mlframe.training._benchmarks.bench_easy_ensemble``

Cost is dominated by ``n_bags`` independent model fits, inherent to the bagging algorithm.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlframe.training._easy_ensemble import easy_ensemble_fit_predict


def _make_data(n_pos: int, n_neg: int, seed: int):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(1, 1, (n_pos, 5)), rng.normal(-1, 1, (n_neg, 5))])
    y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    return X, y


def _run(n_pos: int, n_neg: int, n_bags: int, bag_feature_subsample=None) -> None:
    X, y = _make_data(n_pos, n_neg, seed=0)
    easy_ensemble_fit_predict(
        X, y, X[:200], model_factory=lambda: LogisticRegression(max_iter=200), n_bags=n_bags, random_state=0, bag_feature_subsample=bag_feature_subsample
    )


if __name__ == "__main__":
    for n_pos, n_neg, n_bags in [(100, 10_000, 10), (500, 50_000, 20)]:
        t0 = time.perf_counter()
        _run(n_pos, n_neg, n_bags)
        wall = time.perf_counter() - t0
        print(f"n_pos={n_pos:>5,} n_neg={n_neg:>7,} n_bags={n_bags:>3} -> {wall * 1000:9.2f} ms")

    for n_pos, n_neg, n_bags in [(100, 10_000, 10), (500, 50_000, 20)]:
        t0 = time.perf_counter()
        _run(n_pos, n_neg, n_bags, bag_feature_subsample=0.5)
        wall = time.perf_counter() - t0
        print(f"[feature-subsample=0.5] n_pos={n_pos:>5,} n_neg={n_neg:>7,} n_bags={n_bags:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100, 10_000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100, 10_000, 10, bag_feature_subsample=0.5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[feature-subsample=0.5]")
    print(buf.getvalue())
