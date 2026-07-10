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


if __name__ == "__main__":
    for n in [1_000, 10_000, 60_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n={n:>7,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(60_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
