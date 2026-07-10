"""cProfile harness for ``feature_selection.ridge_forward_prefilter.ridge_coefficient_prefilter``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_ridge_forward_prefilter``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_selection.ridge_forward_prefilter import ridge_coefficient_prefilter


def _make_dataset(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.zeros(d)
    w[:8] = rng.normal(size=8)
    y = X @ w + rng.normal(scale=0.5, size=n)
    return X, y, [f"f{i}" for i in range(d)]


def _run(n: int, d: int) -> None:
    X, y, names = _make_dataset(n, d, seed=0)
    ridge_coefficient_prefilter(X, y, names, cv=3, tol=0.02)


if __name__ == "__main__":
    for n, d in [(500, 200), (500, 2000), (2000, 2000)]:
        t0 = time.perf_counter()
        _run(n, d)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} d={d:>5} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 2000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
