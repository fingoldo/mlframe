"""cProfile harness for ``feature_selection.unanimous_permutation_prune.unanimous_permutation_prune``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_unanimous_permutation_prune``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from mlframe.feature_selection.unanimous_permutation_prune import unanimous_permutation_prune


def _make_dataset(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    w = np.zeros(d)
    w[:5] = rng.normal(size=5)
    y = X.to_numpy() @ w + rng.normal(scale=0.5, size=n)
    return X, y


def _run(n: int, d: int, n_repeats: int) -> None:
    X, y = _make_dataset(n, d, seed=0)
    cv_splits = list(TimeSeriesSplit(n_splits=4).split(X))
    unanimous_permutation_prune(X, y, lambda: Ridge(alpha=1.0), cv_splits, n_repeats=n_repeats, max_iterations=2)


if __name__ == "__main__":
    for n, d, n_repeats in [(500, 10, 5), (2000, 20, 5), (2000, 20, 15)]:
        t0 = time.perf_counter()
        _run(n, d, n_repeats)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} d={d:>3} n_repeats={n_repeats:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 20, 15)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
