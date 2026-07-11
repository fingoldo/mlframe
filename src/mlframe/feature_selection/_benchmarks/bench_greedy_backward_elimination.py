"""cProfile harness for ``feature_selection.greedy_backward_elimination.greedy_backward_elimination``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_greedy_backward_elimination``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination


def _make_data(n: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = (X.iloc[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n)).to_numpy()
    return X, y


def _run(n: int, n_features: int) -> None:
    X, y = _make_data(n, n_features)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    greedy_backward_elimination(Ridge(alpha=0.1), X, y, scoring=r2_score, cv=cv, min_features=max(1, n_features - 5))


if __name__ == "__main__":
    for n, n_features in [(200, 10), (200, 20), (500, 20)]:
        t0 = time.perf_counter()
        _run(n, n_features)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
