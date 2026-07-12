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


def _run(n: int, n_features: int, n_repeats: int = 1) -> None:
    X, y = _make_data(n, n_features)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    greedy_backward_elimination(
        Ridge(alpha=0.1), X, y, scoring=r2_score, cv=cv, min_features=max(1, n_features - 5), n_repeats=n_repeats
    )


if __name__ == "__main__":
    for n, n_features in [(200, 10), (200, 20), (500, 20)]:
        t0 = time.perf_counter()
        _run(n, n_features)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} n_features={n_features:>3} n_repeats=1 -> {wall * 1000:9.2f} ms")

    # Opt-in seed-averaged path: n_repeats>1 multiplies CV work per removal decision -- profile it separately
    # so its cost (linear in n_repeats) is visible next to the default single-run path above.
    for n, n_features, n_repeats in [(200, 10, 3), (200, 20, 3)]:
        t0 = time.perf_counter()
        _run(n, n_features, n_repeats=n_repeats)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} n_features={n_features:>3} n_repeats={n_repeats} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_repeated = cProfile.Profile()
    profiler_repeated.enable()
    _run(500, 20, n_repeats=3)
    profiler_repeated.disable()
    buf_repeated = StringIO()
    stats_repeated = pstats.Stats(profiler_repeated, stream=buf_repeated).sort_stats("cumulative")
    stats_repeated.print_stats(15)
    print(buf_repeated.getvalue())
