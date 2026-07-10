"""cProfile harness for ``feature_selection.filters.boruta_select``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_boruta``

Cost is dominated by ``n_iterations`` importance-function refits over a 2x-width (real + shadow) matrix --
inherent to the Boruta algorithm; an offline feature-selection-decision tool, not a hot loop.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection.filters._boruta import boruta_select


def _importance_fn(X, y):
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=0, n_jobs=1)
    model.fit(X, y)
    return model.feature_importances_


def _make_data(n_rows: int, n_features: int, seed: int):
    rng = np.random.default_rng(seed)
    cols = {f"f{i}": rng.normal(0, 1, n_rows) for i in range(n_features)}
    X = pd.DataFrame(cols)
    y = X["f0"].to_numpy() * 2 + rng.normal(0, 0.5, n_rows)
    return X, y


def _run(n_rows: int, n_features: int, n_iterations: int) -> None:
    X, y = _make_data(n_rows, n_features, seed=0)
    boruta_select(X, y, _importance_fn, n_iterations=n_iterations, random_state=0)


if __name__ == "__main__":
    for n_rows, n_features, n_iterations in [(1_000, 10, 10), (5_000, 20, 20)]:
        t0 = time.perf_counter()
        _run(n_rows, n_features, n_iterations)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>6,} n_features={n_features:>3} n_iterations={n_iterations:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000, 10, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
