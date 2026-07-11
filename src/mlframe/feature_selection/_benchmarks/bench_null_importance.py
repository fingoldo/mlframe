"""cProfile harness for ``feature_selection.filters.null_importance_filter``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_null_importance``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection.filters._null_importance import null_importance_filter


def _importance_fn(X, y):
    model = RandomForestRegressor(n_estimators=20, max_depth=5, n_jobs=-1, random_state=0)
    model.fit(X, y)
    return model.feature_importances_


def _run(n: int, n_features: int, n_shuffles: int, return_margin_score: bool = False) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, n_features))
    y = X[:, 0] + 0.3 * rng.standard_normal(n)
    null_importance_filter(X, y, _importance_fn, n_shuffles=n_shuffles, random_state=0, return_margin_score=return_margin_score)


if __name__ == "__main__":
    t0 = time.perf_counter()
    _run(1000, 20, 20)
    wall = time.perf_counter() - t0
    print(f"n=1000 features=20 shuffles=20 -> {wall * 1000:.2f} ms total ({wall / 21 * 1000:.2f} ms/fit)")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000, 20, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(12)
    print(buf.getvalue())

    # opt-in margin_score path: the extra work is O(n_features * n_shuffles) numpy reductions on top of the
    # same n_shuffles+1 fits, so it should add negligible wall time relative to the refit loop above.
    t0 = time.perf_counter()
    _run(1000, 20, 20, return_margin_score=True)
    wall_margin = time.perf_counter() - t0
    print(f"n=1000 features=20 shuffles=20 return_margin_score=True -> {wall_margin * 1000:.2f} ms total (overhead vs baseline: {(wall_margin - wall) * 1000:.2f} ms)")

    profiler_margin = cProfile.Profile()
    profiler_margin.enable()
    _run(1000, 20, 20, return_margin_score=True)
    profiler_margin.disable()
    buf_margin = StringIO()
    stats_margin = pstats.Stats(profiler_margin, stream=buf_margin).sort_stats("cumulative")
    stats_margin.print_stats(12)
    print(buf_margin.getvalue())
