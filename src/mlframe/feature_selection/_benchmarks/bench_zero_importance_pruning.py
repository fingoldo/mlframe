"""cProfile harness for ``feature_selection.zero_importance_pruning.iterative_zero_importance_pruning``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_zero_importance_pruning``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from mlframe.feature_selection.zero_importance_pruning import iterative_zero_importance_pruning


def _make_data(n: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = (X.iloc[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n)).to_numpy()
    return X, y


def _permutation_importance_fn(fitted_estimator, X_round: pd.DataFrame, y_round: np.ndarray) -> np.ndarray:
    result = permutation_importance(fitted_estimator, X_round, y_round, n_repeats=5, random_state=0)
    return np.asarray(result.importances_mean)


def _run(n: int, n_features: int) -> None:
    X, y = _make_data(n, n_features)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    estimator = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=0)
    iterative_zero_importance_pruning(estimator, X, y, scoring=r2_score, cv=cv, importance_threshold=0.005)


def _run_with_permutation_importance_fn(n: int, n_features: int) -> None:
    X, y = _make_data(n, n_features)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    estimator = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=0)
    iterative_zero_importance_pruning(
        estimator, X, y, scoring=r2_score, cv=cv, importance_threshold=0.005, importance_fn=_permutation_importance_fn
    )


if __name__ == "__main__":
    for n, n_features in [(200, 15), (200, 40), (500, 40)]:
        t0 = time.perf_counter()
        _run(n, n_features)
        wall = time.perf_counter() - t0
        print(f"native      n={n:>5} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

        t0 = time.perf_counter()
        _run_with_permutation_importance_fn(n, n_features)
        wall = time.perf_counter() - t0
        print(f"perm_fn     n={n:>5} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(500, 40)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- native feature_importances_ path ---")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_with_permutation_importance_fn(500, 40)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- custom importance_fn (permutation importance) path ---")
    print(buf.getvalue())
