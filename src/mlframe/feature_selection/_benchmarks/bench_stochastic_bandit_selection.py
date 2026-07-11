"""cProfile harness for ``feature_selection.stochastic_bandit_selection.stochastic_bandit_selection``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_stochastic_bandit_selection``
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

from mlframe.feature_selection.stochastic_bandit_selection import stochastic_bandit_selection
from mlframe.feature_selection.stochastic_bandit_selection_ensemble import stochastic_bandit_selection_ensemble


def _make_data(n: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=[f"f{i}" for i in range(n_features)])
    y = (X.iloc[:, :3].sum(axis=1) + rng.normal(scale=0.5, size=n)).to_numpy()
    return X, y


def _run(n: int, n_features: int, n_epochs: int) -> None:
    X, y = _make_data(n, n_features)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    stochastic_bandit_selection(Ridge(alpha=0.1), X, y, scoring=r2_score, subset_size=8, n_epochs=n_epochs, cv=cv, random_state=0)


def _run_ensemble(n: int, n_features: int, n_epochs: int, n_seeds: int) -> None:
    X, y = _make_data(n, n_features)
    cv = KFold(n_splits=3, shuffle=True, random_state=0)
    stochastic_bandit_selection_ensemble(
        Ridge(alpha=0.1), X, y, scoring=r2_score, subset_size=8, seeds=list(range(n_seeds)), n_epochs=n_epochs, cv=cv
    )


if __name__ == "__main__":
    for n, n_features, n_epochs in [(100, 30, 60), (100, 30, 150), (300, 30, 150)]:
        t0 = time.perf_counter()
        _run(n, n_features, n_epochs)
        wall = time.perf_counter() - t0
        print(f"n={n:>5} n_features={n_features:>3} n_epochs={n_epochs:>4} -> {wall * 1000:9.2f} ms")

    for n, n_features, n_epochs, n_seeds in [(100, 30, 60, 5), (100, 30, 150, 8)]:
        t0 = time.perf_counter()
        _run_ensemble(n, n_features, n_epochs, n_seeds)
        wall = time.perf_counter() - t0
        print(f"[ensemble] n={n:>5} n_features={n_features:>3} n_epochs={n_epochs:>4} n_seeds={n_seeds:>2} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(300, 30, 150)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_ensemble(300, 30, 150, 8)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[ensemble profile]")
    print(buf.getvalue())
