"""cProfile harness for ``feature_selection.cascade_select`` / ``forward_select``.

Run: ``python -m mlframe.feature_selection._benchmarks.bench_cascade_select``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from mlframe.feature_selection import cascade_select, cascade_select_stable


def _make_dataset(n: int, d_informative: int, d_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    X_info = rng.normal(size=(n, d_informative))
    X_noise = rng.normal(size=(n, d_noise))
    w = rng.normal(size=d_informative)
    y = X_info @ w + rng.normal(scale=0.5, size=n)
    cols = [f"info{i}" for i in range(d_informative)] + [f"noise{i}" for i in range(d_noise)]
    return pd.DataFrame(np.concatenate([X_info, X_noise], axis=1), columns=cols), y


def _run(n: int, d_noise: int) -> None:
    X, y = _make_dataset(n, d_informative=4, d_noise=d_noise, seed=0)
    cascade_select(X, y, lambda: RandomForestRegressor(n_estimators=15, random_state=0), n_boruta_iterations=10, cv=3, scoring="neg_mean_squared_error")


def _run_stable(n: int, d_noise: int, n_bootstrap: int) -> None:
    X, y = _make_dataset(n, d_informative=4, d_noise=d_noise, seed=0)
    cascade_select_stable(
        X,
        y,
        lambda: RandomForestRegressor(n_estimators=15, random_state=0),
        n_bootstrap=n_bootstrap,
        stability_threshold=0.6,
        n_boruta_iterations=10,
        cv=3,
        scoring="neg_mean_squared_error",
    )


if __name__ == "__main__":
    for n, d_noise in [(200, 20), (300, 40), (300, 60)]:
        t0 = time.perf_counter()
        _run(n, d_noise)
        wall = time.perf_counter() - t0
        print(f"n={n:>5,} d_noise={d_noise:>3} -> {wall * 1000:9.2f} ms")

    for n, d_noise, n_bootstrap in [(200, 20, 5), (300, 40, 5)]:
        t0 = time.perf_counter()
        _run_stable(n, d_noise, n_bootstrap)
        wall = time.perf_counter() - t0
        print(f"[stable] n={n:>5,} d_noise={d_noise:>3} B={n_bootstrap:>2} -> {wall * 1000:9.2f} ms ({wall * 1000 / n_bootstrap:8.2f} ms/run)")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(300, 60)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_stable = cProfile.Profile()
    profiler_stable.enable()
    _run_stable(300, 60, n_bootstrap=5)
    profiler_stable.disable()
    buf_stable = StringIO()
    stats_stable = pstats.Stats(profiler_stable, stream=buf_stable).sort_stats("cumulative")
    stats_stable.print_stats(15)
    print(buf_stable.getvalue())
