"""cProfile harness for ``inference.recursive_forecast.recursive_multi_step_forecast``.

Run: ``python -m mlframe.inference._benchmarks.bench_recursive_forecast``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mlframe.inference.recursive_forecast import recursive_multi_step_forecast, diagnose_error_accumulation


def _run(n_series: int, n_steps: int) -> None:
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(5000, 1))
    y_train = 0.8 * X_train[:, 0] + rng.normal(scale=0.5, size=5000)
    model = Ridge(alpha=0.1).fit(X_train, y_train)

    features = pd.DataFrame({"lag_1": rng.normal(size=n_series)})
    recursive_multi_step_forecast(model, features, n_steps, "lag_1", lambda f, p, s: f)


def _run_diagnose(n_series: int, n_steps: int) -> None:
    rng = np.random.default_rng(0)
    series = np.zeros((n_series, n_steps + 1))
    series[:, 0] = rng.normal(scale=5.0, size=n_series)
    for t in range(1, n_steps + 1):
        series[:, t] = 0.9 * series[:, t - 1] + rng.normal(scale=1.0, size=n_series)

    X_train = series[:, :-1].reshape(-1, 1)
    y_train = series[:, 1:].reshape(-1)
    model = Ridge(alpha=0.1).fit(X_train, y_train)

    initial_features = pd.DataFrame({"lag_1": series[:, 0]})
    true_targets = series[:, 1 : n_steps + 1].T
    oracle_lag_values = series[:, :n_steps].T
    diagnose_error_accumulation(
        model, initial_features, n_steps, "lag_1", lambda f, p, s: f, true_targets=true_targets, oracle_lag_values=oracle_lag_values
    )


if __name__ == "__main__":
    for n_series, n_steps in [(5000, 10), (50000, 10), (50000, 50)]:
        t0 = time.perf_counter()
        _run(n_series, n_steps)
        wall = time.perf_counter() - t0
        print(f"n_series={n_series:>6} n_steps={n_steps:>3} -> {wall * 1000:9.2f} ms")

    for n_series, n_steps in [(5000, 10), (50000, 10), (50000, 50)]:
        t0 = time.perf_counter()
        _run_diagnose(n_series, n_steps)
        wall = time.perf_counter() - t0
        print(f"[diagnose_error_accumulation] n_series={n_series:>6} n_steps={n_steps:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_diagnose(50000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("[diagnose_error_accumulation]")
    print(buf.getvalue())
