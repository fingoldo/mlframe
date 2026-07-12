"""cProfile harness for ``inference.time_budget_ensemble.TimeBudgetEnsemble``.

Run: ``python -m mlframe.inference._benchmarks.bench_time_budget_ensemble``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.inference.time_budget_ensemble import TimeBudgetEnsemble


class _FastModel:
    def predict(self, X):
        return np.zeros(len(X))


def _run(n_models: int, n_rows: int, n_calls: int) -> None:
    models = [_FastModel() for _ in range(n_models)]
    ensemble = TimeBudgetEnsemble(models, time_budget_seconds=0.05)
    X = np.zeros((n_rows, 10))
    for _ in range(n_calls):
        ensemble.predict(X)


def _run_value_per_ms(n_models: int, n_rows: int, n_calls: int) -> None:
    """Profiles the opt-in value_per_ms path: score computation plus the constructor-time re-sort, in
    addition to the same predict loop as ``_run``."""
    models = [_FastModel() for _ in range(n_models)]
    rng = np.random.default_rng(0)
    metric_lift = rng.uniform(0.01, 5.0, size=n_models)
    latency_seconds = rng.uniform(0.0001, 0.01, size=n_models)
    X = np.zeros((n_rows, 10))
    for _ in range(n_calls):
        scores = TimeBudgetEnsemble.compute_value_per_ms(metric_lift.tolist(), latency_seconds.tolist())
        ensemble = TimeBudgetEnsemble(models, time_budget_seconds=0.05, value_per_ms=scores)
        ensemble.predict(X)


if __name__ == "__main__":
    for n_models, n_rows in [(5, 1000), (20, 1000), (20, 100000)]:
        t0 = time.perf_counter()
        _run(n_models, n_rows, n_calls=200)
        wall = time.perf_counter() - t0
        print(f"n_models={n_models:>3} n_rows={n_rows:>7} -> {wall * 1000:9.2f} ms / 200 calls")

    for n_models, n_rows in [(5, 1000), (20, 1000), (20, 100000)]:
        t0 = time.perf_counter()
        _run_value_per_ms(n_models, n_rows, n_calls=200)
        wall = time.perf_counter() - t0
        print(f"[value_per_ms] n_models={n_models:>3} n_rows={n_rows:>7} -> {wall * 1000:9.2f} ms / 200 calls")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20, 1000, n_calls=500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_value_per_ms(20, 1000, n_calls=500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
