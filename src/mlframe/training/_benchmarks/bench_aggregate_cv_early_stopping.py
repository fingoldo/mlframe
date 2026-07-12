"""cProfile harness for ``training.select_best_iteration_by_aggregate_cv``.

Run: ``python -m mlframe.training._benchmarks.bench_aggregate_cv_early_stopping``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training._aggregate_cv_early_stopping import select_best_iteration_by_aggregate_cv


def _run(n_folds: int, n_rounds: int, n_calls: int, aggregation: str = "mean") -> None:
    rng = np.random.default_rng(0)
    curves = rng.standard_normal((n_folds, n_rounds))
    for _ in range(n_calls):
        select_best_iteration_by_aggregate_cv(curves, aggregation=aggregation)


if __name__ == "__main__":
    for aggregation in ("mean", "trimmed_mean", "median"):
        print(f"--- aggregation={aggregation} ---")
        for n_folds, n_rounds, n_calls in [(10, 1_000, 500), (50, 50_000, 100)]:
            t0 = time.perf_counter()
            _run(n_folds, n_rounds, n_calls, aggregation=aggregation)
            wall = time.perf_counter() - t0
            print(f"folds={n_folds:>3} rounds={n_rounds:>7,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.4f} ms/call")

        profiler = cProfile.Profile()
        profiler.enable()
        _run(50, 50_000, 200, aggregation=aggregation)
        profiler.disable()
        buf = StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(10)
        print(buf.getvalue())
