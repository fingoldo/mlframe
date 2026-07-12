"""cProfile harness for ``feature_engineering.ewma_multi_alpha_features.ewma_multi_alpha_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_ewma_multi_alpha_features``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.ewma_multi_alpha_features import ewma_multi_alpha_features


def _make_data(n_entities: int, avg_rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    values, groups = [], []
    for e in range(n_entities):
        length = rng.integers(max(2, avg_rows - 5), avg_rows + 5)
        values.append(rng.normal(size=length))
        groups.append(np.full(length, e))
    return np.concatenate(values), np.concatenate(groups)


def _run(n_entities: int, avg_rows: int, n_calls: int) -> None:
    values, groups = _make_data(n_entities, avg_rows)
    for _ in range(n_calls):
        ewma_multi_alpha_features(values, groups, alphas=[0.5, 0.1, 0.05])


def _run_adaptive(n_entities: int, avg_rows: int, n_calls: int) -> None:
    values, groups = _make_data(n_entities, avg_rows)
    for _ in range(n_calls):
        ewma_multi_alpha_features(values, groups, alphas=[0.5, 0.1], adaptive_alpha_grid=[0.05, 0.1, 0.3, 0.5, 0.8])


if __name__ == "__main__":
    _run(100, 20, 1)  # warm numba dispatch cache before timing.
    _run_adaptive(100, 20, 1)  # warm the adaptive kernel too.

    for n_entities, avg_rows, n_calls in [(2000, 30, 20), (50000, 30, 20), (50000, 100, 20)]:
        t0 = time.perf_counter()
        _run(n_entities, avg_rows, n_calls)
        wall = time.perf_counter() - t0
        print(f"fixed-alpha    n_entities={n_entities:>7} avg_rows={avg_rows:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    for n_entities, avg_rows, n_calls in [(2000, 30, 20), (50000, 30, 20), (50000, 100, 20)]:
        t0 = time.perf_counter()
        _run_adaptive(n_entities, avg_rows, n_calls)
        wall = time.perf_counter() - t0
        print(f"adaptive-alpha n_entities={n_entities:>7} avg_rows={avg_rows:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 100, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("fixed-alpha profile:")
    print(buf.getvalue())

    profiler_adaptive = cProfile.Profile()
    profiler_adaptive.enable()
    _run_adaptive(50000, 100, 20)
    profiler_adaptive.disable()
    buf_adaptive = StringIO()
    stats_adaptive = pstats.Stats(profiler_adaptive, stream=buf_adaptive).sort_stats("cumulative")
    stats_adaptive.print_stats(15)
    print("adaptive-alpha profile:")
    print(buf_adaptive.getvalue())
