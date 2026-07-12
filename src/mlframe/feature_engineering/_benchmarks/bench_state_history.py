"""cProfile harness for ``feature_engineering.state_history.last_k_distinct_states_with_durations``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_state_history``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.state_history import last_k_distinct_states_with_durations


def _make_data(n_entities: int, avg_rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    states, groups = [], []
    for e in range(n_entities):
        length = rng.integers(max(2, avg_rows - 5), avg_rows + 5)
        states.append(rng.integers(0, 5, size=length))
        groups.append(np.full(length, e))
    return np.concatenate(states), np.concatenate(groups)


def _run(n_entities: int, avg_rows: int, n_calls: int, k: int = 5, min_dwell_duration: float | None = None) -> None:
    states, groups = _make_data(n_entities, avg_rows)
    for _ in range(n_calls):
        last_k_distinct_states_with_durations(states, groups, k=k, min_dwell_duration=min_dwell_duration)


if __name__ == "__main__":
    # warm numba dispatch cache before timing (both the plain and the collapsing paths).
    _run(100, 20, 1)
    _run(100, 20, 1, min_dwell_duration=2)

    for n_entities, avg_rows, n_calls in [(2000, 30, 20), (50000, 30, 20), (50000, 100, 20)]:
        t0 = time.perf_counter()
        _run(n_entities, avg_rows, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7} avg_rows={avg_rows:>4} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

        t0 = time.perf_counter()
        _run(n_entities, avg_rows, n_calls, min_dwell_duration=2)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7} avg_rows={avg_rows:>4} n_calls={n_calls:>4} min_dwell=2 -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 100, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 100, 20, min_dwell_duration=2)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- min_dwell_duration=2 (collapsing path) ---")
    print(buf.getvalue())
