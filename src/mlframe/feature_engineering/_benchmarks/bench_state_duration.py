"""cProfile harness for ``feature_engineering.state_duration.time_since_state_change``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_state_duration``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.state_duration import time_since_state_change


def _run(n_entities: int, n_periods: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    entity_ids = np.repeat(np.arange(n_entities), n_periods)
    n = entity_ids.shape[0]
    state = rng.random(n) < 0.5
    for _ in range(n_calls):
        time_since_state_change(state, entity_ids)


if __name__ == "__main__":
    _run(50, 10, 1)  # warm the njit kernel before timing
    _run(30_000, 10, 1)

    for n_entities, n_periods, n_calls in [(1_000, 20, 50), (100_000, 20, 5)]:
        t0 = time.perf_counter()
        _run(n_entities, n_periods, n_calls)
        wall = time.perf_counter() - t0
        n_rows = n_entities * n_periods
        print(f"rows={n_rows:>9,} entities={n_entities:>7,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.4f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 20, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
