"""cProfile harness for ``feature_engineering.entity_inter_event.entity_inter_event_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_entity_inter_event``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.entity_inter_event import entity_inter_event_features


def _run(n_entities: int, events_per_entity: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    entity_ids = np.repeat(np.arange(n_entities), events_per_entity)
    n = entity_ids.shape[0]
    timestamps = rng.uniform(0, 1e6, size=n)
    values = rng.standard_normal(n)
    for _ in range(n_calls):
        entity_inter_event_features(entity_ids, timestamps, value_col=values)


if __name__ == "__main__":
    _run(50, 5, 1)  # warm the njit kernel before timing
    _run(5_000, 5, 1)

    for n_entities, events_per_entity, n_calls in [(1_000, 10, 20), (100_000, 10, 5)]:
        t0 = time.perf_counter()
        _run(n_entities, events_per_entity, n_calls)
        wall = time.perf_counter() - t0
        n_rows = n_entities * events_per_entity
        print(f"rows={n_rows:>9,} entities={n_entities:>7,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 10, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
