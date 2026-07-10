"""cProfile harness for ``feature_engineering.fuzzy_entity.fuzzy_entity_group_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_fuzzy_entity``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.fuzzy_entity import fuzzy_entity_group_features


def _run(n_entities: int, events_per_entity: int, n_values: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    entity_ids = np.repeat(np.arange(n_entities), events_per_entity)
    n = entity_ids.shape[0]
    values = rng.integers(0, n_values, size=n)
    order = np.arange(n, dtype=np.float64)
    for _ in range(n_calls):
        fuzzy_entity_group_features(entity_ids, values, time_order=order)


if __name__ == "__main__":
    for n_entities, events_per_entity, n_values, n_calls in [(1_000, 10, 50, 20), (50_000, 10, 500, 3)]:
        t0 = time.perf_counter()
        _run(n_entities, events_per_entity, n_values, n_calls)
        wall = time.perf_counter() - t0
        n_rows = n_entities * events_per_entity
        print(f"rows={n_rows:>9,} entities={n_entities:>7,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 10, 500, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
