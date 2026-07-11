"""cProfile harness for ``feature_engineering.recency_weighted_rolling.recency_weighted_rolling_mean``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_recency_weighted_rolling``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.recency_weighted_rolling import recency_weighted_rolling_mean, recency_weighted_rolling_std


def _make_data(n_entities: int, rows_per_entity: int, seed: int):
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    order = np.tile(np.arange(rows_per_entity), n_entities)
    values = rng.normal(size=n_entities * rows_per_entity)
    return values, entity_ids, order


def _run_mean(n_entities: int, rows_per_entity: int, window: int) -> None:
    values, entity_ids, order = _make_data(n_entities, rows_per_entity, seed=0)
    recency_weighted_rolling_mean(values, entity_ids, window=window, order=order, scheme="exp", param=0.7)


def _run_std(n_entities: int, rows_per_entity: int, window: int) -> None:
    values, entity_ids, order = _make_data(n_entities, rows_per_entity, seed=0)
    recency_weighted_rolling_std(values, entity_ids, window=window, order=order, scheme="exp", param=0.7)


if __name__ == "__main__":
    for label, fn in [("mean", _run_mean), ("std", _run_std)]:
        for n_entities, rows_per_entity, window in [(2000, 50, 10), (20000, 50, 10), (20000, 200, 20)]:
            t0 = time.perf_counter()
            fn(n_entities, rows_per_entity, window)
            wall = time.perf_counter() - t0
            print(f"[{label}] n_entities={n_entities:>6} rows_per_entity={rows_per_entity:>4} window={window:>3} -> {wall * 1000:9.2f} ms")

    for label, fn in [("mean", _run_mean), ("std", _run_std)]:
        profiler = cProfile.Profile()
        profiler.enable()
        fn(20000, 200, 20)
        profiler.disable()
        buf = StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(15)
        print(f"--- cProfile: {label} ---")
        print(buf.getvalue())
