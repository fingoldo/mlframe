"""cProfile harness for ``feature_engineering.recency_aggregation.per_group_recency_weighted_agg``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_recency_weighted_agg``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.feature_engineering.recency_aggregation import per_group_recency_weighted_agg


def _make_panel(n_entities: int, hist: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 1.0, size=n_entities * hist)
    group_ids = np.repeat(np.arange(n_entities), hist)
    order = np.tile(np.arange(hist, dtype=np.float64), n_entities)
    return values, group_ids, order


def _run(n_entities: int, hist: int, n_calls: int) -> None:
    values, group_ids, order = _make_panel(n_entities, hist)
    for _ in range(n_calls):
        for agg in ("mean", "sum", "min", "max"):
            per_group_recency_weighted_agg(values, group_ids, agg=agg, order=order, scheme="poly", param=1.0)


if __name__ == "__main__":
    # warm numba dispatch cache before timing.
    _run(100, 8, 1)

    for n_entities, hist, n_calls in [(2000, 12, 20), (50000, 12, 20), (50000, 30, 20)]:
        t0 = time.perf_counter()
        _run(n_entities, hist, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7} hist={hist:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 30, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
