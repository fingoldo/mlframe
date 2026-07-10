"""cProfile harness for ``feature_engineering.windowed_edge_diff.windowed_edge_aggregate_diff``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_windowed_edge_diff``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.windowed_edge_diff import windowed_edge_aggregate_diff


def _make_dataset(n_entities: int, avg_rows_per_entity: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity_ids = rng.integers(0, n_entities, n_entities * avg_rows_per_entity)
    n = len(entity_ids)
    return pd.DataFrame({"entity": entity_ids, "t": rng.uniform(0, 100, n), "value": rng.normal(size=n)})


def _run(n_entities: int, avg_rows_per_entity: int) -> None:
    df = _make_dataset(n_entities, avg_rows_per_entity, seed=0)
    windowed_edge_aggregate_diff(df, "entity", "t", "value", n=3, agg="mean")


if __name__ == "__main__":
    for n_entities, avg_rows in [(500, 10), (2000, 10), (2000, 50)]:
        t0 = time.perf_counter()
        _run(n_entities, avg_rows)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>5} avg_rows/entity={avg_rows:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
