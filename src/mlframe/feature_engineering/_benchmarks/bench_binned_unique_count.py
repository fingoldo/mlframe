"""cProfile harness for ``feature_engineering.binned_unique_count.binned_unique_count``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_binned_unique_count``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.binned_unique_count import binned_unique_count


def _make_dataset(n_entities: int, avg_rows_per_entity: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity_ids = rng.integers(0, n_entities, n_entities * avg_rows_per_entity)
    n = len(entity_ids)
    return pd.DataFrame({"entity": entity_ids, "value": rng.normal(size=n)})


def _run(n_entities: int, avg_rows_per_entity: int, per_entity_bins: bool = False) -> None:
    df = _make_dataset(n_entities, avg_rows_per_entity, seed=0)
    binned_unique_count(df, "entity", "value", n_bins=10, per_entity_bins=per_entity_bins)


if __name__ == "__main__":
    for n_entities, avg_rows in [(500, 10), (5000, 10), (5000, 50)]:
        t0 = time.perf_counter()
        _run(n_entities, avg_rows)
        wall = time.perf_counter() - t0
        print(f"[global]      n_entities={n_entities:>5} avg_rows/entity={avg_rows:>3} -> {wall * 1000:9.2f} ms")

    for n_entities, avg_rows in [(500, 10), (5000, 10), (5000, 50)]:
        t0 = time.perf_counter()
        _run(n_entities, avg_rows, per_entity_bins=True)
        wall = time.perf_counter() - t0
        print(f"[per_entity]  n_entities={n_entities:>5} avg_rows/entity={avg_rows:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5000, 50)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("[global] cProfile:")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5000, 50, per_entity_bins=True)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("[per_entity] cProfile:")
    print(buf.getvalue())
