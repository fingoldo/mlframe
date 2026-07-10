"""cProfile harness for ``feature_engineering.entity_diff_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_entity_diff_features``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.entity_diff_features import entity_diff_features


def _make_data(n_entities: int, rows_per_entity: int, n_features: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    n = len(entity_ids)
    cols = {f"f{i}": rng.normal(0, 1, n) for i in range(n_features)}
    return pd.DataFrame({"entity": entity_ids, **cols})


def _run(n_entities: int, rows_per_entity: int, n_features: int) -> None:
    df = _make_data(n_entities, rows_per_entity, n_features, seed=0)
    entity_diff_features(df, entity_col="entity")


if __name__ == "__main__":
    for n_entities, rows_per_entity, n_features in [(5_000, 10, 10), (50_000, 10, 20)]:
        t0 = time.perf_counter()
        _run(n_entities, rows_per_entity, n_features)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7,} rows/entity={rows_per_entity:>3} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 10, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
