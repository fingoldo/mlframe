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


def _run_multilag(n_entities: int, rows_per_entity: int, n_features: int, lags: list) -> None:
    df = _make_data(n_entities, rows_per_entity, n_features, seed=0)
    entity_diff_features(df, entity_col="entity", lags=lags)


if __name__ == "__main__":
    for n_entities, rows_per_entity, n_features in [(5_000, 10, 10), (50_000, 10, 20)]:
        t0 = time.perf_counter()
        _run(n_entities, rows_per_entity, n_features)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7,} rows/entity={rows_per_entity:>3} n_features={n_features:>3} -> {wall * 1000:9.2f} ms")

    for n_entities, rows_per_entity, n_features, lags in [(5_000, 10, 10, [1, 2, 5]), (50_000, 10, 20, [1, 2, 5])]:
        t0 = time.perf_counter()
        _run_multilag(n_entities, rows_per_entity, n_features, lags)
        wall = time.perf_counter() - t0
        print(
            f"[multilag={lags}] n_entities={n_entities:>7,} rows/entity={rows_per_entity:>3} "
            f"n_features={n_features:>3} -> {wall * 1000:9.2f} ms"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(5_000, 10, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler_multilag = cProfile.Profile()
    profiler_multilag.enable()
    _run_multilag(5_000, 10, 10, [1, 2, 5])
    profiler_multilag.disable()
    buf_multilag = StringIO()
    stats_multilag = pstats.Stats(profiler_multilag, stream=buf_multilag).sort_stats("cumulative")
    stats_multilag.print_stats(15)
    print(buf_multilag.getvalue())
