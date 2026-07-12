"""cProfile harness for ``feature_engineering.multi_window_aggregate``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_multi_window_aggregate``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.multi_window_aggregate import multi_window_aggregate


def _make_data(n_entities: int, history_per_entity: int, seed: int):
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), history_per_entity)
    t = np.tile(np.arange(history_per_entity), n_entities).astype(np.float64)
    amount = rng.normal(0, 1, n_entities * history_per_entity)
    history_df = pd.DataFrame({"entity": entity_ids, "t": t, "amount": amount})
    query_df = pd.DataFrame({"entity": np.arange(n_entities), "as_of": [history_per_entity // 2] * n_entities})
    return history_df, query_df


def _run(n_entities: int, history_per_entity: int) -> None:
    history_df, query_df = _make_data(n_entities, history_per_entity, seed=0)
    multi_window_aggregate(
        history_df, entity_col="entity", time_col="t", as_of=query_df,
        agg_funcs={"amount": ["sum", "count", "mean"]}, lookback_horizons=[3, 10, 30],
    )


def _run_auto_select(n_entities: int, history_per_entity: int) -> None:
    history_df, query_df = _make_data(n_entities, history_per_entity, seed=0)
    # a synthetic label correlated with the recent (short-horizon) mean, so the CV-lift scan has real
    # signal to find and doesn't just spend the whole budget on a hopeless search.
    rng = np.random.default_rng(1)
    y = (rng.random(n_entities) < 0.5).astype(int)
    multi_window_aggregate(
        history_df, entity_col="entity", time_col="t", as_of=query_df,
        agg_funcs={"amount": ["sum", "count", "mean"]}, lookback_horizons=[3, 10, 30],
        auto_select=True, target=y, cv=3,
    )


if __name__ == "__main__":
    for n_entities, history_per_entity in [(1_000, 10), (5_000, 10)]:
        t0 = time.perf_counter()
        _run(n_entities, history_per_entity)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7,} history/entity={history_per_entity:>3} -> {wall * 1000:9.2f} ms")

    for n_entities, history_per_entity in [(1_000, 10), (5_000, 10)]:
        t0 = time.perf_counter()
        _run_auto_select(n_entities, history_per_entity)
        wall = time.perf_counter() - t0
        print(f"[auto_select] n_entities={n_entities:>7,} history/entity={history_per_entity:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1_000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_auto_select(1_000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    print("--- auto_select profile ---")
    stats.print_stats(15)
    print(buf.getvalue())
