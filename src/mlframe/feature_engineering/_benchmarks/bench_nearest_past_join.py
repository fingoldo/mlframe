"""cProfile harness for ``feature_engineering.nearest_past_join``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_nearest_past_join``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.nearest_past_join import nearest_past_join


def _make_data(n_entities: int, history_per_entity: int, seed: int):
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), history_per_entity)
    t = np.tile(np.arange(history_per_entity), n_entities)
    val = rng.normal(0, 1, n_entities * history_per_entity)
    right = pd.DataFrame({"entity": entity_ids, "t": t, "val": val})

    query_t = rng.integers(0, history_per_entity, n_entities) + 0.5
    left = pd.DataFrame({"entity": np.arange(n_entities), "t": query_t})
    return left, right


def _run(n_entities: int, history_per_entity: int) -> None:
    left, right = _make_data(n_entities, history_per_entity, seed=0)
    nearest_past_join(left, right, on="t", by=["entity"], right_value_cols=["val"])


def _make_fallback_chain_data(n_entities: int, history_per_entity: int, seed: int):
    """Sparse fine key (entity+tod+wd, mostly unmatched) so every row exercises the fallback chain."""
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), history_per_entity)
    tod = rng.integers(0, 4, n_entities * history_per_entity)
    wd = rng.integers(0, 7, n_entities * history_per_entity)
    t = np.tile(np.arange(history_per_entity), n_entities)
    val = rng.normal(0, 1, n_entities * history_per_entity)
    right = pd.DataFrame({"entity": entity_ids, "tod": tod, "wd": wd, "t": t, "val": val})

    query_t = rng.integers(0, history_per_entity, n_entities) + 0.5
    query_tod = rng.integers(0, 4, n_entities)
    query_wd = rng.integers(0, 7, n_entities)
    left = pd.DataFrame({"entity": np.arange(n_entities), "tod": query_tod, "wd": query_wd, "t": query_t})
    return left, right


def _run_fallback_chain(n_entities: int, history_per_entity: int) -> None:
    left, right = _make_fallback_chain_data(n_entities, history_per_entity, seed=0)
    nearest_past_join(
        left,
        right,
        on="t",
        by=["entity", "tod", "wd"],
        right_value_cols=["val"],
        fallback_by_chain=[["entity", "tod"], ["entity"]],
        tier_col="tier",
    )


if __name__ == "__main__":
    for n_entities, history_per_entity in [(5_000, 10), (100_000, 10)]:
        t0 = time.perf_counter()
        _run(n_entities, history_per_entity)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>9,} history/entity={history_per_entity:>3} -> {wall * 1000:9.2f} ms")

    for n_entities, history_per_entity in [(5_000, 10), (100_000, 10)]:
        t0 = time.perf_counter()
        _run_fallback_chain(n_entities, history_per_entity)
        wall = time.perf_counter() - t0
        print(f"[fallback_by_chain] n_entities={n_entities:>9,} history/entity={history_per_entity:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100_000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_fallback_chain(100_000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    print("[fallback_by_chain path]")
    stats.print_stats(15)
    print(buf.getvalue())
