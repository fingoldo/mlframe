"""cProfile harness for ``feature_engineering.leakage_safe_aggregate``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_as_of_aggregate``

Optimization history at n_entities=20,000 x 10 history rows: initial per-query pandas .mean()/.sum() slicing
cost 8.7s; a cumsum+searchsorted rewrite for the sum/mean/count fast path cut this to 5.3s (1.6x). The
remaining cost is the outer `groupby(entity_col).__iter__` Python loop, bounded by n_entities (not
n_history_rows) -- one query per entity here means 20,000 loop iterations regardless of the inner vectorization.
Closing that gap needs a fully cross-entity-vectorized rewrite (single global sort + per-entity boundary
offsets via searchsorted on entity id, no Python-level per-entity loop at all) -- left as a documented,
deliberate stopping point rather than a full rewrite, since this is a feature-engineering-cadence helper (not
a per-fit hot loop) and the 1.6x win already ships.
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.as_of_aggregate import leakage_safe_aggregate


def _make_data(n_entities: int, history_per_entity: int, seed: int):
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), history_per_entity)
    t = np.tile(np.arange(history_per_entity), n_entities)
    amount = rng.normal(0, 1, n_entities * history_per_entity)
    history_df = pd.DataFrame({"entity": entity_ids, "t": t, "amount": amount})
    query_df = pd.DataFrame({"entity": np.arange(n_entities), "as_of": [history_per_entity // 2] * n_entities})
    return history_df, query_df


def _run(n_entities: int, history_per_entity: int) -> None:
    history_df, query_df = _make_data(n_entities, history_per_entity, seed=0)
    leakage_safe_aggregate(history_df, entity_col="entity", time_col="t", as_of=query_df, agg_funcs={"amount": ["mean", "sum", "count"]})


if __name__ == "__main__":
    for n_entities, history_per_entity in [(2_000, 10), (20_000, 10)]:
        t0 = time.perf_counter()
        _run(n_entities, history_per_entity)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>8,} history/entity={history_per_entity:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 10)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
