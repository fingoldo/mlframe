"""cProfile harness for ``feature_engineering.two_step_recency_weighted_target_encode``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_two_step_target_encode``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.two_step_target_encode import two_step_recency_weighted_target_encode


def _make_data(n_entities: int, events_per_entity: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), events_per_entity)
    n = len(entity_ids)
    t = np.tile(np.arange(events_per_entity), n_entities).astype(np.float64)
    cat1 = rng.choice(["A", "B", "C"], size=n)
    y = rng.integers(0, 2, n).astype(np.float64)
    return pd.DataFrame({"entity": entity_ids, "t": t, "cat1": cat1, "y": y})


def _run(n_entities: int, events_per_entity: int, causal: bool = False) -> None:
    events_df = _make_data(n_entities, events_per_entity, seed=0)
    two_step_recency_weighted_target_encode(
        events_df, "entity", ["cat1"], events_df["y"].to_numpy(), "t", decay_half_life=2.0, causal=causal
    )


if __name__ == "__main__":
    for causal in (False, True):
        for n_entities, events_per_entity in [(5_000, 10), (50_000, 10)]:
            t0 = time.perf_counter()
            _run(n_entities, events_per_entity, causal=causal)
            wall = time.perf_counter() - t0
            print(f"causal={causal!s:>5} n_entities={n_entities:>7,} events/entity={events_per_entity:>3} -> {wall * 1000:9.2f} ms")

    for causal in (False, True):
        profiler = cProfile.Profile()
        profiler.enable()
        _run(5_000, 10, causal=causal)
        profiler.disable()
        buf = StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(15)
        print(f"--- causal={causal} ---")
        print(buf.getvalue())
