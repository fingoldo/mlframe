"""cProfile harness for ``preprocessing.augment_temporal_drift``.

Run: ``python -m mlframe.preprocessing._benchmarks.bench_temporal_drift_augment``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.preprocessing.temporal_drift_augment import augment_temporal_drift


def _make_panel(n_entities: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity_ids = np.repeat(np.arange(n_entities), 6)
    t = np.tile(np.arange(6), n_entities)
    x1 = rng.standard_normal(n_entities * 6)
    x2 = rng.standard_normal(n_entities * 6)
    return pd.DataFrame({"entity_id": entity_ids, "t": t, "x1": x1, "x2": x2})


def _run(n_entities: int, weight_by_recency: bool = False) -> None:
    df = _make_panel(n_entities, seed=0)
    augment_temporal_drift(
        df,
        entity_col="entity_id",
        time_col="t",
        feature_cols=["x1", "x2"],
        n_drop_options=(1, 2),
        weight_by_recency=weight_by_recency,
    )


if __name__ == "__main__":
    for weight_by_recency in (False, True):
        label = "weight_by_recency=True" if weight_by_recency else "unweighted (default)"
        print(f"--- {label} ---")
        for n_entities in (2_000, 20_000, 100_000):
            t0 = time.perf_counter()
            _run(n_entities, weight_by_recency=weight_by_recency)
            wall = time.perf_counter() - t0
            print(f"n_entities={n_entities:>8,} (x6 rows) -> {wall * 1000:9.2f} ms")

        profiler = cProfile.Profile()
        profiler.enable()
        _run(100_000, weight_by_recency=weight_by_recency)
        profiler.disable()
        buf = StringIO()
        stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
        stats.print_stats(20)
        print(buf.getvalue())
