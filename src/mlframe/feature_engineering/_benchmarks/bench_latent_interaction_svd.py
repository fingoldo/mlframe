"""cProfile harness for ``feature_engineering.latent_interaction_features``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_latent_interaction_svd``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.latent_interaction_svd import latent_interaction_features


def _make_data(n_entities: int, n_items: int, n_events: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"entity": rng.integers(0, n_entities, n_events), "item": rng.integers(0, n_items, n_events)})


def _run(n_entities: int, n_items: int, n_events: int) -> None:
    events_df = _make_data(n_entities, n_items, n_events, seed=0)
    latent_interaction_features(events_df, "entity", "item", n_components=10)


if __name__ == "__main__":
    for n_entities, n_items, n_events in [(2_000, 1_000, 50_000), (20_000, 5_000, 500_000)]:
        t0 = time.perf_counter()
        _run(n_entities, n_items, n_events)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7,} n_items={n_items:>6,} n_events={n_events:>8,} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(2_000, 1_000, 50_000)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
