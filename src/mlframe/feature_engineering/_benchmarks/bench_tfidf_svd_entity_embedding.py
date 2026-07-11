"""cProfile harness for ``feature_engineering.tfidf_svd_entity_embedding.tfidf_svd_entity_embedding``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_tfidf_svd_entity_embedding``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.tfidf_svd_entity_embedding import tfidf_svd_entity_embedding


def _make_dataset(n_entities: int, avg_events_per_entity: int, n_categories: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_entities * avg_events_per_entity
    return pd.DataFrame({"entity": rng.integers(0, n_entities, n), "category": rng.integers(0, n_categories, n).astype(str)})


def _run(n_entities: int, avg_events_per_entity: int, n_categories: int) -> None:
    df = _make_dataset(n_entities, avg_events_per_entity, n_categories, seed=0)
    tfidf_svd_entity_embedding(df, "entity", "category", n_components=10)


def _run_transform_new_entities(n_entities: int, avg_events_per_entity: int, n_categories: int) -> None:
    train_df = _make_dataset(n_entities, avg_events_per_entity, n_categories, seed=0)
    _, fitted = tfidf_svd_entity_embedding(train_df, "entity", "category", n_components=10, return_fitted=True)
    new_df = _make_dataset(n_entities, avg_events_per_entity, n_categories, seed=1)
    new_df["entity"] = new_df["entity"] + n_entities  # disjoint entity ids -- cold-start batch.
    fitted.transform_new_entities(new_df, "entity", "category")


if __name__ == "__main__":
    for n_entities, avg_events, n_cats in [(2000, 20, 100), (10000, 20, 100), (10000, 50, 500)]:
        t0 = time.perf_counter()
        _run(n_entities, avg_events, n_cats)
        wall = time.perf_counter() - t0
        print(f"fit_transform     n_entities={n_entities:>6} avg_events={avg_events:>3} n_categories={n_cats:>4} -> {wall * 1000:9.2f} ms")

    for n_entities, avg_events, n_cats in [(2000, 20, 100), (10000, 20, 100), (10000, 50, 500)]:
        t0 = time.perf_counter()
        _run_transform_new_entities(n_entities, avg_events, n_cats)
        wall = time.perf_counter() - t0
        print(f"transform_new     n_entities={n_entities:>6} avg_events={avg_events:>3} n_categories={n_cats:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10000, 50, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_transform_new_entities(10000, 50, 500)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
