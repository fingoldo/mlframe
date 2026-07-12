"""cProfile harness for ``evaluation.assert_no_group_leakage``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_group_leakage_guard``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.model_selection import GroupKFold, KFold

from mlframe.evaluation.group_leakage_guard import assert_no_group_leakage


def _run(n_entities: int, rows_per_entity: int, n_splits: int) -> None:
    rng = np.random.default_rng(0)
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    n = len(entity_ids)
    X = rng.normal(0, 1, (n, 2))
    splits = list(GroupKFold(n_splits=n_splits).split(X, groups=entity_ids))
    assert_no_group_leakage(splits, entity_ids)


def _run_near_duplicate(n_entities: int, copies_per_entity: int, n_features: int, n_splits: int) -> None:
    # near-duplicate mode uses NearestNeighbors (ball-tree for euclidean) internally -- O(n log n) build + query,
    # not the naive O(n_train * n_test) pairwise matrix, so this stays tractable well past the default rows/fold
    # counts a nested-table featurizer would realistically hit.
    rng = np.random.default_rng(0)
    entity_center = rng.normal(0, 1, (n_entities, n_features))
    entity_ids = np.repeat(np.arange(n_entities), copies_per_entity)
    n = len(entity_ids)
    X = entity_center[entity_ids] + rng.normal(0, 1e-3, (n, n_features))
    anonymous_ids = np.arange(n)
    splits = list(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X))
    try:
        assert_no_group_leakage(
            splits,
            anonymous_ids,
            near_duplicate_features=X,
            near_duplicate_metric="euclidean",
            near_duplicate_max_neighbor_distance=0.05,
        )
    except ValueError:
        pass  # expected: this synthetic dataset IS leaky by construction; we're only timing the check itself.


if __name__ == "__main__":
    for n_entities, rows_per_entity, n_splits in [(5_000, 10, 5), (50_000, 10, 5)]:
        t0 = time.perf_counter()
        _run(n_entities, rows_per_entity, n_splits)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7,} rows/entity={rows_per_entity:>3} n_splits={n_splits} -> {wall * 1000:9.2f} ms")

    # capped well below the group-key-only sizes above: nearest-neighbor search is asymptotically cheap
    # (ball-tree) but still far more expensive per-row than a hash-set overlap check, and this is a check
    # meant to run once per CV setup, not in a hot per-row loop.
    for n_entities, copies_per_entity, n_features, n_splits in [(2_000, 3, 8, 5), (10_000, 3, 8, 5)]:
        t0 = time.perf_counter()
        _run_near_duplicate(n_entities, copies_per_entity, n_features, n_splits)
        wall = time.perf_counter() - t0
        n = n_entities * copies_per_entity
        print(f"[near-dup] n_rows={n:>7,} n_features={n_features:>2} n_splits={n_splits} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 10, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_near_duplicate(10_000, 3, 8, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
