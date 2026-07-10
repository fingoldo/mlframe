"""cProfile harness for ``evaluation.assert_no_group_leakage``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_group_leakage_guard``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
from sklearn.model_selection import GroupKFold

from mlframe.evaluation.group_leakage_guard import assert_no_group_leakage


def _run(n_entities: int, rows_per_entity: int, n_splits: int) -> None:
    rng = np.random.default_rng(0)
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    n = len(entity_ids)
    X = rng.normal(0, 1, (n, 2))
    splits = list(GroupKFold(n_splits=n_splits).split(X, groups=entity_ids))
    assert_no_group_leakage(splits, entity_ids)


if __name__ == "__main__":
    for n_entities, rows_per_entity, n_splits in [(5_000, 10, 5), (50_000, 10, 5)]:
        t0 = time.perf_counter()
        _run(n_entities, rows_per_entity, n_splits)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7,} rows/entity={rows_per_entity:>3} n_splits={n_splits} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 10, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
