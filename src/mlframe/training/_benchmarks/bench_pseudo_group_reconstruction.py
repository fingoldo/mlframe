"""cProfile harness for ``training.reconstruct_pseudo_group_ids``.

Run: ``python -m mlframe.training._benchmarks.bench_pseudo_group_reconstruction``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.training._pseudo_group_reconstruction import reconstruct_pseudo_group_ids


def _run(n_entities: int, n_replicates: int, n_features: int, n_calls: int) -> None:
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((n_entities, n_features)) * 5.0
    entity_id = np.repeat(np.arange(n_entities), n_replicates)
    X = pd.DataFrame(centers[entity_id], columns=[f"f{i}" for i in range(n_features)])
    for _ in range(n_calls):
        reconstruct_pseudo_group_ids(X)


if __name__ == "__main__":
    for n_entities, n_replicates, n_features, n_calls in [(1_000, 5, 10, 20), (50_000, 10, 15, 3)]:
        t0 = time.perf_counter()
        _run(n_entities, n_replicates, n_features, n_calls)
        wall = time.perf_counter() - t0
        n_rows = n_entities * n_replicates
        print(f"rows={n_rows:>9,} entities={n_entities:>7,} -> {wall * 1000:9.2f} ms total, {wall / n_calls * 1000:8.3f} ms/call")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50_000, 10, 15, 5)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
