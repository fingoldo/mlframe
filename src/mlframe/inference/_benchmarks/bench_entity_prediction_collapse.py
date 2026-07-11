"""cProfile harness for ``inference.entity_prediction_collapse.collapse_predictions_by_group``.

Run: ``python -m mlframe.inference._benchmarks.bench_entity_prediction_collapse``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.inference.entity_prediction_collapse import collapse_predictions_by_group


def _run(n_entities: int, rows_per_entity: int, stat: str) -> None:
    rng = np.random.default_rng(0)
    entity_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    predictions = rng.uniform(size=len(entity_ids))
    collapse_predictions_by_group(predictions, entity_ids, stat=stat)


if __name__ == "__main__":
    for n_entities, rows_per_entity in [(5000, 10), (50000, 10), (50000, 30)]:
        t0 = time.perf_counter()
        _run(n_entities, rows_per_entity, "mean")
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>6} rows/entity={rows_per_entity:>3} (mean) -> {wall * 1000:9.2f} ms")

    t0 = time.perf_counter()
    _run(50000, 30, "quantile")
    wall = time.perf_counter() - t0
    print(f"n_entities=50000 rows/entity=30 (quantile) -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 30, "quantile")
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print(buf.getvalue())
