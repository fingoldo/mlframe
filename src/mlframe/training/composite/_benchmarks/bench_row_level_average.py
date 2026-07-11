"""cProfile harness for ``training.composite.compute_row_level_then_average_predictions``.

Run: ``python -m mlframe.training.composite._benchmarks.bench_row_level_average``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.training.composite import compute_row_level_then_average_predictions


def _make_dataset(n_entities: int, k_rows: int, seed: int):
    rng = np.random.default_rng(seed)
    x1_rows: list[float] = []
    x2_rows: list[float] = []
    entity_ids_list: list[int] = []
    y_entity = np.zeros(n_entities)
    for e in range(n_entities):
        x1 = rng.normal(size=k_rows)
        x2 = rng.normal(size=k_rows)
        y_entity[e] = 1.0 if ((x1 * x2) > 0).mean() > 0.5 else 0.0
        x1_rows.extend(x1)
        x2_rows.extend(x2)
        entity_ids_list.extend([e] * k_rows)
    entity_ids = np.array(entity_ids_list)
    X_rows = pd.DataFrame({"x1": x1_rows, "x2": x2_rows})
    return X_rows, y_entity[entity_ids], entity_ids


def _run(n_entities: int, flag_low_confidence_quantile: float | None = None) -> None:
    X_rows, y_row_broadcast, entity_ids = _make_dataset(n_entities, k_rows=10, seed=0)
    compute_row_level_then_average_predictions(
        X_rows, y_row_broadcast, entity_ids, model_factory=lambda: LinearRegression(), n_splits=5,
        flag_low_confidence_quantile=flag_low_confidence_quantile,
    )


if __name__ == "__main__":
    for n in [200, 2_000, 10_000]:
        t0 = time.perf_counter()
        _run(n)
        wall = time.perf_counter() - t0
        print(f"n_entities={n:>7,} -> {wall * 1000:9.2f} ms")

    for n in [200, 2_000, 10_000]:
        t0 = time.perf_counter()
        _run(n, flag_low_confidence_quantile=0.75)
        wall = time.perf_counter() - t0
        print(f"n_entities={n:>7,} (flag_low_confidence_quantile=0.75) -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(10_000, flag_low_confidence_quantile=0.75)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
