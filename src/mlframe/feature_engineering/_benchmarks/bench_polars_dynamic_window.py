"""cProfile harness for ``feature_engineering.polars_dynamic_window_aggregate``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_polars_dynamic_window``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.polars_dynamic_window import polars_dynamic_window_aggregate


def _make_data(n_entities: int, n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    entity_ids = np.repeat(np.arange(n_entities), n_days)
    t = np.tile(dates, n_entities)
    x = rng.normal(0, 1, n_entities * n_days)
    return pd.DataFrame({"entity": entity_ids, "t": t, "x": x})


def _run(n_entities: int, n_days: int) -> None:
    df = _make_data(n_entities, n_days, seed=0)
    polars_dynamic_window_aggregate(df, "t", ["x"], every="7d", group_col="entity", agg_funcs=["mean", "std", "count"])


if __name__ == "__main__":
    for n_entities, n_days in [(2_000, 60), (20_000, 60)]:
        t0 = time.perf_counter()
        _run(n_entities, n_days)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7,} n_days={n_days:>3} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(20_000, 60)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
