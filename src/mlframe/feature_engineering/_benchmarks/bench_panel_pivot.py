"""cProfile harness for ``feature_engineering.panel_pivot.pivot_time_indexed_panel``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_panel_pivot``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.panel_pivot import pivot_time_indexed_panel


def _make_panel(n_entities: int, max_hist: int, n_value_cols: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows_id, rows_t = [], []
    for e in range(n_entities):
        hist_len = rng.integers(3, max_hist)
        rows_id.append(np.full(hist_len, e))
        rows_t.append(np.arange(hist_len))
    ids = np.concatenate(rows_id)
    ts = np.concatenate(rows_t)
    data = {"id": ids, "t": ts}
    for c in range(n_value_cols):
        data[f"x{c}"] = rng.normal(size=len(ids))
    return pd.DataFrame(data)


def _run(
    n_entities: int,
    max_hist: int,
    n_value_cols: int,
    n_calls: int,
    add_time_gaps: bool = False,
    agg_stats: tuple[str, ...] | None = None,
    max_lags: int | None = None,
) -> None:
    df = _make_panel(n_entities, max_hist, n_value_cols)
    value_cols = [c for c in df.columns if c.startswith("x")]
    effective_max_lags = max_hist if max_lags is None else max_lags
    for _ in range(n_calls):
        pivot_time_indexed_panel(
            df, "id", "t", value_cols, max_lags=effective_max_lags, add_time_gaps=add_time_gaps, agg_stats=agg_stats
        )


if __name__ == "__main__":
    # use a max_lags well below max_hist here so agg_stats has truncated-away history to summarize --
    # the earlier add_time_gaps sweep uses max_lags == max_hist (no truncation) on purpose, so agg_stats
    # gets its own sweep below rather than reusing that loop.
    AGG_STATS = ("mean", "std", "min", "max")

    for n_entities, max_hist, n_value_cols, n_calls in [(2000, 13, 5, 20), (50000, 13, 5, 20), (50000, 13, 20, 20)]:
        t0 = time.perf_counter()
        _run(n_entities, max_hist, n_value_cols, n_calls)
        wall = time.perf_counter() - t0
        print(f"n_entities={n_entities:>7} max_hist={max_hist:>3} n_value_cols={n_value_cols:>3} n_calls={n_calls:>4} -> {wall * 1000:9.2f} ms")

        t0 = time.perf_counter()
        _run(n_entities, max_hist, n_value_cols, n_calls, add_time_gaps=True)
        wall_gaps = time.perf_counter() - t0
        print(
            f"n_entities={n_entities:>7} max_hist={max_hist:>3} n_value_cols={n_value_cols:>3} n_calls={n_calls:>4} "
            f"add_time_gaps=True -> {wall_gaps * 1000:9.2f} ms"
        )

    for n_entities, max_hist, n_value_cols, n_calls in [(2000, 40, 5, 20), (50000, 40, 5, 20), (50000, 40, 20, 20)]:
        t0 = time.perf_counter()
        _run(n_entities, max_hist, n_value_cols, n_calls, agg_stats=AGG_STATS, max_lags=13)
        wall_agg = time.perf_counter() - t0
        print(
            f"n_entities={n_entities:>7} max_hist(history cap)={max_hist:>3} max_lags=13 n_value_cols={n_value_cols:>3} "
            f"n_calls={n_calls:>4} agg_stats={AGG_STATS} -> {wall_agg * 1000:9.2f} ms"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 13, 20, 20)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 13, 20, 20, add_time_gaps=True)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- add_time_gaps=True ---")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(50000, 40, 20, 20, agg_stats=AGG_STATS, max_lags=13)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- agg_stats=('mean','std','min','max') ---")
    print(buf.getvalue())
