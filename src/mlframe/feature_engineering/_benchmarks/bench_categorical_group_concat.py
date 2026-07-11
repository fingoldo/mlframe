"""cProfile harness for ``feature_engineering.categorical_group_concat.concat_categorical_group``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_categorical_group_concat``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.feature_engineering.categorical_group_concat import auto_concat_categorical_groups, concat_categorical_group


def _make_dataset(n_rows: int, n_categories: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "a": rng.integers(0, n_categories, n_rows).astype(str),
            "b": rng.integers(0, n_categories, n_rows).astype(str),
            "c": rng.integers(0, n_categories, n_rows).astype(str),
        }
    )


def _run(n_rows: int, n_categories: int) -> None:
    df = _make_dataset(n_rows, n_categories, seed=0)
    concat_categorical_group(df, ["a", "b", "c"])


def _make_search_dataset(n_rows: int, n_columns: int, n_categories: int, seed: int) -> tuple:
    # A small column pool for the greedy MI search path -- deliberately kept narrow (n_columns capped in the
    # __main__ sweep below) since the search is O(k^2) trial-composite MI evaluations for a k-column pool.
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"c{i}": rng.integers(0, n_categories, n_rows).astype(str) for i in range(n_columns)})
    # y depends on the joint pair (c0, c1) only, so the search has a real grouping to find rather than noise.
    y = ((df["c0"].astype(int) + df["c1"].astype(int)) % n_categories == 0).astype(int).to_numpy()
    return df, y


def _run_search(n_rows: int, n_columns: int, n_categories: int, max_group_size: int) -> None:
    df, y = _make_search_dataset(n_rows, n_columns, n_categories, seed=0)
    auto_concat_categorical_groups(df, list(df.columns), y, max_group_size=max_group_size, random_state=0)


if __name__ == "__main__":
    for n_rows, n_categories in [(50000, 20), (1000000, 20), (1000000, 200)]:
        t0 = time.perf_counter()
        _run(n_rows, n_categories)
        wall = time.perf_counter() - t0
        print(f"n_rows={n_rows:>8} n_categories={n_categories:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(1000000, 200)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    # Search path (auto_concat_categorical_groups / discover_categorical_groups): rows kept modest and the
    # column pool capped at 6 (max_group_size=3) so the O(k^2) trial-composite MI cost stays sane -- this is
    # a search-quality/latency profile, not a throughput sweep like the plain concatenator above.
    for n_rows, n_columns, n_categories in [(2000, 6, 10), (10000, 6, 10)]:
        t0 = time.perf_counter()
        _run_search(n_rows, n_columns, n_categories, max_group_size=3)
        wall = time.perf_counter() - t0
        print(f"[search] n_rows={n_rows:>6} n_columns={n_columns:>2} n_categories={n_categories:>3} -> {wall * 1000:9.2f} ms")

    profiler_search = cProfile.Profile()
    profiler_search.enable()
    _run_search(10000, 6, 10, max_group_size=3)
    profiler_search.disable()
    buf_search = StringIO()
    stats_search = pstats.Stats(profiler_search, stream=buf_search).sort_stats("cumulative")
    stats_search.print_stats(15)
    print(buf_search.getvalue())
