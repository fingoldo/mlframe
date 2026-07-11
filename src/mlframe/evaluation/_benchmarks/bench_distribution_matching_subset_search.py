"""cProfile harness for ``evaluation.distribution_matching_subset_search.distribution_matching_subset_search``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_distribution_matching_subset_search``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np
import pandas as pd

from mlframe.evaluation.distribution_matching_subset_search import distribution_matching_subset_search


def _make_data(n_blocks_total: int, rows_per_block: int, n_features: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    data = {"block": np.repeat(np.arange(n_blocks_total), rows_per_block)}
    n = n_blocks_total * rows_per_block
    for c in range(n_features):
        data[f"f{c}"] = rng.normal(size=n)
    train_df = pd.DataFrame(data)
    target_df = pd.DataFrame({f"f{c}": rng.normal(size=500) for c in range(n_features)})
    return train_df, target_df


def _run(n_blocks_total: int, rows_per_block: int, n_features: int, n_blocks: int, n_trials: int) -> None:
    train_df, target_df = _make_data(n_blocks_total, rows_per_block, n_features)
    feature_cols = [f"f{c}" for c in range(n_features)]
    distribution_matching_subset_search(train_df, target_df, block_col="block", feature_cols=feature_cols, n_blocks=n_blocks, n_trials=n_trials, random_state=0)


if __name__ == "__main__":
    for n_blocks_total, rows_per_block, n_features, n_blocks, n_trials in [(30, 50, 5, 5, 200), (100, 200, 5, 10, 200), (100, 200, 20, 10, 100)]:
        t0 = time.perf_counter()
        _run(n_blocks_total, rows_per_block, n_features, n_blocks, n_trials)
        wall = time.perf_counter() - t0
        print(f"n_blocks_total={n_blocks_total:>4} rows_per_block={rows_per_block:>4} n_features={n_features:>3} n_trials={n_trials:>4} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100, 200, 20, 10, 100)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
