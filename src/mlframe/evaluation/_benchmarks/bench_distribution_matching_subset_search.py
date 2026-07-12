"""cProfile harness for ``evaluation.distribution_matching_subset_search.distribution_matching_subset_search``.

Run: ``python -m mlframe.evaluation._benchmarks.bench_distribution_matching_subset_search``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO
from typing import Optional

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


def _run(
    n_blocks_total: int,
    rows_per_block: int,
    n_features: int,
    n_blocks: int,
    n_trials: int,
    search_strategy: str = "random",
    joint_distance_mode: Optional[str] = None,
) -> None:
    train_df, target_df = _make_data(n_blocks_total, rows_per_block, n_features)
    feature_cols = [f"f{c}" for c in range(n_features)]
    distribution_matching_subset_search(
        train_df,
        target_df,
        block_col="block",
        feature_cols=feature_cols,
        n_blocks=n_blocks,
        n_trials=n_trials,
        random_state=0,
        search_strategy=search_strategy,
        joint_distance_mode=joint_distance_mode,
    )


if __name__ == "__main__":
    for search_strategy in ["random", "greedy_swap"]:
        print(f"--- search_strategy={search_strategy} ---")
        for n_blocks_total, rows_per_block, n_features, n_blocks, n_trials in [(30, 50, 5, 5, 200), (100, 200, 5, 10, 200), (100, 200, 20, 10, 100)]:
            t0 = time.perf_counter()
            _run(n_blocks_total, rows_per_block, n_features, n_blocks, n_trials, search_strategy)
            wall = time.perf_counter() - t0
            print(f"n_blocks_total={n_blocks_total:>4} rows_per_block={rows_per_block:>4} n_features={n_features:>3} n_trials={n_trials:>4} -> {wall * 1000:9.2f} ms")

    # cProfile hotspot check for greedy_swap specifically (the new code path) -- looking for wasted work, e.g.
    # recomputing the full mean-KS from scratch on every swap instead of incrementally updating it.
    profiler = cProfile.Profile()
    profiler.enable()
    _run(100, 200, 20, 10, 100, "random")
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print("--- cProfile: random ---")
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run(100, 200, 20, 10, 100, "greedy_swap")
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("--- cProfile: greedy_swap ---")
    print(buf.getvalue())

    # cProfile hotspot check for the opt-in joint_distance_mode="energy" path -- looking for whether the
    # O(n_sample * n_target * n_features) pairwise-distance cost of _energy_distance dominates the wall time,
    # and whether the target-vs-itself term is genuinely computed only once (not per trial) as documented.
    # Uses a smaller scale than the KS profiles above (5 features, 30 trials, not 20/100) -- a first run at
    # the KS scale (20 features, 100 trials, 2000-row samples x 500-row target) measured 105-125s for this
    # mode alone, entirely from materializing (n_sample, n_target, n_features) broadcast arrays in
    # np.linalg.norm; that cost is the documented reason this mode is opt-in, not something to hide by
    # shrinking the demonstration away, so both scales are recorded here.
    for search_strategy in ["random", "greedy_swap"]:
        t0 = time.perf_counter()
        _run(30, 50, 5, 5, 30, search_strategy, joint_distance_mode="energy")
        wall = time.perf_counter() - t0
        print(f"joint_distance_mode=energy (small scale) search_strategy={search_strategy:>11} -> {wall * 1000:9.2f} ms")

    profiler = cProfile.Profile()
    profiler.enable()
    _run(30, 50, 5, 5, 30, "greedy_swap", joint_distance_mode="energy")
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(20)
    print("--- cProfile: greedy_swap, joint_distance_mode=energy (small scale) ---")
    print(buf.getvalue())

    # Findings (measured): at the KS-profile scale (n_blocks_total=100, rows_per_block=200, n_features=20,
    # n_blocks=10, n_trials=100 -> ~2000-row samples vs a 500-row target), joint_distance_mode="energy" took
    # 125.17s (random) / 108.22s (greedy_swap) wall vs ~0.7-0.8s for the default KS-only path at the SAME
    # n_trials -- roughly two orders of magnitude slower, almost entirely inside np.linalg.norm's broadcast
    # (cProfile: _energy_distance tottime 51.5s, np.linalg.norm cumtime 51.9s across 201 calls, out of 105s
    # total). This is O(n_sample * n_target * n_features) pairwise-distance work vs KS's O(n log n) sort per
    # feature -- a fundamentally more expensive statistic, not an implementation inefficiency: the
    # target-vs-itself term IS precomputed exactly once per search call (confirmed by call count: one
    # non-trial call site vs the per-trial _energy_distance calls), so there is no further wasted-work
    # hotspot to fix. This cost is *why* the mode is opt-in and documented in its docstring; callers with a
    # large target_df or many features should subsample target_df (and/or restrict feature_cols) before
    # enabling it, or accept the added wall time as the price of catching joint-structure mismatches that
    # per-feature KS cannot see at all.

    # Findings (measured, see PR/commit for exact run): greedy_swap's hotspot is the SAME _mean_ks_statistic /
    # _ks_statistic path as "random" -- there is no incremental-update opportunity here because each swap only
    # changes ONE block out of n_blocks, but _mean_ks_statistic still has to re-concatenate/re-sort the full
    # sample column per feature (the per-feature KS statistic isn't a simple sum/reducible statistic over
    # blocks -- it depends on the full empirical CDF shape, so a single-block swap cannot be reflected via a
    # cheap delta without re-deriving the CDF anyway). The single-block lazy scoring cache (block_scores) DOES
    # already avoid the one wasted-work case that exists: never rescoring an already-scored block while it
    # stays in the subset. No further actionable hotspot found at this data scale.
