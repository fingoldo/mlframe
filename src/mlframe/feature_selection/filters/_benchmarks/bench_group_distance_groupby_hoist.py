"""Bench: linear-time Wasserstein-1 in ``generate_group_distance_features``.

The hottest leg (cProfile: ~73% of wall) was the per-(group, num_col) call to
``scipy.stats.wasserstein_distance(group_vals, global_sorted)``, which re-sorts
and re-searchsorts the LARGE global array on every group -- even though that
global array is a per-num_col invariant already sorted once by the caller. The
``_wasserstein1_sorted_kernel`` njit merge exploits the pre-sorted global to
compute scipy's exact integral in O(nu+nv), no per-group re-sort. This bench
measures the full-function win (~2.8x@20k / 3.9x@50k / 5.3x@100k).

Run:
    CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection.filters._benchmarks.bench_group_distance_groupby_hoist
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _make_frame(n: int, n_num: int, n_groups: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"grp": rng.integers(0, n_groups, size=n)}
    for j in range(n_num):
        data[f"x{j}"] = rng.normal(size=n)
    return pd.DataFrame(data)


def _bench(fn, X, group_cols, num_cols, repeats: int = 5) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(X, group_cols, num_cols)
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    from mlframe.feature_selection.filters._group_distance_fe import (
        generate_group_distance_features,
    )

    for n, n_num, n_groups in [(20_000, 12, 50), (50_000, 12, 100), (100_000, 16, 200)]:
        X = _make_frame(n, n_num, n_groups)
        group_cols = ["grp"]
        num_cols = [f"x{j}" for j in range(n_num)]
        # warm
        generate_group_distance_features(X.head(200), group_cols, num_cols)
        t = _bench(generate_group_distance_features, X, group_cols, num_cols)
        print(f"n={n:>7} num_cols={n_num} groups={n_groups}: best={t*1e3:8.1f} ms")


if __name__ == "__main__":
    main()
