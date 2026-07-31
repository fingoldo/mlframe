"""A/B bench for the BFS level-order batching of MDLP validated-split permutation testing.

Compares the old DFS recursion (`_mdlp_recurse_validated`, one permutation-null njit call per node)
against the new BFS recursion (`_mdlp_recurse_validated_bfs`, one batched njit call per tree level)
on a constructed multi-segment "staircase" signal that actually exercises many splits/permutation
fallbacks (pure noise / already-optimized short signals don't recurse deep enough to show the win).
"""

import time

import numpy as np

from mlframe.feature_selection.filters._mdlp_validated_split import (
    _mdlp_recurse_validated,
    _mdlp_recurse_validated_bfs,
)


def _make_staircase_signal(n: int, n_segments: int, noise_flip: float, seed: int):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.0, 1.0, size=n))
    seg = np.minimum((x * n_segments).astype(np.int64), n_segments - 1)
    y = (seg % 2).astype(np.int64)
    flip = rng.random(n) < noise_flip
    y = np.where(flip, 1 - y, y)
    return x, y


def run_dfs(x, y, min_split_size, max_depth, alpha, n_permutations, seed, bonferroni, tree_wide_alpha):
    splits: list = []
    _mdlp_recurse_validated(
        x, y, splits, 0, min_split_size, max_depth, alpha, n_permutations, seed, bonferroni, tree_wide_alpha=tree_wide_alpha
    )
    splits.sort()
    return splits


def run_bfs(x, y, min_split_size, max_depth, alpha, n_permutations, seed, bonferroni, tree_wide_alpha):
    splits = _mdlp_recurse_validated_bfs(x, y, min_split_size, max_depth, alpha, n_permutations, seed, bonferroni, tree_wide_alpha)
    splits.sort()
    return splits


def main():
    n, n_segments, noise_flip, seed = 500_000, 40, 0.35, 42
    min_split_size, max_depth, alpha, n_permutations, bonferroni, tree_wide_alpha = 30, 10, 0.05, 200, True, None

    x, y = _make_staircase_signal(n, n_segments, noise_flip, seed)

    # warm JIT caches for both paths before timing
    x_w, y_w = _make_staircase_signal(2000, n_segments, noise_flip, seed)
    run_dfs(x_w, y_w, min_split_size, max_depth, alpha, n_permutations, seed, bonferroni, tree_wide_alpha)
    run_bfs(x_w, y_w, min_split_size, max_depth, alpha, n_permutations, seed, bonferroni, tree_wide_alpha)

    t0 = time.perf_counter()
    dfs_splits = run_dfs(x, y, min_split_size, max_depth, alpha, n_permutations, seed, bonferroni, tree_wide_alpha)
    t_dfs = time.perf_counter() - t0

    t0 = time.perf_counter()
    bfs_splits = run_bfs(x, y, min_split_size, max_depth, alpha, n_permutations, seed, bonferroni, tree_wide_alpha)
    t_bfs = time.perf_counter() - t0

    print(f"dfs: {t_dfs:.3f}s splits={len(dfs_splits)}")
    print(f"bfs: {t_bfs:.3f}s splits={len(bfs_splits)}")
    print(f"speedup: {t_dfs / t_bfs:.2f}x")
    print(f"overlap check: same split count within tolerance: {abs(len(dfs_splits) - len(bfs_splits)) <= 2}")


if __name__ == "__main__":
    main()
