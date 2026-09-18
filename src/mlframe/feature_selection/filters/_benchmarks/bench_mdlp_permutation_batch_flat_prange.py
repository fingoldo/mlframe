"""A/B of ``_mdlp_permutation_batch_njit``: the previous prange-over-NODES layout vs the flat prange over
(node, permutation) pairs, on the shapes the BFS recursion actually produces (1 root node, 2 children, many
leaves). Checks the accept flags are identical and reports the median of repeats.

Usage::

    python -m mlframe.feature_selection.filters._benchmarks.bench_mdlp_permutation_batch_flat_prange --n 20000 --repeats 5
"""

from __future__ import annotations

import argparse
import math
import statistics
import time

import numba
import numpy as np
from numba import njit

from mlframe.feature_selection.filters._mdlp_validated_split import _mdlp_permutation_batch_njit
from mlframe.feature_selection.filters.supervised_binning import _mdlp_best_split_njit


@njit(nogil=True, parallel=True)
def _batch_prange_over_nodes(x_padded, y_padded, node_sizes, node_gains, n_classes_arr, min_split_size, n_permutations, base_seeds, alpha):
    """The layout before the flat prange (kept here as the A/B reference)."""
    n_nodes = x_padded.shape[0]
    accept = np.zeros(n_nodes, dtype=np.bool_)
    for node in numba.prange(n_nodes):
        ni = node_sizes[node]
        x_i = x_padded[node, :ni]
        y_i = y_padded[node, :ni]
        nc = n_classes_arr[node]
        gain = node_gains[node]
        base_seed = base_seeds[node]
        null_gains = np.empty(n_permutations, dtype=np.float64)
        for p in range(n_permutations):
            np.random.seed(base_seed + p)
            y_perm = y_i.copy()
            for i in range(ni - 1, 0, -1):
                j = int(np.random.randint(0, i + 1))
                tmp = y_perm[i]
                y_perm[i] = y_perm[j]
                y_perm[j] = tmp
            _, g, _, _, _ = _mdlp_best_split_njit(x_i, y_perm, nc, min_split_size)
            null_gains[p] = g if g > 0.0 else 0.0
        null_gains.sort()
        q_idx = min(math.ceil((1.0 - alpha) * n_permutations) - 1, n_permutations - 1)
        q_idx = max(0, q_idx)
        accept[node] = gain > null_gains[q_idx]
    return accept


def _level(n_nodes: int, n: int, n_classes: int, rng):
    size = n // n_nodes
    x = np.sort(rng.random((n_nodes, size)), axis=1)
    y = (x * n_classes + rng.normal(scale=2.0, size=x.shape)).clip(0, n_classes - 1).astype(np.int64)
    gains = np.full(n_nodes, 0.01)
    return x, y, np.full(n_nodes, size, dtype=np.int64), gains, np.full(n_nodes, n_classes, dtype=np.int64), np.arange(n_nodes, dtype=np.int64) * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--perms", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    print("numba threads:", numba.get_num_threads())
    for n_nodes in (1, 2, 8, 64):
        x, y, sizes, gains, ncls, seeds = _level(n_nodes, args.n, args.classes, rng)
        call = (x, y, sizes, gains, ncls, 5, args.perms, seeds, 0.05)
        a_old = _batch_prange_over_nodes(*call)
        a_new = _mdlp_permutation_batch_njit(*call)
        assert np.array_equal(a_old, a_new), (n_nodes, a_old, a_new)
        t_old, t_new = [], []
        for _ in range(args.repeats):
            t = time.perf_counter(); _batch_prange_over_nodes(*call); t_old.append(time.perf_counter() - t)
            t = time.perf_counter(); _mdlp_permutation_batch_njit(*call); t_new.append(time.perf_counter() - t)
        mo, mn = statistics.median(t_old), statistics.median(t_new)
        print(f"n_nodes={n_nodes:3d} rows/node={x.shape[1]:6d} prange-over-nodes={mo:.3f}s flat={mn:.3f}s speedup={mo / mn:.2f}x accepts identical")


if __name__ == "__main__":
    main()
