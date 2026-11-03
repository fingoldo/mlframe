"""The early-stopping MDLP permutation batch accepts exactly the nodes the full null (every permutation, sorted, quantile threshold) accepts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.feature_selection.filters._mdlp_validated_split import _mdlp_best_split_njit, _mdlp_permutation_batch_njit


def _full_null_verdicts(xs, ys, sizes, gains, ncl, min_split, n_perm, seeds, alpha):
    """Reference: every permutation drawn with the kernel's seeding, nulls sorted, accept when the gain exceeds the (1 - alpha) order statistic."""
    out = np.zeros(len(sizes), dtype=bool)
    q = max(0, min(math.ceil((1.0 - alpha) * n_perm) - 1, n_perm - 1))
    for node in range(len(sizes)):
        ni = int(sizes[node])
        nulls = []
        for p in range(n_perm):
            np.random.seed(int(seeds[node] + p))
            y = ys[node, :ni].copy()
            for i in range(ni - 1, 0, -1):
                j = int(np.random.randint(0, i + 1))
                y[i], y[j] = y[j], y[i]
            _, g, _, _, _ = _mdlp_best_split_njit(xs[node, :ni], y, int(ncl[node]), min_split)
            nulls.append(max(g, 0.0))
        out[node] = gains[node] > np.sort(nulls)[q]
    return out


@pytest.mark.parametrize("n_perm,alpha", [(39, 0.05), (20, 0.2)])
def test_verdicts_equal_the_full_null(n_perm, alpha):
    """Signal and noise nodes of different sizes, gains scaled around their own observed gain so some sit near the threshold."""
    rng = np.random.default_rng(5)
    n_nodes, max_n = 6, 400
    xs = np.zeros((n_nodes, max_n))
    ys = np.zeros((n_nodes, max_n), dtype=np.int64)
    sizes = np.empty(n_nodes, dtype=np.int64)
    gains = np.empty(n_nodes)
    ncl = np.full(n_nodes, 2, dtype=np.int64)
    seeds = rng.integers(0, 1 << 30, n_nodes).astype(np.int64)
    for i in range(n_nodes):
        ni = int(rng.integers(60, max_n + 1))
        x = np.sort(rng.standard_normal(ni))
        y = ((x > 0).astype(int) ^ (rng.random(ni) < 0.35)) if i % 2 == 0 else rng.integers(0, 2, ni)
        xs[i, :ni], ys[i, :ni], sizes[i] = x, y, ni
        _, g, _, _, _ = _mdlp_best_split_njit(x, y.astype(np.int64), 2, 5)
        gains[i] = g * (0.7, 1.0, 1.3)[i % 3]
    got = _mdlp_permutation_batch_njit(xs, ys, sizes, gains, ncl, 5, n_perm, seeds, alpha)
    want = _full_null_verdicts(xs, ys, sizes, gains, ncl, 5, n_perm, seeds, alpha)
    assert np.array_equal(got, want)
