"""Cluster 22 regression: the cluster-bootstrap resample is one gather and returns exactly what the per-group concatenation did (audit 2026-11-20, 22-F24)."""

import numpy as np

from mlframe.evaluation._bootstrap_clusters import _group_slices, cluster_resample_indices


def _reference(rng, order, offsets):
    """The pre-fix list-comprehension form."""
    n_groups = offsets.shape[0] - 1
    picks = rng.integers(0, n_groups, size=n_groups)
    return np.concatenate([order[offsets[g] : offsets[g + 1]] for g in picks])


def test_matches_the_per_group_concatenation_for_uneven_groups():
    """Random group sizes (including singletons): same indices, same order, same rng consumption (22-F24)."""
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 60, size=900)
    order, offsets = _group_slices(groups)
    for seed in range(20):
        got = cluster_resample_indices(np.random.default_rng(seed), order, offsets)
        want = _reference(np.random.default_rng(seed), order, offsets)
        np.testing.assert_array_equal(got, want)


def test_single_group_returns_all_rows():
    """With one group every resample is that group's rows (22-F24)."""
    order, offsets = _group_slices(np.zeros(7, dtype=int))
    np.testing.assert_array_equal(cluster_resample_indices(np.random.default_rng(0), order, offsets), order)
