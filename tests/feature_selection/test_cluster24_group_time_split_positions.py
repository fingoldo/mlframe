"""Cluster 24 regression: GroupTimeSeriesSplit finds each row's group position by search, with the folds the dictionary form gave (audit 2026-11-20, 24-F47)."""

import numpy as np
import pytest

from mlframe.feature_selection._cv_splitters import GroupTimeSeriesSplit, _temporal_positions


def _dict_positions(ordered, groups):
    """The pre-fix dictionary lookup."""
    order_of = {g: i for i, g in enumerate(ordered)}
    return np.fromiter((order_of[g] for g in groups), dtype=np.int64, count=len(groups))


@pytest.mark.parametrize("kind", ["int", "str", "unordered_ints"])
def test_positions_match_the_dictionary_lookup(kind):
    """Temporal order that differs from the sort order (and string labels) gives the dict's positions (24-F47)."""
    rng = np.random.default_rng(1)
    if kind == "str":
        labels = np.array([f"p{i:02d}" for i in range(40)])
    else:
        labels = np.arange(100, 140)
    ordered = rng.permutation(labels) if kind == "unordered_ints" else labels[::-1].copy()
    groups = rng.choice(labels, size=2000)
    np.testing.assert_array_equal(_temporal_positions(ordered, groups), _dict_positions(ordered, groups))


def test_split_folds_are_unchanged_for_a_shuffled_group_labelling():
    """Row blocks of each fold follow the groups' first-seen temporal order, train strictly before test (24-F47)."""
    rng = np.random.default_rng(2)
    n_groups, per = 12, 15
    ids = rng.permutation(n_groups * 10)[:n_groups]
    groups = np.repeat(ids, per)
    cv = GroupTimeSeriesSplit(n_splits=3)
    folds = list(cv.split(np.zeros((groups.size, 1)), groups=groups))
    assert folds
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])
