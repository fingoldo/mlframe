"""CPX15 identity: GroupTimeSeriesSplit.split must yield bit-identical train/test arrays after the incremental
train-accumulation rewrite (drop per-fold sorted(set(...)) rebuild over the growing prefix).

The OLD reference recomputes sorted(set(...)) from scratch every fold; the NEW code is the live split(). Both must
agree exactly (same int64 dtype, same values, same order) across all folds, including the interleaved-group case
where a group's index list is not a contiguous slice, and including max_train_size slicing.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.selection import GroupTimeSeriesSplit


def _old_split(groups, n_splits, max_train_size=None):
    """Verbatim reference of the pre-CPX15 per-fold sorted(set(...)) accumulation."""
    from sklearn.model_selection._split import indexable, _num_samples

    _, _, groups = indexable(None, None, groups)
    n_samples = _num_samples(groups)
    n_folds = n_splits + 1
    u, ind = np.unique(groups, return_index=True)
    unique_groups = u[np.argsort(ind)]
    n_groups = _num_samples(unique_groups)
    group_dict = {}
    for idx in range(n_samples):
        g = groups[idx]
        group_dict.setdefault(g, []).append(idx)
    group_test_size = n_groups // n_folds
    group_test_starts = range(n_groups - n_splits * group_test_size, n_groups, group_test_size)
    out = []
    for group_test_start in group_test_starts:
        train_buf = []
        for tg in unique_groups[:group_test_start]:
            train_buf.extend(group_dict[tg])
        train_array = np.array(sorted(set(train_buf)), dtype=np.int64)
        train_end = train_array.size
        if max_train_size and max_train_size < train_end:
            train_array = train_array[train_end - max_train_size : train_end]
        test_buf = []
        for tg in unique_groups[group_test_start : group_test_start + group_test_size]:
            test_buf.extend(group_dict[tg])
        test_array = np.array(sorted(set(test_buf)), dtype=np.int64)
        out.append((train_array, test_array))
    return out


def _make_groups(n_samples, n_groups, interleave, seed=0):
    rng = np.random.default_rng(seed)
    if not interleave:
        bounds = np.linspace(0, n_samples, n_groups + 1).astype(np.int64)
        groups = np.empty(n_samples, dtype=np.int64)
        for g in range(n_groups):
            groups[bounds[g] : bounds[g + 1]] = g
        return groups
    groups = rng.integers(0, n_groups, size=n_samples).astype(np.int64)
    groups[:n_groups] = np.arange(n_groups)
    return groups


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("max_train_size", [None, 137])
@pytest.mark.parametrize("n_samples,n_groups,n_splits", [(2000, 50, 5), (5000, 17, 3), (3000, 300, 10)])
def test_cpx15_split_identity(interleave, max_train_size, n_samples, n_groups, n_splits):
    groups = _make_groups(n_samples, n_groups, interleave)
    old = _old_split(groups, n_splits, max_train_size)
    new = list(GroupTimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size).split(groups, groups=groups))
    assert len(old) == len(new) == n_splits
    for (tr_o, te_o), (tr_n, te_n) in zip(old, new):
        assert tr_n.dtype == np.int64 and te_n.dtype == np.int64
        np.testing.assert_array_equal(tr_o, tr_n)
        np.testing.assert_array_equal(te_o, te_n)


def test_cpx15_docstring_example_unchanged():
    groups = np.array(["a"] * 6 + ["b"] * 5 + ["c"] * 4 + ["d"] * 3)
    folds = list(GroupTimeSeriesSplit(n_splits=3).split(groups, groups=groups))
    np.testing.assert_array_equal(folds[0][0], np.arange(6))
    np.testing.assert_array_equal(folds[0][1], np.arange(6, 11))
    np.testing.assert_array_equal(folds[2][0], np.arange(15))
    np.testing.assert_array_equal(folds[2][1], np.arange(15, 18))
