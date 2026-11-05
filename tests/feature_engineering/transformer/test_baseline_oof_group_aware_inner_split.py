"""The baseline's out-of-fold predictions must not see a row's own duplicates in training when the rows are grouped.

Each group is a block of identical rows that share one label. With a shuffled inner split, the duplicates of a held-out row
sit in the training folds and the model reads its own target back, so the out-of-fold accuracy is near 1. A group-aware
inner split holds whole groups out, so the same predictions fall to chance.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._baseline_oof import fit_baseline_predict_oof, fold_train_groups


def _grouped_duplicates(n_groups: int = 40, rows_per_group: int = 4, seed: int = 0):
    """Identical rows per group with a random feature vector and a random label per group.

    The feature vector is drawn per group, so a model can only tell a group's rows apart by having seen them: held-out
    groups carry no information, and an ordinal group id would let neighbouring groups' labels leak in instead.
    """
    rng = np.random.default_rng(seed)
    group_feat = rng.normal(size=(n_groups, 8)).astype(np.float32)
    group_label = rng.integers(0, 2, size=n_groups)
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    X = group_feat[groups]
    y = group_label[groups].astype(np.int32)
    return X, y, groups


def test_shuffled_inner_split_reads_a_rows_own_target_back():
    """Without group labels the duplicates leak: the out-of-fold predictions reproduce the labels."""
    X, y, _ = _grouped_duplicates()
    preds = fit_baseline_predict_oof(X, y, "binary", seed=0)
    assert float(np.mean((preds > 0.5) == y)) > 0.9


def test_group_aware_inner_split_holds_whole_groups_out():
    """With group labels each group is predicted from other groups only, so the accuracy falls to chance."""
    X, y, groups = _grouped_duplicates()
    preds = fit_baseline_predict_oof(X, y, "binary", seed=0, groups=groups)
    assert float(np.mean((preds > 0.5) == y)) < 0.7


def test_single_group_falls_back_to_the_in_sample_fit():
    """One distinct group cannot be held out, so the baseline fits in-sample instead of raising."""
    X, y, _ = _grouped_duplicates(n_groups=1, rows_per_group=12)
    preds = fit_baseline_predict_oof(X, y, "binary", seed=0, groups=np.zeros(len(y), dtype=int))
    assert preds.shape == (len(y),)


class _GroupedSplitter:
    """Stands in for a ``BoundSplitter``: it only exposes the group labels it was bound with."""

    def __init__(self, groups):
        self.groups = groups


def test_fold_train_groups_slices_the_training_rows_and_passes_none_through():
    """The helper returns the group labels of the training rows, and ``None`` for a splitter without groups."""
    groups = np.array([7, 7, 8, 9])
    np.testing.assert_array_equal(fold_train_groups(_GroupedSplitter(groups), np.array([0, 2, 3])), np.array([7, 8, 9]))
    assert fold_train_groups(object(), np.array([0, 1])) is None


@pytest.mark.parametrize("task", ["regression"])
def test_regression_baseline_accepts_groups(task):
    """The regression path takes the same group-aware split and returns one prediction per row."""
    X, y, groups = _grouped_duplicates()
    preds = fit_baseline_predict_oof(X, y.astype(np.float32), task, seed=0, groups=groups)
    assert preds.shape == (len(y),)
