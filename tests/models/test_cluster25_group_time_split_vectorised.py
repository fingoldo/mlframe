"""Audit 2026-11-20 finding 25-F31: the group-time split groups samples with one sort and gives the same folds as before.

The per-sample Python loop was replaced by a stable argsort. The expected folds were recorded from the loop before the change, for a
string-group and an integer-group fixture, so the test pins the old output exactly.
"""

import numpy as np

from mlframe.models.selection import GroupTimeSeriesSplit


def _folds(groups, n_splits=2):
    """The (train, test) index lists of every fold, as plain Python lists."""
    X = np.zeros((len(groups), 1))
    return [(tr.tolist(), te.tolist()) for tr, te in GroupTimeSeriesSplit(n_splits=n_splits).split(X, groups=groups)]


def test_string_groups_give_the_recorded_folds():
    """String group labels: the folds equal the ones the per-sample loop produced."""
    groups = np.array(["b", "a", "b", "c", "a", "c", "d", "d", "b", "e", "e", "a"])
    assert _folds(groups) == [([0, 1, 2, 3, 4, 5, 8, 11], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7, 8, 11], [9, 10])]


def test_integer_groups_give_the_recorded_folds():
    """Integer group labels in a non-sorted order: the same folds, the first-appearance order of groups unchanged."""
    groups = np.array([3, 1, 3, 2, 1, 2, 0, 0, 3, 4, 4, 1])
    assert _folds(groups) == [([0, 1, 2, 3, 4, 5, 8, 11], [6, 7]), ([0, 1, 2, 3, 4, 5, 6, 7, 8, 11], [9, 10])]
