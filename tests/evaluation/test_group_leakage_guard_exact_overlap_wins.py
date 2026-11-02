"""An exact group leak in a later fold is reported even when an earlier fold holds a near-duplicate (tracker 2026-11-02 wave 14, DOC14-6)."""

import numpy as np
import pytest

from mlframe.evaluation.group_leakage_guard import assert_no_group_leakage


def test_exact_overlap_in_fold_one_beats_a_near_duplicate_in_fold_zero():
    """Fold 0: clean groups but identical feature rows across train and test; fold 1: group 5 on both sides. The group-id error must surface."""
    groups = np.array([0, 1, 2, 3, 4, 5, 5, 6])
    features = np.zeros((8, 2))
    splits = [(np.array([0, 1, 2, 3]), np.array([4])), (np.array([5, 0, 1]), np.array([6, 7]))]
    with pytest.raises(ValueError, match="group id"):
        assert_no_group_leakage(splits, groups, near_duplicate_features=features, near_duplicate_metric="euclidean", near_duplicate_max_neighbor_distance=0.5)
