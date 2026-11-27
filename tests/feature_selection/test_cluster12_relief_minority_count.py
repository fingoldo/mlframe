"""Cluster 12 regression: the ReliefF neighbour count comes from the rarest class that occurs, not from an empty one (audit 2026-11-20, 12-F2)."""

import numpy as np

from mlframe.feature_selection._benchmarks.fs_hybrid.fs_selectors import _minority_class_count


def test_labels_one_and_two_do_not_pick_the_absent_class_zero():
    """Balanced {1, 2} labels give 500, not the 0 of the empty class that argmin of bincount selected (12-F2)."""
    y = np.array([1] * 500 + [2] * 500)
    assert _minority_class_count(y) == 500


def test_imbalanced_and_negative_labels():
    """The rarest present class is counted whatever the label values are (12-F2)."""
    assert _minority_class_count(np.array([0] * 90 + [1] * 10)) == 10
    assert _minority_class_count(np.array([-1] * 7 + [3] * 20 + [9] * 11)) == 7
