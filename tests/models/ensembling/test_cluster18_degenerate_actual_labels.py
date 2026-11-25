"""Cluster 18 regression: the DEGENERATE marker judges the actual label values (audit 2026-11-20, finding 18-F7).

Labels {0, 2} used to count as one class, so every confident subset was tagged. The marker now counts the real values: a single label is
degenerate, two labels are judged by their ratio, and a multiclass subset is not tagged by this binary rule.
"""

import numpy as np

from mlframe.models.ensembling.process_method import _flag_degenerate_confident_subset


def _mark(target, ratio=0.05):
    """Run the degenerate check on one subset target and return the marker it leaves."""
    return _flag_degenerate_confident_subset(True, False, np.asarray(target), ratio, "")


def test_labels_zero_and_two_are_not_tagged_when_balanced():
    """A balanced {0, 2} subset is not degenerate, even though 1 never occurs (18-F7)."""
    assert _mark([0, 2, 0, 2, 0, 2]) == ""


def test_single_label_is_tagged():
    """A subset with one label is degenerate (18-F7)."""
    assert _mark([1, 1, 1, 1]) == "[DEGENERATE] "


def test_multiclass_subset_is_not_tagged_by_the_binary_rule():
    """A three-class subset with a rare class 1 is not tagged by this binary rule (18-F7)."""
    target = [0] * 50 + [1] + [2] * 49
    assert _mark(target) == ""


def test_imbalanced_binary_is_tagged_by_ratio():
    """Two labels with a ratio under the threshold are tagged (18-F7)."""
    assert _mark([0] * 200 + [1]) == "[DEGENERATE] "
