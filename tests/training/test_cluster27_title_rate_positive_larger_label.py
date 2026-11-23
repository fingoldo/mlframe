"""Cluster 27 regression: the title rate counts the larger label as positive, matching the decision-threshold block (audit 2026-11-20, finding 27-F22)."""

import numpy as np

from mlframe.training._eval_helpers import _append_split_rate_suffix


def test_labels_one_and_two_report_the_larger_label_rate():
    """With labels {1, 2} the positive share is the share of label 2, not 0% (27-F22)."""
    arr = np.array([1, 2, 2, 2])
    out = _append_split_rate_suffix("BTTR=50%", split_name="test", target=arr)
    assert "75%" in out


def test_zero_one_labels_unchanged():
    """Control: {0, 1} labels keep the positive-share title (27-F22)."""
    arr = np.array([0, 1, 1, 0])
    out = _append_split_rate_suffix("BTTR=50%", split_name="test", target=arr)
    assert "50%" in out
