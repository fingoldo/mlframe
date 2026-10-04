"""Hard labels derived from probabilities must not wrap for wide multiclass problems."""

from __future__ import annotations

import numpy as np

from mlframe.inference.explainability import _hard_labels_from_probs


def test_hard_labels_do_not_wrap_beyond_127_classes() -> None:
    """With 200 classes the argmax of the last class is 199, not a negative int8 wrap-around."""
    probs = np.zeros((3, 200))
    probs[0, 199] = 1.0
    probs[1, 130] = 1.0
    probs[2, 5] = 1.0
    assert _hard_labels_from_probs(probs, 200).tolist() == [199, 130, 5]


def test_hard_labels_binary_thresholds_positive_column() -> None:
    """Binary probabilities are thresholded at 0.5 on the positive-class column."""
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    assert _hard_labels_from_probs(probs, 2).tolist() == [0, 1]
