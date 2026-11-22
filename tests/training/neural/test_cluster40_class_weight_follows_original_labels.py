"""Regression tests for the audit findings in this module."""
import pytest
"""Audit 2026-11-20 finding 40-F5: class_weight keyed by original labels maps to the encoded positions through classes_."""

import numpy as np

pytest.importorskip("lightning")
from mlframe.training.neural._recurrent_wrapper_base import _class_weight_vector


def test_weights_follow_the_original_labels_not_their_positions():
    """Labels {3, 7} encoded to positions 0, 1: the weight for label 7 lands at position 1, not at the label-3 position."""
    assert _class_weight_vector({3: 5.0, 7: 1.0}, np.array([3, 7]), 2) == [5.0, 1.0]
    assert _class_weight_vector({3: 5.0, 7: 1.0}, np.array([7, 3]), 2) == [1.0, 5.0]


def test_without_classes_the_positions_are_the_labels():
    """No classes_ means labels are already 0..k-1, so the dict is read by position as before."""
    assert _class_weight_vector({0: 2.0, 1: 3.0}, None, 2) == [2.0, 3.0]


def test_missing_label_defaults_to_one():
    """A label absent from the dict keeps weight 1.0."""
    assert _class_weight_vector({3: 5.0}, np.array([3, 7]), 2) == [5.0, 1.0]
