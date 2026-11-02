"""Val / train slices of a Series sample_weight or group_ids follow row positions, not the Series' index labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training._trainer_train_and_evaluate_parts import _by_position


def test_series_with_shifted_labels_is_sliced_by_position():
    """Labels 100..104 would KeyError (or pick other rows) under label lookup; position 1 and 3 are 0.2 and 0.4."""
    w = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=[100, 101, 102, 103, 104])
    assert _by_position(w, np.array([1, 3])).tolist() == [0.2, 0.4]


def test_series_with_permuted_labels_does_not_return_other_rows():
    """Labels that are a permutation of the positions give different rows under label lookup."""
    g = pd.Series(["a", "b", "c"], index=[2, 0, 1])
    assert _by_position(g, np.array([0, 1])).tolist() == ["a", "b"]


def test_ndarray_is_sliced_as_before():
    """Plain arrays keep their behaviour."""
    assert _by_position(np.array([5, 6, 7]), np.array([2, 0])).tolist() == [7, 5]
