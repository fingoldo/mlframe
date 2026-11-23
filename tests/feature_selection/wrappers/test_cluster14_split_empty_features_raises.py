"""Cluster 14 regression: an empty features_indices is a clear ValueError, not an IndexError (audit 2026-11-20, finding 14-F12)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers._helpers import split_into_train_test


def test_empty_features_indices_raises_value_error():
    """Passing an empty list names the problem instead of indexing features_indices[0] (14-F12)."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="features_indices is empty"):
        split_into_train_test(X, y, np.array([0, 1]), np.array([2, 3]), features_indices=[])


def test_none_features_indices_still_splits():
    """Control: None keeps every column (14-F12)."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    y = np.array([0, 1, 0, 1])
    X_tr, _, X_te, _ = split_into_train_test(X, y, np.array([0, 1]), np.array([2, 3]), features_indices=None)
    assert X_tr.shape == (2, 2) and X_te.shape == (2, 2)
