"""Audit 2026-11-20 finding 40-F7: preparing a dataset does not scale the caller's feature array in place."""

from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import StandardScaler

from mlframe.training.neural._recurrent_wrappers import RecurrentClassifierWrapper


def test_scaling_with_a_categorical_block_leaves_the_caller_array_unchanged():
    """Column 0 is a categorical code (passed through); column 1 is numeric and scaled. The caller's array keeps its values."""
    wrapper = RecurrentClassifierWrapper.__new__(RecurrentClassifierWrapper)
    wrapper._cat_cardinalities_ = [3]
    wrapper._n_cat_features_ = 1
    scaler = StandardScaler().fit(np.array([[1.0], [3.0], [5.0]]))  # fitted on the numeric block only
    wrapper._feature_scaler = scaler
    wrapper._cfg_resolved = SimpleNamespace(sequence_preprocessing="none")
    wrapper._is_regression = False
    features = np.array([[0, 1.0], [1, 3.0], [2, 5.0]], dtype=np.float32)
    original = features.copy()
    wrapper._create_dataset(None, features, np.array([0, 1, 0]))
    np.testing.assert_array_equal(features, original)
