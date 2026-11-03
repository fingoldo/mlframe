"""Multiclass SHAP panels rank a feature by the class it moves most, not by the last class only."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.shap_panels import _shap_values_2d


def test_feature_that_only_separates_the_first_class_is_not_reduced_to_zero():
    """3 classes; feature 0 matters only for class 0 (large |SHAP|), the last class's column is ~0 for it: it used to vanish."""
    rng = np.random.default_rng(0)
    values = np.zeros((50, 2, 3))
    values[:, 0, 0] = rng.normal(scale=2.0, size=50)
    values[:, 1, 2] = rng.normal(scale=0.5, size=50)
    out = _shap_values_2d(values)
    assert out.shape == (50, 2)
    assert np.abs(out[:, 0]).mean() > np.abs(out[:, 1]).mean()


def test_binary_keeps_the_positive_class():
    """Two classes: unchanged, the last column (negative control)."""
    values = np.stack([np.zeros((5, 2)), np.ones((5, 2))], axis=-1)
    np.testing.assert_array_equal(_shap_values_2d(values), np.ones((5, 2)))


def test_legacy_list_of_arrays_is_handled_for_multiclass():
    """The legacy list-of-K layout reduces the same way as the 3-D array."""
    rng = np.random.default_rng(1)
    arrays = [rng.normal(size=(20, 3)) for _ in range(3)]
    assert _shap_values_2d(arrays).shape == (20, 3)
