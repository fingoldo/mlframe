"""rf_outlier_measure treats distinct float or string labels as distinct classes."""

from __future__ import annotations

import numpy as np

from mlframe.models.rf_proximity import rf_outlier_measure


def _prox(n=40):
    """Block proximity: two groups of equal size, strong inside and weak across."""
    p = np.full((n, n), 0.05)
    p[: n // 2, : n // 2] = 0.8
    p[n // 2 :, n // 2 :] = 0.8
    np.fill_diagonal(p, 1.0)
    return p


def test_float_labels_are_not_merged_by_integer_truncation():
    """Labels 0.2 and 0.7 both truncate to 0 under an int cast; they must score like the equivalent integer classes."""
    n = 40
    ints = np.repeat([0, 1], n // 2)
    floats = np.repeat([0.2, 0.7], n // 2)
    np.testing.assert_allclose(rf_outlier_measure(_prox(n), floats), rf_outlier_measure(_prox(n), ints))


def test_string_labels_are_accepted():
    """String class labels work instead of raising."""
    n = 40
    ints = np.repeat([0, 1], n // 2)
    strs = np.repeat(["a", "b"], n // 2)
    np.testing.assert_allclose(rf_outlier_measure(_prox(n), strs), rf_outlier_measure(_prox(n), ints))
