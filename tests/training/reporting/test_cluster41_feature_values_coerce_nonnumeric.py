"""Cluster 41 regression: non-numeric feature values coerce to NaN instead of raising (audit 2026-11-20, finding 41-F16)."""

import numpy as np

from mlframe.training.reporting._reporting_probabilistic_calib import _numeric_feature_values


def test_categorical_values_become_nan_not_an_error():
    """A string (categorical) entry becomes NaN; numeric entries keep their value (41-F16)."""
    out = _numeric_feature_values(np.array(["1.5", "low", 2.0], dtype=object))
    assert out[0] == 1.5 and out[2] == 2.0
    assert np.isnan(out[1])


def test_numeric_values_are_unchanged():
    """Control: a numeric column keeps its values and dtype float64."""
    out = _numeric_feature_values(np.array([1, 2, 3]))
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0]
