"""Audit 2026-11-20 finding 25-F44: the multilabel coercion validates integer input by range, not by a float64 copy.

The 0/1 rule is unchanged: integer arrays must sit in [0, 1], floats must be exact 0 or 1, and NaN or out-of-range values are
refused. The returned uint8 array is the same as before.
"""

import numpy as np
import pytest

from mlframe.metrics._multilabel_metrics import _coerce_multilabel_array


def test_integer_indicator_is_returned_as_uint8_matrix():
    """A 0/1 int matrix comes back as a contiguous uint8 matrix of the same values."""
    out = _coerce_multilabel_array(np.array([[0, 1], [1, 0]], dtype=np.int64))
    assert out.dtype == np.uint8 and out.tolist() == [[0, 1], [1, 0]]


def test_integer_value_outside_zero_one_is_refused():
    """A class index of 2 in an int column is refused."""
    with pytest.raises(ValueError, match="0/1"):
        _coerce_multilabel_array(np.array([[0, 2]], dtype=np.int64))


def test_float_non_indicator_and_nan_are_refused():
    """A 0.5 and a NaN in float input are both refused, as before."""
    with pytest.raises(ValueError, match="0/1"):
        _coerce_multilabel_array(np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="NaN"):
        _coerce_multilabel_array(np.array([0.0, np.nan]))
