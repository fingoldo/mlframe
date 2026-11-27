"""Cluster 14 regression: the auto-tune fingerprint no longer crashes on a non-pandas frame with a string column (audit 2026-11-20, 14-F3)."""

import numpy as np

from mlframe.feature_selection.wrappers._auto_tune import DataFingerprint, _float_columns


def test_float_columns_casts_numbers_and_factorises_strings():
    """Numeric-looking columns are cast, string columns become codes, missing entries stay NaN (14-F3)."""
    arr = np.array([["1.5", "a"], ["2.5", "b"], ["3.5", "a"], [None, None]], dtype=object)
    out = _float_columns(arr)
    assert out.dtype == np.float64
    assert out[0, 0] == 1.5
    assert np.isnan(out[3, 0])
    assert out[0, 1] == out[2, 1] != out[1, 1]
    assert np.isnan(out[3, 1])


def test_fingerprint_of_an_object_array_with_a_string_column_does_not_raise():
    """DataFingerprint.from_xy on an object matrix holding a text column returns a fingerprint with the NaN flag set (14-F3)."""
    rng = np.random.default_rng(0)
    n = 60
    X = np.empty((n, 3), dtype=object)
    X[:, 0] = rng.normal(size=n)
    X[:, 1] = rng.choice(["u", "v", "w"], size=n)
    X[:, 2] = rng.normal(size=n)
    X[3, 2] = np.nan
    y = rng.normal(size=n)
    fp = DataFingerprint.from_xy(X, y)
    assert fp.n_rows == n and fp.n_features == 3
    assert fp.has_nan is True
