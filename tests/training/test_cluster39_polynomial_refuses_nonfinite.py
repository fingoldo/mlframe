"""Audit 2026-11-20 finding 39-F37: polynomial expansion refuses NaN/inf input with a message that names the fix."""

import numpy as np
import pytest

from mlframe.training.feature_handling.polynomial import PolynomialFeatureExpander


def test_fit_refuses_nan_input_with_a_clear_message():
    """NaN in the fit input raises ValueError naming the non-finite count and the remedy."""
    X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
    with pytest.raises(ValueError, match="non-finite values; impute"):
        PolynomialFeatureExpander(degree=2, prefer_polarsds=False).fit(X)


def test_transform_refuses_nan_input_after_a_clean_fit():
    """A clean fit followed by a NaN transform raises the same refusal."""
    expander = PolynomialFeatureExpander(degree=2, prefer_polarsds=False).fit(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
    with pytest.raises(ValueError, match="non-finite values"):
        expander.transform(np.array([[np.inf, 1.0]]))


def test_clean_input_still_expands():
    """Finite input is unaffected: a degree-2 expansion of 2 columns gives the expected 5 output columns (no bias)."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out = PolynomialFeatureExpander(degree=2, prefer_polarsds=False).fit(X).transform(X)
    assert np.asarray(out).shape[1] == 5
