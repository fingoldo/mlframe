"""Cluster 39 regression: a custom handler refuses ndarray input instead of passing every column (audit 2026-11-20, finding 39-F23)."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.feature_handling.custom_handler import CustomHandler, CustomParams


class _Identity:
    """Minimal sklearn-style transformer that returns its input unchanged."""

    def fit(self, X, y=None):
        """Accept and ignore the data."""
        return self

    def transform(self, X):
        """Return the input as given."""
        return X


def test_ndarray_input_is_refused_not_passed_whole():
    """A plain ndarray has no column names, so the configured column cannot be selected: raise TypeError (39-F23)."""
    handler = CustomHandler("a", CustomParams(transformer=_Identity()))
    with pytest.raises(TypeError, match="DataFrame"):
        handler._extract_column(np.arange(12).reshape(4, 3))


def test_pandas_input_still_selects_the_named_column():
    """Control: the named column is selected from a pandas frame (39-F23)."""
    handler = CustomHandler("a", CustomParams(transformer=_Identity()))
    out = handler._extract_column(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert out.shape == (3, 1)
    assert out[:, 0].tolist() == [1, 2, 3]


def test_polars_input_still_selects_the_named_column():
    """Control: the named column is selected from a polars frame (39-F23)."""
    handler = CustomHandler("b", CustomParams(transformer=_Identity()))
    out = handler._extract_column(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert out[:, 0].tolist() == [4, 5, 6]
