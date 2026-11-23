"""Cluster 41 regression: a numeric transform that drops a column and exposes no names fails loudly, not by positional guessing (audit 2026-11-20, finding 41-F18)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from mlframe.training.strategies.base import _NumericOnlyTransformer


class _DropsOneColumnNoNames(BaseEstimator, TransformerMixin):
    """Inner transformer that drops the first column and has no get_feature_names_out (41-F18)."""

    def fit(self, X, y=None):
        """No state to learn."""
        return self

    def transform(self, X):
        """Return the frame without its first column as a bare array."""
        return np.asarray(X)[:, 1:]


def test_dropped_column_without_names_raises_instead_of_misassigning():
    """A positional guess would write values into the wrong named columns; the transform now raises (41-F18)."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
    t = _NumericOnlyTransformer(inner=_DropsOneColumnNoNames(), cat_features=["c"])
    t.fit(X)
    with pytest.raises(ValueError, match="changed width"):
        t.transform(X)
