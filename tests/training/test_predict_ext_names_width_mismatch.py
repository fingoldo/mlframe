"""Regression: predict-side extensions pipeline must not crash when get_feature_names_out count != output width.

Fuzz surfaced ``ValueError: Shape of passed values is (N, M), indices imply (N, K)`` (cb/mlp) when a transformer's
``get_feature_names_out`` returned a name count disagreeing with the transformed array width. ``_apply_extensions_pipeline``
now falls back to positional ``ext_<i>`` names on a mismatch (mirroring the train-side ``_build_output_column_names``).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from mlframe.training.core._predict_pre_pipeline import _apply_extensions_pipeline


class _WidthVsNamesMismatch(BaseEstimator, TransformerMixin):
    """Outputs ``n_out`` columns but reports a DIFFERENT number of feature names (the exact fit/predict drift bug)."""

    def __init__(self, n_out=3, n_names=5):
        self.n_out = n_out
        self.n_names = n_names

    def fit(self, X, y=None):
        """Fit."""
        self.feature_names_in_ = np.asarray([str(c) for c in X.columns])
        return self

    def transform(self, X):
        """Transform."""
        return np.tile(np.asarray(X.iloc[:, :1], dtype=float), (1, self.n_out))

    def get_feature_names_out(self, input_features=None):
        """Get feature names out."""
        return np.asarray([f"bad_{i}" for i in range(self.n_names)])


def test_predict_ext_pipeline_survives_name_width_mismatch():
    """Predict ext pipeline survives name width mismatch."""
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    pipe = _WidthVsNamesMismatch(n_out=3, n_names=5).fit(df)
    out = _apply_extensions_pipeline(df, pipe, verbose=0)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (10, 3), out.shape
    assert list(out.columns) == ["ext_0", "ext_1", "ext_2"]


def test_predict_ext_pipeline_keeps_names_when_count_matches():
    """Predict ext pipeline keeps names when count matches."""
    class _Matching(_WidthVsNamesMismatch):
        """Groups tests covering matching."""
        def get_feature_names_out(self, input_features=None):
            """Get feature names out."""
            return np.asarray([f"ok_{i}" for i in range(self.n_out)])

    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)})
    pipe = _Matching(n_out=3, n_names=3).fit(df)
    out = _apply_extensions_pipeline(df, pipe, verbose=0)
    assert list(out.columns) == ["ok_0", "ok_1", "ok_2"]
