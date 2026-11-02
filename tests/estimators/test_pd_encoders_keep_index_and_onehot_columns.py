"""Pandas wrappers keep the caller's index and name the one-hot columns (tracker 2026-11-02 wave 10, ALIGN10-3, ALIGN10-4)."""

import numpy as np
import pandas as pd

from mlframe.estimators.custom import PdKBinsDiscretizer, PdOrdinalEncoder


def _frame():
    """Two numeric columns under a non-default index."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60)}, index=np.arange(1000, 1060))


def test_default_onehot_kbins_on_a_dataframe_does_not_crash_and_keeps_index():
    """encode='onehot' yields n_features*n_bins columns; the frame used n_features names and raised a shape error."""
    X = _frame()
    out = PdKBinsDiscretizer(n_bins=5).fit_transform(X)
    assert out.shape == (60, 10)
    assert list(out.index) == list(X.index)


def test_ordinal_encoder_keeps_the_index():
    """The encoded frame lines up with the input rows under concat."""
    X = pd.DataFrame({"c": ["x", "y", "x", "z"]}, index=[10, 20, 30, 40])
    out = PdOrdinalEncoder().fit_transform(X)
    assert list(out.index) == [10, 20, 30, 40]
