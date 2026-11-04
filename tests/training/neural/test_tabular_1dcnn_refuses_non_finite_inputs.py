"""Tabular1DCNN refuses NaN/inf inputs instead of training to NaN weights."""
import numpy as np
import pytest

from mlframe.training.neural.tabular_1dcnn import Tabular1DCNNRegressor


def test_one_nan_cell_raises():
    """A single NaN made every weight and prediction NaN with no error."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 6)).astype(np.float32)
    X[3, 2] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        Tabular1DCNNRegressor(n_epochs=2).fit(X, rng.standard_normal(50))
