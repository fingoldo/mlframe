"""Cluster 40 regression: the trunk-residual and field-grouped regressors refuse NaN/inf input (audit 2026-11-20, finding 40-F21)."""

import numpy as np
import pytest

pytest.importorskip("torch")
from mlframe.training.neural.field_grouped_mlp import FieldGroupedMLPRegressor
from mlframe.training.neural.trunk_residual_mlp import TrunkResidualMLPRegressor


def _data(nan=False):
    """Synthetic regression frame with an optional NaN cell."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    if nan:
        X[3, 1] = np.nan
    return X, X[:, 0] + 0.1 * rng.normal(size=40)


def test_trunk_regressor_refuses_nan_at_fit():
    """A NaN cell in X is refused at fit, not silently turned into NaN weights (40-F21)."""
    X, y = _data(nan=True)
    with pytest.raises(ValueError, match="NaN or inf"):
        TrunkResidualMLPRegressor(n_epochs=2).fit(X, y)


def test_field_grouped_regressor_refuses_nan_at_fit():
    """Field-grouped regressor refuses NaN at fit too (40-F21)."""
    X, y = _data(nan=True)
    groups = {"a": [0, 1], "b": [2, 3]}
    with pytest.raises(ValueError, match="NaN or inf"):
        FieldGroupedMLPRegressor(field_groups=groups, n_epochs=2).fit(X, y)


def test_field_grouped_regressor_refuses_nan_at_predict():
    """predict with a NaN cell raises instead of returning NaN predictions (40-F21)."""
    X, y = _data()
    groups = {"a": [0, 1], "b": [2, 3]}
    m = FieldGroupedMLPRegressor(field_groups=groups, n_epochs=2).fit(X, y)
    Xn = X.copy()
    Xn[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        m.predict(Xn)
