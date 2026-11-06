"""The trunk-residual and multitask regressors record the fitted width and refuse a different one at predict.

A narrower input used to reach the network and fail with a torch shape error, or with an index error, instead of a clear message (audit API3-7
and API3-8).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.multitask_auxiliary_loss import MultiTaskAuxiliaryLossRegressor
from mlframe.training.neural.trunk_residual_mlp import TrunkResidualMLPRegressor


def _data():
    """Three features and a target that depends on them."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    return X, X @ np.array([1.0, -0.5, 0.25]) + 0.1 * rng.normal(size=80)


def test_trunk_regressor_records_width_and_refuses_a_narrower_input():
    """Fitted on three features, predicting on two raises ValueError."""
    X, y = _data()
    est = TrunkResidualMLPRegressor(n_epochs=5).fit(X, y)
    assert est.n_features_in_ == 3
    with pytest.raises(ValueError, match="features"):
        est.predict_ensemble_mean(X[:, :2])


def test_multitask_regressor_refuses_a_narrower_input():
    """Fitted on three features, predicting on two raises ValueError."""
    X, y = _data()
    est = MultiTaskAuxiliaryLossRegressor(n_epochs=5).fit(X, y)
    assert est.n_features_in_ == 3
    with pytest.raises(ValueError, match="features"):
        est.predict(X[:, :2])


def test_tabular_1dcnn_regressor_refuses_a_different_width():
    """Fitted on three features, predicting on two or on four raises ValueError instead of cutting or failing inside the network (API3-5)."""
    from mlframe.training.neural.tabular_1dcnn import Tabular1DCNNRegressor

    X, y = _data()
    est = Tabular1DCNNRegressor(n_epochs=2).fit(X, y)
    assert est.n_features_in_ == 3
    with pytest.raises(ValueError, match="features"):
        est.predict(X[:, :2])
    with pytest.raises(ValueError, match="features"):
        est.predict(np.hstack([X, X[:, :1]]))
