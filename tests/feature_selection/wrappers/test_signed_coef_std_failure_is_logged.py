"""Unscaled signed coefficients are reported, not mixed in silently with scaled runs (tracker 2026-11-02 wave 9, EXC9-3)."""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.feature_selection.wrappers._helpers_importance_agg import get_signed_linear_coef


def _model():
    """A fitted linear model on two features."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    return LinearRegression().fit(X, X[:, 0])


def test_std_length_mismatch_warns(caplog):
    """Train data with three columns against two coefficients leaves the coefficients unscaled and says so."""
    train = pd.DataFrame(np.ones((10, 3)))
    with caplog.at_level(logging.WARNING):
        out = get_signed_linear_coef(_model(), ["a", "b"], train_data=train)
    assert out is not None
    assert any("unscaled" in r.getMessage() for r in caplog.records)
