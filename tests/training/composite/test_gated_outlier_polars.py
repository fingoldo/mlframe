"""TRAINING_COMPOSITE_CORE_A-2 (2026-08-05 audit): GatedOutlierEstimator.fit's row-subsetting fell back to
``np.asarray(X)[mask]`` for any X lacking ``.loc`` (e.g. a polars DataFrame), silently flattening it to a
raw/object-dtype ndarray before fitting the regressor -- breaking feature-name and dtype consistency, unlike
sibling composite modules (bagging.py, classification.py, glm.py) which handle polars explicitly. Fixed by
adding a flavour-native ``_subset_rows`` helper (polars ``.filter`` / pandas ``.loc`` / ndarray boolean
index), matching ``bagging.py``'s existing ``_take_rows`` pattern.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.training.composite import GatedOutlierEstimator


def test_gated_outlier_fit_on_polars_df_keeps_it_a_polars_frame_for_the_regressor():
    """The regressor must receive a real polars DataFrame (feature names + dtypes intact), not a
    flattened object-dtype ndarray -- verified by monkeypatching the regressor's .fit to capture its
    actual X argument's type."""
    rng = np.random.default_rng(0)
    n = 200
    f0 = rng.standard_normal(n)
    f1 = rng.standard_normal(n)
    y = np.where(rng.random(n) < 0.4, 0.0, f0 * 2.0 + rng.standard_normal(n))
    X = pl.DataFrame({"f0": f0, "f1": f1})

    captured = {}

    class _CapturingRegressor(LinearRegression):
        """LinearRegression subclass that records the type of X it was actually fit on."""

        def fit(self, X, y, sample_weight=None):
            """Record X's type/columns before delegating to the real LinearRegression.fit."""
            captured["X_type"] = type(X)
            captured["X_columns"] = list(X.columns) if hasattr(X, "columns") else None
            return super().fit(np.asarray(X), y, sample_weight=sample_weight)

    est = GatedOutlierEstimator(regressor=_CapturingRegressor(), classifier=LogisticRegression(max_iter=1000))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        est.fit(X, y)

    assert captured["X_type"] is pl.DataFrame, f"regressor must be fit on a real polars DataFrame, got {captured['X_type']}"
    assert captured["X_columns"] == ["f0", "f1"]


def test_gated_outlier_polars_predict_matches_pandas_equivalent():
    """Sanity: fitting on an equivalent polars vs. pandas frame produces the same predictions (the fix
    is a plumbing change, not a behavior change)."""
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 150
    f0 = rng.standard_normal(n)
    y = np.where(rng.random(n) < 0.3, 0.0, f0 * 1.5 + 0.1 * rng.standard_normal(n))
    data = {"f0": f0}

    X_pl = pl.DataFrame(data)
    X_pd = pd.DataFrame(data)

    preds = []
    for X in (X_pl, X_pd):
        est = GatedOutlierEstimator(regressor=LinearRegression(), classifier=LogisticRegression(max_iter=1000))
        est.fit(X, y)
        preds.append(est.predict(X))

    assert np.allclose(preds[0], preds[1], atol=1e-8)
