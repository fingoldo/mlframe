"""TRAINING_COMPOSITE_CORE_A-2b (2026-08-05 audit): GatedRegressionMixture.fit's / ._predict_branch's
row-subsetting fell back to ``np.asarray(X)[mask]`` for any X lacking ``.iloc`` (e.g. a polars DataFrame),
silently down-converting it to an untyped/object ndarray before fitting/predicting a branch regressor --
the same bug class as TRAINING_COMPOSITE_CORE_A-2's ``gated_outlier.py``, and directly contradicted by
this file's own ``_concat_feature`` a few lines away, which DOES handle polars explicitly. Fixed by adding
a flavour-native ``_subset_rows`` helper (polars ``DataFrame[idx]`` / pandas ``.iloc`` / ndarray fancy
index).
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
from sklearn.linear_model import LinearRegression, LogisticRegression

from mlframe.training.composite import GatedRegressionMixture


def test_gated_regression_mixture_fit_on_polars_df_keeps_it_a_polars_frame_for_branch_regressors():
    """Both branch regressors must receive a real polars DataFrame (feature names + dtypes intact),
    not a flattened object-dtype ndarray -- verified by capturing each branch regressor's actual X."""
    rng = np.random.default_rng(0)
    n = 300
    f0 = rng.standard_normal(n)
    f1 = rng.standard_normal(n)
    y = f0 * 2.0 + rng.standard_normal(n)
    subpop_label = (rng.random(n) < 0.4).astype(int)
    X = pl.DataFrame({"f0": f0, "f1": f1})

    captured = {"low": None, "high": None}

    def _make_capturing_regressor(branch_name):
        """Build a LinearRegression subclass instance that records its own branch's fit-time X type."""

        class _CapturingRegressor(LinearRegression):
            """LinearRegression subclass that records the type of X it was actually fit on."""

            def fit(self, X, y, sample_weight=None):
                """Record X's type before delegating to the real LinearRegression.fit."""
                captured[branch_name] = type(X)
                return super().fit(np.asarray(X), y, sample_weight=sample_weight)

        return _CapturingRegressor()

    est = GatedRegressionMixture(
        gate_classifier=LogisticRegression(max_iter=1000),
        low_regressor=_make_capturing_regressor("low"),
        high_regressor=_make_capturing_regressor("high"),
        n_splits=3,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        est.fit(X, y, subpop_label=subpop_label)

    assert captured["low"] is pl.DataFrame, f"low branch regressor must be fit on a real polars DataFrame, got {captured['low']}"
    assert captured["high"] is pl.DataFrame, f"high branch regressor must be fit on a real polars DataFrame, got {captured['high']}"


def test_gated_regression_mixture_polars_predict_matches_pandas_equivalent():
    """Sanity: fitting/predicting on an equivalent polars vs. pandas frame produces the same predictions
    (the fix is a plumbing change, not a behavior change)."""
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 250
    f0 = rng.standard_normal(n)
    y = f0 * 1.5 + 0.1 * rng.standard_normal(n)
    subpop_label = (rng.random(n) < 0.35).astype(int)
    data = {"f0": f0}

    X_pl = pl.DataFrame(data)
    X_pd = pd.DataFrame(data)

    preds = []
    for X in (X_pl, X_pd):
        est = GatedRegressionMixture(
            gate_classifier=LogisticRegression(max_iter=1000, random_state=0),
            low_regressor=LinearRegression(),
            high_regressor=LinearRegression(),
            n_splits=3,
            random_state=0,
        )
        est.fit(X, y, subpop_label=subpop_label)
        preds.append(est.predict(X))

    assert np.allclose(preds[0], preds[1], atol=1e-6)
