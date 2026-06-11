"""Regression: a feature-selector pre-step on a requires_encoding strategy must run AFTER cat-encoding.

Fuzz surfaced ``ValueError: could not convert string to float: 'C'`` on the linear leg: ``build_pipeline`` placed the
feature selector (RFECV / RFE / MRMR) FIRST and the CatBoostEncoder AFTER it, so the selector's internal numeric
estimator (LinearRegression) saw raw string categoricals and crashed. The encoder must precede the selector for
strategies that declare ``requires_encoding=True``.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

ce = pytest.importorskip("category_encoders")

from mlframe.training.strategies import LinearModelStrategy


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": pd.Series(rng.choice(["A", "B", "C"], size=n)).astype("category"),
        }
    )
    y = X["num"] + X["cat"].cat.codes * 0.5 + rng.normal(scale=0.1, size=n)
    return X, y


def test_linear_pipeline_encodes_before_feature_selector():
    X, y = _frame()
    strat = LinearModelStrategy()
    pipe = strat.build_pipeline(
        base_pipeline=RFE(LinearRegression(), n_features_to_select=1),
        cat_features=["cat"],
        category_encoder=ce.CatBoostEncoder(random_state=1),
        imputer=SimpleImputer(),
        scaler=StandardScaler(),
    )
    step_names = [s[0] for s in pipe.steps]
    assert "ce" in step_names and "pre" in step_names
    assert step_names.index("ce") < step_names.index("pre"), f"encoder must precede selector: {step_names}"
    # The fit must NOT raise "could not convert string to float".
    pipe.fit(X, y)


def test_linear_pipeline_without_selector_still_encodes():
    X, y = _frame()
    strat = LinearModelStrategy()
    pipe = strat.build_pipeline(
        base_pipeline=None,
        cat_features=["cat"],
        category_encoder=ce.CatBoostEncoder(random_state=1),
        imputer=SimpleImputer(),
        scaler=StandardScaler(),
    )
    pipe.fit(X, y)
    assert "ce" in [s[0] for s in pipe.steps]
