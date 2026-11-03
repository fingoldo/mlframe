"""LightGBM models with categorical or zero-as-missing splits are not eligible for the flat TreeSHAP path (it cannot reproduce their routing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import is_supported_lightgbm


def _data(n=400):
    """Numeric feature a, categorical feature c, binary-ish regression target depending on both."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=n), "c": pd.Categorical(rng.integers(0, 5, size=n))})
    y = df["a"].to_numpy() + (df["c"].astype(int).to_numpy() == 3) * 2.0 + rng.normal(scale=0.1, size=n)
    return df, y


def test_model_with_a_categorical_split_is_not_supported():
    """The categorical feature is split with ==; the extractor read its category-list threshold as a number."""
    df, y = _data()
    model = lgb.LGBMRegressor(n_estimators=10, min_child_samples=5, verbose=-1).fit(df, y)
    assert is_supported_lightgbm(model) is False


def test_all_numeric_model_stays_supported():
    """Numeric-only model (negative control)."""
    df, y = _data()
    model = lgb.LGBMRegressor(n_estimators=10, verbose=-1).fit(df[["a"]], y)
    assert is_supported_lightgbm(model) is True


def test_zero_as_missing_model_is_not_supported():
    """zero_as_missing routes exact zeros like NaN, which the flat kernels do not do."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 2))
    X[rng.random(400) < 0.4, 0] = 0.0
    y = X[:, 0] + X[:, 1]
    model = lgb.LGBMRegressor(n_estimators=10, zero_as_missing=True, verbose=-1).fit(X, y)
    assert is_supported_lightgbm(model) is False
