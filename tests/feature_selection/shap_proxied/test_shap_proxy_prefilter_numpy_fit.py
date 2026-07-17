"""Regression: the prefilter ranking boosters fit on a NAMELESS numpy array (speed lever
2026-06-08) instead of a named pandas DataFrame, because they consume only ``feature_importances_``
(a positional vector). Pin the bit-identity invariant: numpy-fit ranking == named-DataFrame-fit
ranking (same importances, same working_cols / stage-B selection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

xgb = pytest.importorskip("xgboost")

from sklearn.base import clone

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import (
    _unwrap_estimator,
    make_default_estimator,
)
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import (
    _as_nameless_array,
    _importances_from_fitted,
    _rank_model,
    _rank_two_stage,
)


def _make_xy(n=2500, p=60, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"col_{i}" for i in range(p)])
    logit = 1.3 * X.iloc[:, 2] - 0.9 * X.iloc[:, 7] + 0.5 * X.iloc[:, 15]
    y = (logit + 0.5 * rng.normal(size=n) > 0).astype(int).to_numpy()
    return X, y


def test_as_nameless_array_passthrough():
    X, _ = _make_xy()
    a = _as_nameless_array(X)
    b = _as_nameless_array(X.to_numpy())
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    assert np.array_equal(a, b)
    assert np.array_equal(a, X.to_numpy())


def test_rank_model_numpy_bit_identical_to_named_dataframe():
    X, y = _make_xy()
    model = make_default_estimator(True, random_state=0)
    # Lever path (numpy via _rank_model).
    imp_lever = _rank_model(model, X, y, n_features=X.shape[1], n_estimators_cap=100)
    # Reference: pre-lever named-DataFrame fit.
    pf = clone(model)
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_prefilter import _apply_booster_cap

    _apply_booster_cap(pf, 100)
    pf.fit(X, y)
    imp_ref = _importances_from_fitted(_unwrap_estimator(pf), X.shape[1])
    assert np.array_equal(imp_lever, imp_ref), "rank_model numpy importances differ from named-DF"


def test_two_stage_working_cols_numpy_bit_identical():
    X, y = _make_xy()
    model = make_default_estimator(True, random_state=0)
    wc_df, _ = _rank_two_stage(model, X, y, n_features=X.shape[1], classification=True, prefilter_top=20, stage1_keep=40, n_estimators_cap=100)
    wc_np, _ = _rank_two_stage(model, X.to_numpy(), y, n_features=X.shape[1], classification=True, prefilter_top=20, stage1_keep=40, n_estimators_cap=100)
    assert np.array_equal(wc_df, wc_np), "two_stage working_cols differ DataFrame vs ndarray input"
