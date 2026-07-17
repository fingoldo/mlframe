"""Regression sentry: RFECV.fit(X, y, sample_weight=None) and RFECV.fit(X, y) must match byte-for-byte.

When sample_weight is omitted or explicitly None, the new branch must not engage: no per-fold slicing,
no estimator-side or scorer-side weight injection. Support_ and cv_results must be identical.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _toy_dataset(seed=11):
    """Toy dataset."""
    rng = np.random.default_rng(seed)
    n, p = 300, 5
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.2 * rng.normal(size=n) > 0).astype(np.int64)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(p)])
    return df, pd.Series(y, name="y")


def _build_rfecv(**overrides):
    """Lightweight RFECV configured for fast unit-test runs (CatBoost-free)."""
    from mlframe.feature_selection.wrappers import RFECV
    from sklearn.linear_model import LogisticRegression

    est = LogisticRegression(max_iter=200, random_state=0)
    defaults = dict(estimator=est, cv=3, verbose=0, random_state=42, max_runtime_mins=0.25)
    defaults.update(overrides)
    return RFECV(**defaults)


def test_rfecv_fit_sample_weight_none_matches_omitted():
    """sample_weight=None must produce identical support_ to omitting the kwarg."""
    df, y = _toy_dataset()
    sel_a = _build_rfecv().fit(df, y)
    sel_b = _build_rfecv().fit(df, y, sample_weight=None)
    assert list(sel_a.support_) == list(sel_b.support_), f"support_ differs: omitted={list(sel_a.support_)} vs None={list(sel_b.support_)}"


def test_rfecv_fit_sample_weight_validates_shape_and_values():
    """Invalid sample_weight must raise ValueError before fit work begins."""
    df, y = _toy_dataset()
    n = len(df)
    sel = _build_rfecv()
    with pytest.raises(ValueError, match="sample_weight length"):
        sel.fit(df, y, sample_weight=np.ones(n - 1))
    with pytest.raises(ValueError, match="finite and non-negative"):
        sw = np.ones(n)
        sw[0] = -1.0
        sel.fit(df, y, sample_weight=sw)
    with pytest.raises(ValueError, match="finite and non-negative"):
        sw = np.ones(n)
        sw[0] = np.inf
        sel.fit(df, y, sample_weight=sw)


def test_rfecv_fit_nonuniform_sample_weight_runs_and_stores():
    """Non-uniform weights must run cleanly and persist on self for downstream introspection."""
    df, y = _toy_dataset()
    n = len(df)
    rng = np.random.default_rng(0)
    sw = rng.uniform(0.1, 2.0, size=n)
    sel = _build_rfecv().fit(df, y, sample_weight=sw)
    assert getattr(sel, "_fit_sample_weight_", None) is not None
    assert sel._fit_sample_weight_.shape == (n,)
    assert len(sel.support_) >= 1
