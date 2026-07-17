"""Unit tests for ``MissingAwareComposite`` (NaN-in-base robustness)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.composite.missing import MissingAwareComposite


def _make_composite() -> CompositeTargetEstimator:
    return CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_column="base",
    )


def _frame(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(10.0, 3.0, size=n)
    feat = rng.normal(0.0, 1.0, size=n)
    y = base + 2.0 * feat + rng.normal(0.0, 0.3, size=n)
    X = pd.DataFrame({"base": base, "feat": feat})
    return X, y


def test_nan_base_yields_finite_predict():
    X, y = _frame(300)
    est = MissingAwareComposite(composite=_make_composite()).fit(X, y)
    X_pred = X.copy()
    X_pred.loc[:20, "base"] = np.nan
    pred = est.predict(X_pred)
    assert pred.shape == (len(X_pred),)
    assert np.all(np.isfinite(pred)), "NaN base must never produce NaN prediction"


def test_missing_indicator_learned_train_only():
    X, y = _frame(300, seed=1)
    # Inject MNAR missingness in TRAIN: high-y rows get NaN base.
    thr = np.quantile(y, 0.8)
    X_train = X.copy()
    X_train.loc[y >= thr, "base"] = np.nan
    est = MissingAwareComposite(composite=_make_composite()).fit(X_train, y)
    assert est.missing_indicator_learned_ is True
    assert est.use_offset_ is True
    assert np.isfinite(est.base_impute_value_)
    # Offset should be positive: missing rows were the HIGH-y rows.
    assert est.missing_offset_ > 0.0
    # Impute value learned only from finite base rows (the low-y rows).
    finite_base = X_train["base"].to_numpy()
    finite_base = finite_base[np.isfinite(finite_base)]
    assert est.base_impute_value_ == pytest.approx(float(np.median(finite_base)))


def test_all_missing_base_fallback():
    X, y = _frame(200, seed=2)
    X_all_nan = X.copy()
    X_all_nan["base"] = np.nan
    est = MissingAwareComposite(composite=_make_composite()).fit(X_all_nan, y)
    # No finite base at fit -> offset path disabled, median fallback active.
    assert est.use_offset_ is False
    pred = est.predict(X_all_nan)
    assert np.all(np.isfinite(pred))
    assert np.allclose(pred, est.y_train_median_)


def test_too_many_missing_triggers_median_fallback():
    X, y = _frame(300, seed=5)
    X_train = X.copy()
    # 60% missing > default max_missing_frac=0.5 -> median fallback.
    miss = np.zeros(len(X), dtype=bool)
    miss[: int(0.6 * len(X))] = True
    X_train.loc[miss, "base"] = np.nan
    est = MissingAwareComposite(composite=_make_composite(), max_missing_frac=0.5).fit(X_train, y)
    assert est.use_offset_ is False
    pred = est.predict(X_train)
    assert np.allclose(pred[miss], est.y_train_median_)


def test_no_missing_identical_to_plain_composite():
    X, y = _frame(300, seed=3)
    plain = _make_composite().fit(X, y)
    wrapped = MissingAwareComposite(composite=_make_composite()).fit(X, y)
    p_plain = plain.predict(X)
    p_wrapped = wrapped.predict(X)
    assert np.allclose(p_plain, p_wrapped), "no-missing path must match plain composite"


def test_polars_frame_supported():
    pl = pytest.importorskip("polars")
    X, y = _frame(200, seed=7)
    Xpl = pl.from_pandas(X)
    est = MissingAwareComposite(composite=_make_composite()).fit(Xpl, y)
    Xpred = X.copy()
    Xpred.loc[:10, "base"] = np.nan
    pred = est.predict(pl.from_pandas(Xpred))
    assert np.all(np.isfinite(pred))


def test_multi_base_rejected():
    comp = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_columns=("a", "b"),
    )
    X, y = _frame(50)
    with pytest.raises(ValueError, match="multi-base"):
        MissingAwareComposite(composite=comp).fit(X.assign(a=1.0, b=2.0), y)
