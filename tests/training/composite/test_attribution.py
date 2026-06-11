"""Unit + biz_value tests for composite-target base-vs-residual attribution.

``explain_prediction`` decomposes each composite prediction into a BASE
contribution (what the transform recovers from the base column alone) and a
RESIDUAL contribution (the inner model on the y-scale). For the additive /
linear-residual family the two SUM to ``y_hat``; for the multiplicative family
(ratio / logratio) they MULTIPLY. ``attribution_summary`` reports the mean
absolute base share over a dataset.

biz_value: a base-DOMINATED target has base_share > 0.7; a residual-DOMINATED
target has base_share < 0.3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.attribution import (
    explain_prediction,
    attribution_summary,
)


def _fit(transform_name, X, y, base_column="base", estimator=None):
    est = CompositeTargetEstimator(
        base_estimator=estimator if estimator is not None else LinearRegression(),
        transform_name=transform_name,
        base_column=base_column,
    )
    est.fit(X, y)
    return est


# ---------------------------------------------------------------------------
# Unit: decomposition reconstructs y_hat exactly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transform_name", ["linear_residual", "diff"])
def test_additive_decomposition_sums_to_yhat(transform_name):
    rng = np.random.default_rng(0)
    n = 200
    base = rng.normal(10.0, 2.0, n)
    f1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * base + 0.5 * f1 + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"base": base, "f1": f1})
    est = _fit(transform_name, X, pd.Series(y))

    table = explain_prediction(est, X)
    assert set(table.columns) == {
        "base_contribution", "residual_contribution", "y_hat", "mode",
    }
    assert (table["mode"] == "additive").all()
    recon = table["base_contribution"] + table["residual_contribution"]
    np.testing.assert_allclose(recon.to_numpy(), table["y_hat"].to_numpy(), rtol=1e-9, atol=1e-9)
    # And y_hat matches the wrapper's own predict.
    np.testing.assert_allclose(table["y_hat"].to_numpy(), est.predict(X), rtol=1e-9, atol=1e-9)


def test_ratio_decomposition_multiplies_to_yhat():
    rng = np.random.default_rng(1)
    n = 200
    base = rng.uniform(2.0, 8.0, n)  # strictly positive -> |base|>0
    f1 = rng.normal(0.0, 1.0, n)
    y = base * (1.5 + 0.1 * f1)
    X = pd.DataFrame({"base": base, "f1": f1})
    est = _fit("ratio", X, pd.Series(y))

    table = explain_prediction(est, X)
    assert (table["mode"] == "multiplicative").all()
    # base factor should be ~base (ratio: y = T*base, base factor = base).
    np.testing.assert_allclose(table["base_contribution"].to_numpy(), base, rtol=1e-6)
    recon = table["base_contribution"] * table["residual_contribution"]
    np.testing.assert_allclose(recon.to_numpy(), table["y_hat"].to_numpy(), rtol=1e-9, atol=1e-9)


def test_logratio_multiplicative_semantics():
    rng = np.random.default_rng(2)
    n = 200
    base = rng.uniform(2.0, 8.0, n)
    f1 = rng.normal(0.0, 1.0, n)
    y = base * np.exp(0.05 * f1)  # y, base > 0
    X = pd.DataFrame({"base": base, "f1": f1})
    est = _fit("logratio", X, pd.Series(y))

    table = explain_prediction(est, X)
    assert (table["mode"] == "multiplicative").all()
    # logratio: y = base*exp(T); base factor (T=0) = base.
    np.testing.assert_allclose(table["base_contribution"].to_numpy(), base, rtol=1e-6)
    recon = table["base_contribution"] * table["residual_contribution"]
    np.testing.assert_allclose(recon.to_numpy(), table["y_hat"].to_numpy(), rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Unit: error paths
# ---------------------------------------------------------------------------

def test_unfitted_raises():
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="base",
    )
    with pytest.raises(NotFittedError):
        explain_prediction(est, pd.DataFrame({"base": [1.0], "f1": [2.0]}))


def test_summary_unfitted_raises():
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="diff",
        base_column="base",
    )
    with pytest.raises(NotFittedError):
        attribution_summary(est, pd.DataFrame({"base": [1.0], "f1": [2.0]}))


def test_base_free_unary_raises():
    rng = np.random.default_rng(3)
    n = 120
    y = np.abs(rng.normal(5.0, 1.0, n)) + 0.5
    X = pd.DataFrame({"f1": rng.normal(0.0, 1.0, n)})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="log_y",
    )
    est.fit(X, pd.Series(y))
    with pytest.raises(ValueError, match="base-free"):
        explain_prediction(est, X)


# ---------------------------------------------------------------------------
# biz_value: base-dominated vs residual-dominated targets
# ---------------------------------------------------------------------------

def test_biz_val_attribution_base_dominated_high_share():
    """A target that is almost ENTIRELY the base level (tiny residual) must
    report base_share > 0.7. Floor 0.7; measured ~0.97."""
    rng = np.random.default_rng(10)
    n = 1000
    base = rng.normal(20.0, 3.0, n)
    f1 = rng.normal(0.0, 1.0, n)
    # y is ~base (slope 1, intercept 0) plus a TINY residual signal.
    y = base + 0.05 * f1 + rng.normal(0.0, 0.02, n)
    X = pd.DataFrame({"base": base, "f1": f1})
    est = _fit("linear_residual", X, pd.Series(y),
               estimator=HistGradientBoostingRegressor(max_iter=60, random_state=0))
    summ = attribution_summary(est, X)
    assert summ["mode"] == "additive"
    assert summ["base_share"] > 0.7, summ
    assert abs(summ["base_share"] + summ["residual_share"] - 1.0) < 1e-9


def test_biz_val_attribution_residual_dominated_low_share():
    """A target whose base column is pure noise (base explains nothing, the
    residual carries the signal AND the level) must report base_share < 0.3.
    Ceiling 0.3; measured ~0.05."""
    rng = np.random.default_rng(11)
    n = 1000
    # base is small-magnitude noise uncorrelated with y; the real signal is in
    # f1/f2 and a large level the inner must learn.
    base = rng.normal(0.0, 0.1, n)
    f1 = rng.normal(0.0, 1.0, n)
    f2 = rng.normal(0.0, 1.0, n)
    y = 50.0 + 8.0 * f1 - 5.0 * f2 + rng.normal(0.0, 0.3, n)
    X = pd.DataFrame({"base": base, "f1": f1, "f2": f2})
    est = _fit("linear_residual", X, pd.Series(y),
               estimator=HistGradientBoostingRegressor(max_iter=80, random_state=0))
    summ = attribution_summary(est, X)
    assert summ["mode"] == "additive"
    assert summ["base_share"] < 0.3, summ


def test_summary_n_rows_reported():
    rng = np.random.default_rng(12)
    n = 150
    base = rng.normal(10.0, 2.0, n)
    f1 = rng.normal(0.0, 1.0, n)
    y = base + f1
    X = pd.DataFrame({"base": base, "f1": f1})
    est = _fit("diff", X, pd.Series(y))
    summ = attribution_summary(est, X)
    assert summ["n_rows"] == n
