"""Unit tests for the extreme-value (POT/GPD) tail composite estimator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mlframe.training.composite.extremes import (
    TailCompositeEstimator,
    fit_gpd_exceedances,
    gpd_tail_quantile,
)


class _ZeroBase:
    """Trivial regressor predicting 0 -> body composite leaves residual == y.

    With ``transform_name='diff'`` and a base column that is 0, the composite
    target is ``y - base = y`` and the inverse adds base back, so the point
    prediction equals the inner predict (here 0). Residuals = y.
    """

    def fit(self, X, y):
        self._mean = float(np.mean(np.asarray(y, dtype=float)))
        try:
            self.feature_names_in_ = np.asarray(list(X.columns))
        except Exception:
            pass
        return self

    def predict(self, X):
        n = X.shape[0]
        return np.zeros(n, dtype=float)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **kw):
        return self


def _frame(n, base=0.0):
    return pd.DataFrame({"f0": np.linspace(0, 1, n), "base": np.full(n, base)})


# -- GPD fit kernel ----------------------------------------------------------


def test_gpd_fit_recovers_known_shape_mle():
    """MLE recovers a known GPD shape within tolerance on a large sample."""
    rng = np.random.default_rng(0)
    xi_true, beta_true = 0.3, 2.0
    sample = stats.genpareto.rvs(c=xi_true, scale=beta_true, size=20000,
                                 random_state=rng)
    xi, beta = fit_gpd_exceedances(sample, method="mle")
    assert abs(xi - xi_true) < 0.06, f"xi {xi} vs {xi_true}"
    assert abs(beta - beta_true) < 0.25, f"beta {beta} vs {beta_true}"


def test_gpd_fit_recovers_known_shape_mom():
    """Method-of-moments recovers shape on a light-tailed (xi<0.5) GPD."""
    rng = np.random.default_rng(1)
    xi_true, beta_true = 0.1, 1.5
    sample = stats.genpareto.rvs(c=xi_true, scale=beta_true, size=20000,
                                 random_state=rng)
    xi, beta = fit_gpd_exceedances(sample, method="mom")
    assert abs(xi - xi_true) < 0.08
    assert abs(beta - beta_true) < 0.2


def test_gpd_fit_empty_returns_default():
    xi, beta = fit_gpd_exceedances(np.array([]))
    assert xi == 0.0 and beta == 1.0


def test_gpd_tail_quantile_monotone_and_unbounded():
    """Tail quantile strictly increasing in q; xi>0 -> exceeds threshold a lot."""
    prev = -np.inf
    for q in (0.96, 0.97, 0.99, 0.999, 0.9999):
        val = gpd_tail_quantile(q, threshold=5.0, threshold_cov=0.95,
                                xi=0.3, beta=2.0)
        assert val > prev, f"non-monotone at q={q}"
        prev = val
    # The 0.999 quantile sits well above the threshold for a heavy tail.
    assert gpd_tail_quantile(0.999, 5.0, 0.95, 0.3, 2.0) > 5.0


def test_gpd_tail_quantile_xi_zero_exponential():
    """xi==0 branch reduces to the exponential tail (beta * -ln(surv_ratio))."""
    val = gpd_tail_quantile(0.99, threshold=5.0, threshold_cov=0.95,
                            xi=0.0, beta=2.0)
    expected = 5.0 + 2.0 * (-np.log((1 - 0.99) / (1 - 0.95)))
    assert abs(val - expected) < 1e-9


# -- estimator end-to-end ----------------------------------------------------


def test_fit_predict_basic_and_params():
    rng = np.random.default_rng(2)
    n = 3000
    X = _frame(n)
    y = stats.t.rvs(df=3, size=n, random_state=rng)  # heavy-tailed residuals
    est = TailCompositeEstimator(base_estimator=_ZeroBase(),
                                 transform_name="diff", base_column="base")
    est.fit(X, y)
    assert est.gpd_fitted_ is True
    params = est.tail_params_
    assert params["gpd_shape"] is not None
    assert params["threshold"] > 0
    assert params["threshold_cov"] == pytest.approx(0.95)
    point = est.predict(X)
    assert point.shape == (n,)


def test_tail_quantile_monotone_in_q_and_exceeds_body():
    rng = np.random.default_rng(3)
    n = 4000
    X = _frame(n)
    y = stats.t.rvs(df=3, size=n, random_state=rng)
    est = TailCompositeEstimator(base_estimator=_ZeroBase(),
                                 transform_name="diff", base_column="base")
    est.fit(X, y)
    Xq = _frame(5)
    prev = -np.inf
    for q in (0.9, 0.95, 0.99, 0.999):
        v = est.predict_tail_quantile(Xq, q)
        assert np.all(v == v[0])  # constant base -> constant offset
        assert v[0] > prev, f"tail quantile not increasing at q={q}"
        prev = v[0]
    # The 0.999 tail quantile exceeds a central (body) quantile.
    body = est.predict_tail_quantile(Xq, 0.5)[0]
    assert est.predict_tail_quantile(Xq, 0.999)[0] > body


def test_too_few_exceedances_fallback_to_empirical():
    """Tiny sample -> too few exceedances -> empirical fallback, no GPD."""
    X = _frame(30)
    y = np.arange(30, dtype=float)
    est = TailCompositeEstimator(base_estimator=_ZeroBase(),
                                 transform_name="diff", base_column="base",
                                 min_exceedances=50)
    est.fit(X, y)
    assert est.gpd_fitted_ is False
    assert est.tail_params_["gpd_shape"] is None
    # Empirical fallback still yields a monotone, finite tail quantile.
    q99 = est.predict_tail_quantile(_frame(1), 0.99)[0]
    q90 = est.predict_tail_quantile(_frame(1), 0.90)[0]
    assert np.isfinite(q99) and q99 >= q90


def test_held_out_residual_split_used():
    """Explicit residual_X/y drives the threshold + GPD, not the train rows."""
    rng = np.random.default_rng(4)
    Xtr = _frame(2000)
    ytr = np.zeros(2000)  # train residuals all ~0 -> degenerate threshold
    Xho = _frame(2000)
    yho = stats.t.rvs(df=3, size=2000, random_state=rng)
    est = TailCompositeEstimator(base_estimator=_ZeroBase(),
                                 transform_name="diff", base_column="base")
    est.fit(Xtr, ytr, residual_X=Xho, residual_y=yho)
    # Threshold reflects the heavy held-out residuals, not the zero train ones.
    assert est.threshold_ > 0.5


def test_threshold_pct_validation():
    est = TailCompositeEstimator(base_estimator=_ZeroBase(),
                                 transform_name="diff", base_column="base",
                                 threshold_pct=1.5)
    with pytest.raises(ValueError):
        est.fit(_frame(100), np.arange(100, dtype=float))


def test_q_out_of_range_raises():
    est = TailCompositeEstimator(base_estimator=_ZeroBase(),
                                 transform_name="diff", base_column="base")
    est.fit(_frame(2000), stats.t.rvs(df=3, size=2000, random_state=5))
    with pytest.raises(ValueError):
        est.predict_tail_quantile(_frame(1), 1.0)


def test_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    est = TailCompositeEstimator(base_estimator=_ZeroBase())
    with pytest.raises(NotFittedError):
        est.predict(_frame(1))


def test_mom_method_fits_gpd():
    est = TailCompositeEstimator(base_estimator=_ZeroBase(),
                                 transform_name="diff", base_column="base",
                                 gpd_method="mom")
    est.fit(_frame(4000), stats.t.rvs(df=3, size=4000, random_state=6))
    assert est.gpd_fitted_ is True
    assert est.tail_params_["gpd_method"] == "mom"
