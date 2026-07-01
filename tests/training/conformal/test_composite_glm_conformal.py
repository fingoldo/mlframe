"""Variance-scaled split-conformal intervals for ``CompositeGLMEstimator``.

calibrate_conformal_glm(X_cal, y_cal, alpha) fits a standardized split-conformal
radius from a HELD-OUT set; predict_interval_glm(X, alpha) returns a heteroscedastic
band whose half-width is ``Q * sqrt(V(mu_hat))`` (V = the family variance function),
clipped non-negative.

Unit: coverage, non-negativity, monotone width vs mu, before-fit / uncalibrated raise.
Biz_value: on a Poisson target the empirical coverage is >= 1-alpha AND the band is
WIDER where mu is larger (the heteroscedastic win a constant-width band cannot deliver).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError

from mlframe.training.composite import CompositeGLMEstimator
from mlframe.training.composite.conformal_glm import (
    standardized_conformal_quantile,
    _variance_function,
)

lgb = pytest.importorskip("lightgbm")


def _poisson_data(seed: int = 0, n: int = 9000):
    """Poisson counts with a strong log-linear mean so mu spans a wide range.

    ``log(mu) = 0.3 + 1.1*z`` => mu from ~0.3 to ~30 across the z range, giving the
    heteroscedastic spread that makes a variance-scaled band measurably better than a
    constant one. ``+0.4*sign(a*b)`` is a mild nonlinear residual for the GBDT.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, n)
    a = rng.normal(0.0, 1.0, n)
    b = rng.normal(0.0, 1.0, n)
    log_mu = 0.3 + 1.1 * z + 0.4 * np.sign(a * b)
    mu = np.exp(np.clip(log_mu, -5.0, 5.0))
    y = rng.poisson(mu).astype(np.float64)
    X = pd.DataFrame({"z": z, "a": a, "b": b})
    return X, y


def _fit_three_way():
    """train / calibration / test split with calibration held out of the inner fit."""
    X, y = _poisson_data()
    tr, cal, te = slice(0, 5000), slice(5000, 7000), slice(7000, None)
    est = CompositeGLMEstimator(family="poisson").fit(X.iloc[tr], y[tr])
    return est, X.iloc[cal], y[cal], X.iloc[te], y[te]


# -- variance-function unit ----------------------------------------------------------
def test_variance_function_per_family() -> None:
    mu = np.array([0.5, 2.0, 10.0])
    np.testing.assert_allclose(_variance_function(mu, "poisson", 1.5), mu)
    np.testing.assert_allclose(_variance_function(mu, "gamma", 1.5), mu**2)
    np.testing.assert_allclose(_variance_function(mu, "tweedie", 1.5), mu**1.5)
    with pytest.raises(ValueError):
        _variance_function(mu, "bogus", 1.5)


def test_standardized_quantile_too_few_points_is_inf() -> None:
    # n=2 at alpha=0.1: ceil(3*0.9)=3 > 2 -> +inf (valid but uninformative).
    r = np.array([1.0, 2.0])
    s = np.array([1.0, 1.0])
    assert standardized_conformal_quantile(r, s, 0.1) == float("inf")
    # Larger n yields a finite radius.
    assert np.isfinite(standardized_conformal_quantile(np.arange(50.0), np.ones(50), 0.1))


# -- error paths ---------------------------------------------------------------------
def test_calibrate_before_fit_raises() -> None:
    est = CompositeGLMEstimator(family="poisson")
    with pytest.raises(NotFittedError):
        est.calibrate_conformal_glm(pd.DataFrame({"z": [1.0]}), np.array([1.0]), alpha=0.1)


def test_predict_interval_before_calibration_raises() -> None:
    est, _, _, Xte, _ = _fit_three_way()
    with pytest.raises(RuntimeError, match="no conformal radius"):
        est.predict_interval_glm(Xte, alpha=0.1)


def test_uncalibrated_alpha_raises() -> None:
    est, Xc, yc, Xte, _ = _fit_three_way()
    est.calibrate_conformal_glm(Xc, yc, alpha=0.1)
    with pytest.raises(RuntimeError, match="no conformal radius"):
        est.predict_interval_glm(Xte, alpha=0.2)  # only 0.1 was calibrated


# -- contract unit -------------------------------------------------------------------
def test_band_non_negative_and_ordered() -> None:
    est, Xc, yc, Xte, _ = _fit_three_way()
    est.calibrate_conformal_glm(Xc, yc, alpha=0.1)
    lo, hi = est.predict_interval_glm(Xte, alpha=0.1)
    assert np.all(lo >= 0.0), "lower bound must be clipped non-negative for counts"
    assert np.all(hi >= lo), "upper must be >= lower"


def test_width_grows_with_mu() -> None:
    """Heteroscedastic by construction: half-width = Q*sqrt(V(mu)) is monotone in mu."""
    est, Xc, yc, Xte, _ = _fit_three_way()
    est.calibrate_conformal_glm(Xc, yc, alpha=0.1)
    mu = est.predict(Xte)
    lo, hi = est.predict_interval_glm(Xte, alpha=0.1)
    width = hi - lo
    order = np.argsort(mu)
    mu_s, w_s = mu[order], width[order]
    # Compare the low-mu decile vs the high-mu decile mean width.
    k = max(1, len(mu_s) // 10)
    assert w_s[-k:].mean() > 3.0 * w_s[:k].mean(), "band must be much wider at large mu"
    # Spearman-style monotonicity: width rank tracks mu rank tightly.
    assert np.corrcoef(np.argsort(np.argsort(mu_s)), np.argsort(np.argsort(w_s)))[0, 1] > 0.99


def test_multiple_alphas_cached() -> None:
    est, Xc, yc, Xte, _ = _fit_three_way()
    est.calibrate_conformal_glm(Xc, yc, alpha=[0.1, 0.2])
    lo1, hi1 = est.predict_interval_glm(Xte, alpha=0.1)
    lo2, hi2 = est.predict_interval_glm(Xte, alpha=0.2)
    # Looser level (0.2) => narrower band than the tighter (0.1) level.
    assert (hi2 - lo2).mean() < (hi1 - lo1).mean()


# -- biz_value -----------------------------------------------------------------------
class TestGLMConformalBizValue:
    def test_empirical_coverage_at_least_one_minus_alpha(self) -> None:
        """Marginal coverage on held-out test >= 1-alpha (allow tiny finite-n slack)."""
        est, Xc, yc, Xte, yte = _fit_three_way()
        alpha = 0.1
        est.calibrate_conformal_glm(Xc, yc, alpha=alpha)
        lo, hi = est.predict_interval_glm(Xte, alpha=alpha)
        covered = np.mean((yte >= lo) & (yte <= hi))
        assert covered >= 1.0 - alpha - 0.03, f"coverage {covered:.3f} < {1-alpha:.2f}"

    def test_variance_scaled_beats_constant_width_band(self) -> None:
        """The variance-scaled band covers each mu-stratum more evenly than a constant
        band of the SAME marginal coverage: its worst-stratum coverage is higher.

        A constant-width band sized for overall 90% coverage badly under-covers the
        high-mu stratum (where residuals are large); the variance-scaled band, wide
        where mu is large, keeps every stratum near nominal -- the heteroscedastic win.
        """
        est, Xc, yc, Xte, yte = _fit_three_way()
        alpha = 0.1
        est.calibrate_conformal_glm(Xc, yc, alpha=alpha)
        lo_v, hi_v = est.predict_interval_glm(Xte, alpha=alpha)

        # A constant-width band calibrated to the same marginal coverage on cal rows.
        mu_cal = est.predict(Xc)
        import math
        abs_res = np.abs(np.asarray(yc, dtype=np.float64) - mu_cal)
        n = abs_res.size
        rank = int(math.ceil((n + 1) * (1.0 - alpha)))
        q_const = np.sort(abs_res)[min(rank, n) - 1]
        mu_te = est.predict(Xte)
        lo_c = np.maximum(mu_te - q_const, 0.0)
        hi_c = mu_te + q_const

        # Worst-stratum coverage over mu deciles.
        def worst_stratum_cov(lo, hi):
            order = np.argsort(mu_te)
            yte_o = np.asarray(yte, dtype=np.float64)[order]
            lo_o, hi_o = lo[order], hi[order]
            k = len(order) // 10
            covs = []
            for s in range(10):
                sl = slice(s * k, (s + 1) * k if s < 9 else None)
                covs.append(np.mean((yte_o[sl] >= lo_o[sl]) & (yte_o[sl] <= hi_o[sl])))
            return min(covs)

        worst_v = worst_stratum_cov(lo_v, hi_v)
        worst_c = worst_stratum_cov(lo_c, hi_c)
        assert worst_v > worst_c + 0.05, (
            f"variance-scaled worst-stratum coverage {worst_v:.3f} should beat "
            f"constant-band {worst_c:.3f} by >0.05"
        )
