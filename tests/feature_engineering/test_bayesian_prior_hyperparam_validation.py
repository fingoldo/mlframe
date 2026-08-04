"""FE_ROOT_A-4 (2026-08-05 audit): bocpd_features / online_bayesian_linear_regression had no validation
on their conjugate-prior hyperparameters.

- ``bocpd_features``'s Normal-Inverse-Gamma prior requires kappa0 > 0, alpha0 > 0, beta0 > 0. Before this
  fix, ``alpha0=0.0`` divided by zero inside the njit Student-t predictive-scale computation
  (``scale_sq = beta[r] * (kappa[r] + 1.0) / (alpha[r] * kappa[r])``), crashing with an unhandled numba
  exception instead of a clear Python-level error.
- ``online_bayesian_linear_regression``'s isotropic prior covariance is ``Sigma_0 = I / prior_precision``.
  Before this fix, ``prior_precision <= 0`` silently produced inf/nan propagating through every output
  (predictive_mean/var, log_marginal_lik) with no error or warning.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_bocpd_alpha0_zero_raises_instead_of_crashing():
    """alpha0=0.0 must raise a clear ValueError, not crash inside the njit kernel."""
    from mlframe.feature_engineering.bayesian import bocpd_features

    x = np.random.default_rng(0).normal(size=200).astype(np.float64)
    with pytest.raises(ValueError, match="alpha0"):
        bocpd_features(x, alpha0=0.0)


@pytest.mark.parametrize("kwargs", [{"kappa0": 0.0}, {"kappa0": -1.0}, {"alpha0": -0.5}, {"beta0": 0.0}, {"beta0": -2.0}])
def test_bocpd_nonpositive_nig_hyperparams_raise(kwargs):
    """Every NIG hyperparameter (kappa0, alpha0, beta0) must be strictly positive."""
    from mlframe.feature_engineering.bayesian import bocpd_features

    x = np.random.default_rng(1).normal(size=100).astype(np.float64)
    with pytest.raises(ValueError):
        bocpd_features(x, **kwargs)


def test_bocpd_positive_hyperparams_still_work():
    """Sanity: valid (default) hyperparameters are unaffected by the new guard."""
    from mlframe.feature_engineering.bayesian import bocpd_features

    x = np.random.default_rng(2).normal(size=300).astype(np.float64)
    out = bocpd_features(x, kappa0=1.0, alpha0=1.0, beta0=1.0)
    assert np.isfinite(out["p_change"]).all()


@pytest.mark.parametrize("prior_precision", [0.0, -1.0, -0.001])
def test_oblr_nonpositive_prior_precision_raises_instead_of_silent_nan(prior_precision):
    """prior_precision <= 0 must raise, not silently return inf/nan predictions."""
    from mlframe.feature_engineering._bayesian_oblr import online_bayesian_linear_regression

    rng = np.random.default_rng(3)
    n = 50
    y = rng.normal(size=n).astype(np.float64)
    X = np.column_stack([np.ones(n), rng.normal(size=n)]).astype(np.float64)
    with pytest.raises(ValueError, match="prior_precision"):
        online_bayesian_linear_regression(y, X, prior_precision=prior_precision)


def test_oblr_positive_prior_precision_still_works():
    """Sanity: a valid (default) prior_precision is unaffected by the new guard and returns finite output."""
    from mlframe.feature_engineering._bayesian_oblr import online_bayesian_linear_regression

    rng = np.random.default_rng(4)
    n = 50
    y = rng.normal(size=n).astype(np.float64)
    X = np.column_stack([np.ones(n), rng.normal(size=n)]).astype(np.float64)
    out = online_bayesian_linear_regression(y, X, prior_precision=1.0)
    assert np.isfinite(out["predictive_mean"]).all()
    assert np.isfinite(out["predictive_var"]).all()
