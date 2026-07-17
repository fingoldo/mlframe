"""biz_value tests for ``CompositeDistributionEstimator``.

The win: on a HETEROSCEDASTIC target the composite predictive distribution (a
dense per-quantile fan that learns the input-dependent spread) achieves a LOWER
mean CRPS than a homoscedastic-Gaussian baseline (mean +/- a single constant
std), AND its per-quantile pinball coverage is near-nominal.

CRPS is strictly proper, so a distribution that correctly narrows where the noise
is small and widens where it is large strictly beats one with a fixed width. A
regression that flattens the per-input spread (e.g. heads collapsing to the mean)
trips this test by losing the CRPS gap.

Uses a real pinball-capable inner if LightGBM is installed; otherwise an
``importorskip`` keeps CI green without hard-requiring the dep.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mlframe.training.composite.distributional import (
    CompositeDistributionEstimator,
)


def _heteroscedastic_xy(n: int = 4000, seed: int = 0):
    """y = 2*base + eps(base), where the noise std GROWS with |base|.

    The conditional spread is input-dependent, so a constant-width predictor
    cannot be calibrated everywhere -- exactly where the composite distribution
    (per-quantile, hence per-input width) wins.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-3.0, 3.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    sigma = 0.3 + 0.9 * np.abs(base)  # spread grows with |base|
    y = 2.0 * base + 0.5 * x1 + sigma * rng.standard_normal(n)
    return pd.DataFrame({"base": base, "x1": x1}), y


def _homoscedastic_gaussian_crps(mu: np.ndarray, sd: float, y: np.ndarray) -> float:
    """Closed-form CRPS of N(mu_i, sd) vs y_i (Gneiting & Raftery 2007), mean.

    CRPS(N(mu,sd), y) = sd * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ],
    z = (y - mu)/sd. This is the homoscedastic-Gaussian baseline: one constant
    std for every row.
    """
    z = (y - mu) / sd
    crps = sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def _make_inner(alpha: float = 0.5):
    lgbm = pytest.importorskip("lightgbm")
    return lgbm.LGBMRegressor(
        n_estimators=120,
        num_leaves=31,
        learning_rate=0.1,
        min_child_samples=20,
        verbose=-1,
    )


def test_biz_val_distribution_beats_homoscedastic_gaussian_crps():
    """Composite distribution mean CRPS < homoscedastic-Gaussian baseline CRPS.

    Measured gap is large (the baseline is mis-calibrated by construction in the
    high-|base| tails); floor at a 5% improvement to absorb seed noise while
    still detecting a head-collapse regression.
    """
    X, y = _heteroscedastic_xy()
    est = CompositeDistributionEstimator(
        base_estimator=_make_inner(),
        transform_name="linear_residual",
        base_column="base",
    ).fit(X, y)

    crps_dist = est.crps(X, y)

    # Homoscedastic baseline: predicted mean = the distribution's median head,
    # single std = the global residual std (the best a constant-width model can do).
    mu = est.predict(X)
    sd = float(np.std(y - mu))
    crps_gauss = _homoscedastic_gaussian_crps(mu, sd, y)

    assert crps_dist < crps_gauss * 0.95, (
        f"composite CRPS {crps_dist:.4f} should beat homoscedastic-Gaussian {crps_gauss:.4f} by >=5% on a heteroscedastic target"
    )


def test_biz_val_distribution_per_quantile_coverage_near_nominal():
    """Empirical coverage P(y <= Q(q)) is near the nominal level q for each q.

    A calibrated predictive distribution has the fraction of targets below its
    q-quantile approximately equal to q. We check the dense grid: max absolute
    deviation from nominal <= 0.10 (in-sample, dense-grid tolerance).
    """
    X, y = _heteroscedastic_xy()
    est = CompositeDistributionEstimator(
        base_estimator=_make_inner(),
        transform_name="linear_residual",
        base_column="base",
    ).fit(X, y)

    levels = est.quantiles_
    qmat = est.predict_quantile(X, quantiles=levels)
    y_arr = np.asarray(y, dtype=np.float64)
    coverage = (y_arr[:, None] <= qmat).mean(axis=0)  # empirical P(y <= Q(q))
    max_dev = float(np.max(np.abs(coverage - levels)))
    assert max_dev <= 0.10, f"per-quantile coverage off nominal by {max_dev:.3f} (coverage={coverage.round(3).tolist()}, levels={levels.round(2).tolist()})"
