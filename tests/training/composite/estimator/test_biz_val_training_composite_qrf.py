"""biz_value tests for ``CompositeQRFEstimator``.

The win: on a HETEROSCEDASTIC target the quantile-regression-forest composite, from a
SINGLE fitted forest, (a) achieves near-nominal per-quantile coverage on held-out
data, (b) beats a homoscedastic-Gaussian baseline (constant width) on the strictly
proper CRPS, and (c) is CRPS-competitive with the dense-quantile
:class:`CompositeDistributionEstimator` -- which needs K separate pinball refits --
while serving any number of quantiles from one model.

A regression that flattens the input-dependent spread (forest collapsing toward the
mean, broken leaf-residual reconstruction) loses the CRPS gap and trips these tests.

The QRF backend is pure sklearn, so these run with no optional inner lib. The
dense-quantile comparison uses LightGBM when present (``importorskip``); otherwise the
comparison sub-test is skipped but the standalone QRF wins still assert.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from mlframe.training.composite.qrf import CompositeQRFEstimator


def _hetero_xy(n: int = 5000, seed: int = 0, strong: bool = False):
    """y = 2*base + eps(base); noise std grows with |base| (input-dependent spread).

    ``strong=True`` steepens the spread (std ~ |base|^2) so the gap between an
    input-adaptive predictor and any CONSTANT-width baseline is large and unambiguous
    on the strictly-proper CRPS.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-3.0, 3.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    sigma = (0.15 + 0.45 * base**2) if strong else (0.3 + 0.9 * np.abs(base))
    y = 2.0 * base + 0.5 * x1 + sigma * rng.standard_normal(n)
    return pd.DataFrame({"base": base, "x1": x1}), y


def _homoscedastic_gaussian_crps(mu: np.ndarray, sd: float, y: np.ndarray) -> float:
    """Closed-form mean CRPS of N(mu_i, sd) vs y_i (constant width baseline)."""
    z = (y - mu) / sd
    crps = sd * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def _split(n: int = 5000, seed: int = 0, strong: bool = False):
    X, y = _hetero_xy(n=n, seed=seed, strong=strong)
    ntr = (n * 4) // 5
    return X.iloc[:ntr], y[:ntr], X.iloc[ntr:], y[ntr:]


def test_biz_val_qrf_single_fit_coverage_near_nominal():
    """ONE QRF fit yields near-nominal held-out coverage at every queried level.

    Max per-quantile coverage error across q in {0.1,0.25,0.5,0.75,0.9} must be
    <= 0.05 -- a calibrated conditional distribution, not a mean regressor. All five
    levels come from the SAME fitted forest (no refit), which is the headline win.
    """
    Xtr, ytr, Xte, yte = _split()
    est = CompositeQRFEstimator(
        base_column="base",
        n_estimators=200,
        min_samples_leaf=10,
        random_state=0,
    ).fit(Xtr, ytr)
    levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    Q = est.predict_quantile(Xte, quantiles=levels)  # one model, all levels
    below = (yte[:, None] <= Q).mean(axis=0)
    max_err = float(np.max(np.abs(below - levels)))
    assert max_err <= 0.05, f"max per-quantile coverage error {max_err:.3f} (cov={below})"


def test_biz_val_qrf_beats_homoscedastic_gaussian_crps():
    """QRF mean CRPS strictly below a constant-width Gaussian baseline.

    The baseline uses the held-out mean fit (forest mean) plus the single best
    constant std; on the strongly-heteroscedastic target QRF's input-dependent width
    must score clearly lower on the strictly-proper CRPS. Floor: QRF CRPS <= 0.93 *
    baseline CRPS (a 7%% edge; measured ~9.6%%, margin absorbs seed noise).
    """
    Xtr, ytr, Xte, yte = _split(strong=True)
    est = CompositeQRFEstimator(
        base_column="base",
        n_estimators=200,
        min_samples_leaf=10,
        random_state=0,
    ).fit(Xtr, ytr)
    qrf_crps = est.crps(Xte, yte)
    mu = est.predict(Xte)
    # Best constant std for the baseline: the held-out RMSE of the mean prediction.
    sd = float(np.sqrt(np.mean((yte - mu) ** 2)))
    base_crps = _homoscedastic_gaussian_crps(mu, sd, yte)
    assert qrf_crps <= 0.93 * base_crps, f"QRF crps {qrf_crps:.4f} vs baseline {base_crps:.4f}"


def test_biz_val_qrf_crps_competitive_with_dense_quantile():
    """QRF CRPS is competitive with the dense-quantile estimator from a single fit.

    The dense-quantile :class:`CompositeDistributionEstimator` refits K pinball heads;
    QRF fits ONE forest. On this heteroscedastic target QRF must be within 15%% of the
    dense-quantile CRPS (measured: QRF actually <= dense). Skips if LightGBM absent.
    """
    lgbm = pytest.importorskip("lightgbm")
    from mlframe.training.composite.distributional import CompositeDistributionEstimator

    Xtr, ytr, Xte, yte = _split()
    qrf = CompositeQRFEstimator(
        base_column="base",
        n_estimators=200,
        min_samples_leaf=10,
        random_state=0,
    ).fit(Xtr, ytr)
    dense = CompositeDistributionEstimator(
        base_estimator=lgbm.LGBMRegressor(n_estimators=150, num_leaves=31, verbose=-1),
        base_column="base",
    ).fit(Xtr, ytr)
    qrf_crps = qrf.crps(Xte, yte)
    dense_crps = dense.crps(Xte, yte)
    assert qrf_crps <= 1.15 * dense_crps, f"QRF crps {qrf_crps:.4f} vs dense {dense_crps:.4f}"


def test_biz_val_qrf_sharper_than_global_marginal_quantiles():
    """QRF intervals adapt to |base|: narrow where noise is small, wide where large.

    A trivial baseline predicts the GLOBAL marginal quantiles (same band for every
    row). QRF's conditional band must be much narrower in the low-noise region
    (|base| small) than that global band -- proof it learned the input-dependent
    spread rather than a constant width. Ratio floor: low-noise QRF band <= 0.6 *
    global band.
    """
    Xtr, ytr, Xte, _yte = _split()
    est = CompositeQRFEstimator(
        base_column="base",
        n_estimators=200,
        min_samples_leaf=10,
        random_state=0,
    ).fit(Xtr, ytr)
    Q = est.predict_quantile(Xte, quantiles=[0.1, 0.9])
    width = Q[:, 1] - Q[:, 0]
    global_band = float(np.quantile(ytr, 0.9) - np.quantile(ytr, 0.1))
    low_noise = np.abs(Xte["base"].to_numpy()) < 0.5
    qrf_low = float(np.mean(width[low_noise]))
    assert qrf_low <= 0.6 * global_band, f"low-noise QRF band {qrf_low:.3f} vs global {global_band:.3f}"
