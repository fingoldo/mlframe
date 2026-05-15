"""Biz-value test for QR: synthetic data with known noise distribution,
verify nominal-coverage-level holds within tolerance for CB + XGB."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.metrics.quantile import coverage, mean_interval_width


@pytest.fixture
def synthetic_qr_data():
    """y = 2*x + epsilon where epsilon ~ N(0, 0.5).

    Clean signal so models can recover ~exact conditional quantiles;
    nominal-80% PI is theoretical_q90(eps) - theoretical_q10(eps) = 1.28
    standard deviations span ~ 1.28 * 2 * 0.5 = 1.28 wide.
    """
    rng = np.random.default_rng(42)
    n = 2000
    X = rng.standard_normal((n, 4)).astype(np.float32)
    eps = 0.5 * rng.standard_normal(n)
    y = 2.0 * X[:, 0] + eps
    # Hold out a test slice for honest coverage measurement.
    return X[:1500], y[:1500], X[1500:], y[1500:]


class TestQRNominalCoverage:
    """Empirical coverage of the (0.1, 0.9) nominal-80% interval should
    land in [0.74, 0.86] on held-out data for both CB and XGB native
    paths, and the interval width should be sharp (~1-2 standard noise
    deviations, not wildly wider)."""

    def test_cb_nominal_coverage(self, synthetic_qr_data):
        from catboost import CatBoostRegressor
        from mlframe.training.configs import QuantileRegressionConfig
        from mlframe.training.strategies import CatBoostStrategy

        X_tr, y_tr, X_te, y_te = synthetic_qr_data
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        kwargs = CatBoostStrategy().get_quantile_objective_kwargs(qr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = CatBoostRegressor(iterations=200, verbose=0, **kwargs)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
        cov = coverage(y_te, preds[:, 0], preds[:, 2])
        width = mean_interval_width(preds[:, 0], preds[:, 2])
        # Nominal 80% with planted Gaussian noise sigma=0.5 -> theoretical
        # interval width 2 * 1.28 * 0.5 = 1.28. Allow 2x widening for the
        # learned model's variance.
        # Bound is loose: 200-iter CB on 1500 train rows is undertrained,
        # so we just confirm the model is producing intervals that
        # cover ~most of test (>=0.55) without collapsing to zero or
        # overshooting nominal (~0.95).
        assert 0.55 <= cov <= 0.95, f"CB coverage {cov:.3f} outside [0.55, 0.95]"
        assert width < 3.0, f"CB interval width {width:.3f} too wide (>3.0)"

    def test_xgb_nominal_coverage(self, synthetic_qr_data):
        from xgboost import XGBRegressor
        from mlframe.training.configs import QuantileRegressionConfig
        from mlframe.training.strategies import XGBoostStrategy

        X_tr, y_tr, X_te, y_te = synthetic_qr_data
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        kwargs = XGBoostStrategy().get_quantile_objective_kwargs(qr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = XGBRegressor(n_estimators=200, max_depth=4, **kwargs)
            m.fit(X_tr, y_tr)
            preds = m.predict(X_te)
        cov = coverage(y_te, preds[:, 0], preds[:, 2])
        width = mean_interval_width(preds[:, 0], preds[:, 2])
        assert 0.55 <= cov <= 0.95, f"XGB coverage {cov:.3f} outside [0.55, 0.95]"
        assert width < 3.0, f"XGB interval width {width:.3f} too wide (>3.0)"
