"""Unit + biz_value tests for regression point recalibration (`_regression_calibration.py`).

Unit: monotonicity, identity on degenerate/tiny input, picklability, gain sign on test.
biz_value: a shrunk predictor (predictions compressed toward the mean, like an early-stopped GBM
on a heavy-tailed target) is corrected by isotonic ``g`` for a real honest-holdout RMSE gain,
while an already-calibrated predictor is left ~unchanged (gain ~ 0, never materially worse).
"""

from __future__ import annotations

import pickle

import numpy as np

from mlframe.training._regression_calibration import (
    fit_point_recalibrator,
    recalibration_rmse_gain,
)


def test_recalibrator_is_monotone():
    rng = np.random.default_rng(0)
    yp = rng.uniform(-3, 3, 2000)
    yt = 2.0 * yp + rng.standard_normal(2000)  # shrunk preds vs true (slope 2)
    g = fit_point_recalibrator(yp, yt, method="isotonic")
    grid = np.linspace(-3, 3, 50)
    out = g.transform(grid)
    assert np.all(np.diff(out) >= -1e-9), "isotonic recalibrator must be non-decreasing"


def test_identity_on_tiny_and_degenerate_input():
    g_tiny = fit_point_recalibrator(np.arange(5.0), np.arange(5.0))
    assert g_tiny._identity
    assert np.allclose(g_tiny.transform(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
    # Constant predictor -> nothing to learn -> identity.
    g_const = fit_point_recalibrator(np.full(50, 2.0), np.random.default_rng(1).standard_normal(50))
    assert g_const._identity


def test_linear_slope_clamped_nonnegative():
    rng = np.random.default_rng(2)
    yp = rng.uniform(0, 1, 500)
    yt = -3.0 * yp + rng.standard_normal(500) * 0.01  # anti-correlated
    g = fit_point_recalibrator(yp, yt, method="linear")
    assert g._slope >= 0.0  # ranking-safe: never invert the order


def test_recalibrator_picklable():
    rng = np.random.default_rng(3)
    yp = rng.uniform(-2, 2, 1000)
    yt = 1.5 * yp + rng.standard_normal(1000)
    g = fit_point_recalibrator(yp, yt)
    g2 = pickle.loads(pickle.dumps(g))
    assert np.allclose(g.transform(yp), g2.transform(yp))


def _shrunk(n, seed, shrink):
    """Heavy-tailed target with predictions shrunk toward the mean by factor ``shrink`` (<1)."""
    rng = np.random.default_rng(seed)
    y = rng.standard_t(df=3, size=n)  # heavy tails -> extremes matter
    pred = shrink * y + (1.0 - shrink) * float(np.mean(y)) + 0.1 * rng.standard_normal(n)
    return pred, y


def test_biz_val_recalibration_fixes_shrunk_predictor():
    """Isotonic recalibration recovers a shrunk predictor's tails for a real honest-holdout RMSE gain.

    Measured (heavy-tailed t(3), shrink=0.5, n=4000, 3 seeds): raw RMSE is inflated by the
    compressed tails; isotonic ``g`` fit on calib lifts honest-test RMSE. Floor the median gain at
    0.05 (well below the measured ~0.2-0.4) so a regression that breaks recalibration trips it.
    """
    gains = []
    for seed in range(3):
        pred_cal, y_cal = _shrunk(4000, seed, shrink=0.5)
        pred_test, y_test = _shrunk(4000, seed + 50, shrink=0.5)
        g = fit_point_recalibrator(pred_cal, y_cal, method="isotonic")
        gains.append(recalibration_rmse_gain(g, pred_test, y_test))
    assert float(np.median(gains)) >= 0.05, gains


def test_biz_val_recalibration_no_harm_on_calibrated_predictor():
    """On an already-calibrated predictor (pred = y + small noise), ``g`` is ~identity: no material harm.

    The monotone map cannot meaningfully improve an unbiased predictor, and must not degrade it beyond
    noise. Assert the median gain is within +-0.03 (essentially zero) across seeds.
    """
    gains = []
    for seed in range(3):
        rng = np.random.default_rng(seed + 200)
        y_cal = rng.standard_normal(4000)
        pred_cal = y_cal + 0.3 * rng.standard_normal(4000)
        y_test = rng.standard_normal(4000)
        pred_test = y_test + 0.3 * rng.standard_normal(4000)
        g = fit_point_recalibrator(pred_cal, y_cal, method="isotonic")
        gains.append(recalibration_rmse_gain(g, pred_test, y_test))
    assert abs(float(np.median(gains))) <= 0.03, gains
