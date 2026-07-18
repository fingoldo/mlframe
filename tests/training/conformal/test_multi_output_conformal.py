"""Per-output split-conformal intervals for CompositeMultiOutputEstimator.

Unit tests:
- shape (n, K) for both lower and upper
- before-fit raises NotFittedError
- uncalibrated predict_interval raises RuntimeError
- single-row predict yields (1, K)
- wrong-K calibration target raises
- failed (all-NaN) column gets a degenerate band (lower == upper) and does not
  break the rectangular (n, K) output

biz_value:
- on a 3-output target with INDEPENDENT per-column scales, the EMPIRICAL
  per-column coverage on a fresh test set is >= 1 - alpha for EVERY column.
  This is the conformal guarantee the feature exists to deliver; a regression
  that pooled one radius or mis-sliced columns would drop a column below the
  floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression

from mlframe.training.composite.multi_output import CompositeMultiOutputEstimator


def _make_xy(n, seed=0):
    """3-output target; columns have DIFFERENT scales + noise so a single pooled
    radius would mis-cover at least one column."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f0": rng.standard_normal(n),
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
        }
    )
    y0 = 2.0 * X["f0"].to_numpy() + rng.normal(0, 0.3, n)
    y1 = -1.0 * X["f1"].to_numpy() + rng.normal(0, 1.5, n)  # wider noise
    y2 = 0.5 * X["f2"].to_numpy() + rng.normal(0, 5.0, n)  # widest noise
    y = np.column_stack([y0, y1, y2])
    return X, y


def _fit_estimator(n=600, seed=0):
    """Fit estimator."""
    X, y = _make_xy(n, seed)
    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs={"transform_name": "yeo_johnson_y"},
    )
    est.fit(X, y)
    return est, X, y


# --------------------------------------------------------------------------
# Unit tests
# --------------------------------------------------------------------------


def test_predict_interval_shape_n_by_k():
    """Predict interval shape n by k."""
    est, X, _y = _fit_estimator()
    X_cal, y_cal = _make_xy(300, seed=1)
    est.calibrate_conformal(X_cal, y_cal, alpha=0.1)
    lower, upper = est.predict_interval(X, alpha=0.1)
    assert lower.shape == (len(X), 3)
    assert upper.shape == (len(X), 3)
    assert (upper >= lower).all()


def test_predict_interval_before_fit_raises():
    """Predict interval before fit raises."""
    est = CompositeMultiOutputEstimator(base_estimator=LinearRegression())
    X = pd.DataFrame({"f0": [1.0], "f1": [2.0], "f2": [3.0]})
    with pytest.raises(NotFittedError):
        est.predict_interval(X, alpha=0.1)


def test_calibrate_conformal_before_fit_raises():
    """Calibrate conformal before fit raises."""
    est = CompositeMultiOutputEstimator(base_estimator=LinearRegression())
    X, y = _make_xy(50)
    with pytest.raises(NotFittedError):
        est.calibrate_conformal(X, y, alpha=0.1)


def test_predict_interval_uncalibrated_raises():
    """Predict interval uncalibrated raises."""
    est, X, _ = _fit_estimator()
    with pytest.raises(RuntimeError, match="no per-column conformal radius"):
        est.predict_interval(X, alpha=0.1)


def test_predict_interval_uncalibrated_for_this_alpha_raises():
    """Predict interval uncalibrated for this alpha raises."""
    est, X, _y = _fit_estimator()
    X_cal, y_cal = _make_xy(300, seed=1)
    est.calibrate_conformal(X_cal, y_cal, alpha=0.1)
    # Calibrated 0.1 but asking for 0.05.
    with pytest.raises(RuntimeError, match="alpha=0.05"):
        est.predict_interval(X, alpha=0.05)


def test_single_row_predict_interval():
    """Single row predict interval."""
    est, _, _ = _fit_estimator()
    X_cal, y_cal = _make_xy(300, seed=1)
    est.calibrate_conformal(X_cal, y_cal, alpha=0.1)
    X1 = pd.DataFrame({"f0": [0.5], "f1": [-0.3], "f2": [1.2]})
    lower, upper = est.predict_interval(X1, alpha=0.1)
    assert lower.shape == (1, 3)
    assert upper.shape == (1, 3)
    assert (upper >= lower).all()


def test_calibrate_wrong_k_raises():
    """Calibrate wrong k raises."""
    est, _, _ = _fit_estimator()
    X_cal, _ = _make_xy(300, seed=1)
    y_bad = np.zeros((300, 2))  # only 2 columns, est fit on 3
    with pytest.raises(ValueError, match="columns but the estimator"):
        est.calibrate_conformal(X_cal, y_bad, alpha=0.1)


def test_failed_column_gives_degenerate_band():
    """An all-NaN output column is recorded as a failed column (constant
    fallback, no inner estimator); its conformal band is degenerate (lower ==
    upper) and does not break the rectangular (n, K) output."""
    rng = np.random.default_rng(3)
    n = 400
    X = pd.DataFrame({"f0": rng.standard_normal(n), "f1": rng.standard_normal(n)})
    y0 = 2.0 * X["f0"].to_numpy() + rng.normal(0, 0.3, n)
    y1 = np.full(n, np.nan)  # fully-NaN -> failed column
    y = np.column_stack([y0, y1])
    est = CompositeMultiOutputEstimator(
        base_estimator=LinearRegression(),
        column_specs={"transform_name": "yeo_johnson_y"},
    )
    est.fit(X, y)
    assert 1 in est.failed_columns_
    est.calibrate_conformal(X, y, alpha=0.1)
    lower, upper = est.predict_interval(X, alpha=0.1)
    assert lower.shape == (n, 2)
    # Failed column: degenerate +/- 0 band.
    np.testing.assert_array_equal(lower[:, 1], upper[:, 1])
    # Live column: a real (non-degenerate) band.
    assert (upper[:, 0] > lower[:, 0]).any()


# --------------------------------------------------------------------------
# biz_value: empirical per-column coverage >= 1 - alpha
# --------------------------------------------------------------------------


def test_biz_val_multioutput_conformal_per_column_coverage():
    """On a 3-output target with INDEPENDENT per-column scales, the empirical
    coverage on a fresh test set must be >= 1 - alpha for EVERY column.

    Measured ~0.90-0.95 per column at alpha=0.1; floor set at 0.85 (5pp margin
    below the 0.90 nominal to absorb finite-sample noise). A regression that
    pooled one shared radius would push the widest-noise column (y2, sigma=5)
    below the floor while over-covering the tight column (y0, sigma=0.3)."""
    alpha = 0.1
    est, _, _ = _fit_estimator(n=2000, seed=0)
    X_cal, y_cal = _make_xy(2000, seed=1)
    est.calibrate_conformal(X_cal, y_cal, alpha=alpha)

    X_test, y_test = _make_xy(4000, seed=2)
    lower, upper = est.predict_interval(X_test, alpha=alpha)
    covered = (y_test >= lower) & (y_test <= upper)
    per_col = covered.mean(axis=0)
    for k, cov in enumerate(per_col):
        assert cov >= 0.85, f"column {k} coverage {cov:.3f} below floor 0.85 (nominal {1 - alpha})"
    # And each column's band width tracks its own noise scale (the per-column
    # point of independent calibration): widest-noise column has the widest band.
    widths = (upper - lower).mean(axis=0)
    assert widths[2] > widths[1] > widths[0], f"band widths {widths} should grow with per-column noise scale"
