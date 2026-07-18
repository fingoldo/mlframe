"""Biz_value + unit tests for M8 split-conformal prediction intervals.

The headline guarantee: a calibrated conformal band achieves marginal coverage
>= 1 - alpha on exchangeable held-out rows, for any inner model, with no
distributional assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.conformal import conformal_quantile


class TestConformalQuantile:
    """Groups tests covering conformal quantile."""
    def test_finite_sample_rank(self) -> None:
        # 99 residuals 1..99; alpha=0.1 -> rank ceil(100*0.9)=90 -> value 90.
        """Finite sample rank."""
        r = np.arange(1, 100, dtype=float)
        assert conformal_quantile(r, 0.1) == pytest.approx(90.0)

    def test_too_few_points_returns_inf(self) -> None:
        # n=5, alpha=0.1 -> rank ceil(6*0.9)=6 > 5 -> inf (valid, uninformative).
        """Too few points returns inf."""
        assert conformal_quantile(np.arange(5.0), 0.1) == float("inf")

    def test_empty_returns_inf(self) -> None:
        """Empty returns inf."""
        assert conformal_quantile(np.array([]), 0.1) == float("inf")

    def test_alpha_out_of_range_raises(self) -> None:
        """Alpha out of range raises."""
        with pytest.raises(ValueError):
            conformal_quantile(np.arange(10.0), 1.5)


def _fit_calibrate(seed, alpha=0.1, n=3000):
    """Fit calibrate."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = b + 0.5 * f + rng.normal(0.0, 1.0, n)
    X = pd.DataFrame({"b": b, "feat": f})
    nf = n // 3
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="b",
    )
    est.fit(X.iloc[:nf], y[:nf])
    est.calibrate_conformal(X.iloc[nf : 2 * nf], y[nf : 2 * nf], alpha=alpha)
    lo, hi = est.predict_interval(X.iloc[2 * nf :], alpha)
    cov = float(np.mean((y[2 * nf :] >= lo) & (y[2 * nf :] <= hi)))
    return cov, float(np.mean(hi - lo))


class TestConformalCoverage:
    """Groups tests covering conformal coverage."""
    def test_biz_marginal_coverage_at_least_1_minus_alpha(self) -> None:
        """Across seeds the calibrated band covers >= 1-alpha (with a small
        finite-sample slack)."""
        covs = [_fit_calibrate(s, 0.1)[0] for s in range(5)]
        mean_cov = float(np.mean(covs))
        assert mean_cov >= 0.88, f"conformal under-covered: mean {mean_cov:.3f}"

    def test_tighter_alpha_widens_band(self) -> None:
        """Tighter alpha widens band."""
        _, w10 = _fit_calibrate(0, 0.10)
        _, w02 = _fit_calibrate(0, 0.02)
        assert w02 > w10, "alpha=0.02 band must be wider than alpha=0.10"


class TestConformalErrors:
    """Groups tests covering conformal errors."""
    def test_predict_interval_without_calibration_raises(self) -> None:
        """Predict interval without calibration raises."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"b": rng.normal(size=200), "feat": rng.normal(size=200)})
        y = X["b"].to_numpy() + rng.normal(size=200)
        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        ).fit(X, y)
        with pytest.raises(RuntimeError, match="no conformal radius calibrated"):
            est.predict_interval(X, 0.1)

    def test_calibrate_before_fit_raises(self) -> None:
        """Calibrate before fit raises."""
        from sklearn.exceptions import NotFittedError

        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        )
        X = pd.DataFrame({"b": [1.0, 2.0], "feat": [3.0, 4.0]})
        with pytest.raises(NotFittedError):
            est.calibrate_conformal(X, np.array([1.0, 2.0]), 0.1)
