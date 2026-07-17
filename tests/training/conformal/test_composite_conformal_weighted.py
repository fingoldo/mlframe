"""Unit + biz_value tests for weighted (covariate-shift) split-conformal.

Headline guarantees:
- Under a deliberate covariate shift between calibration and test (the base /
  feature law is shifted at test time), the Tibshirani-et-al. importance-weighted
  band keeps marginal coverage CLOSER to 1-alpha than the unweighted split band,
  which mis-covers because cal/test residuals are no longer exchangeable.
- Uniform weights reduce EXACTLY to the unweighted finite-sample radius.
- Degenerate weights (a single dominating point) and tiny-n inputs never crash
  and never silently mis-cover (return +inf valid-but-uninformative bands).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.conformal import (
    conformal_quantile,
    weighted_conformal_quantile,
)


# --------------------------------------------------------------------------- #
# Pure-function unit tests of the weighted quantile.
# --------------------------------------------------------------------------- #
class TestWeightedQuantileUnit:
    def test_uniform_weights_equal_unweighted(self) -> None:
        rng = np.random.default_rng(0)
        r = rng.normal(size=500)
        w = np.ones(500)
        for alpha in (0.05, 0.1, 0.2):
            assert weighted_conformal_quantile(r, w, alpha) == pytest.approx(conformal_quantile(r, alpha))

    def test_scaling_weights_is_invariant(self) -> None:
        # Importance weights are used only after normalisation; a global scale
        # must not change the radius.
        rng = np.random.default_rng(1)
        r = rng.normal(size=300)
        w = rng.uniform(0.1, 2.0, size=300)
        base = weighted_conformal_quantile(r, w, 0.1)
        assert weighted_conformal_quantile(r, 7.5 * w, 0.1) == pytest.approx(base)

    def test_upweighting_large_residuals_widens_band(self) -> None:
        # Putting more test-law mass on the high-residual rows must not shrink
        # the band below the uniform one.
        r = np.array([0.1, 0.2, 0.3, 5.0, 6.0, 7.0])
        uni = weighted_conformal_quantile(r, np.ones(6), 0.1)
        heavy = weighted_conformal_quantile(
            r,
            np.array([0.1, 0.1, 0.1, 5.0, 5.0, 5.0]),
            0.1,
        )
        assert heavy >= uni

    def test_degenerate_single_dominating_weight(self) -> None:
        # One point carries essentially all the mass; no crash, finite-or-inf.
        r = np.array([0.5, 1.0, 2.0, 3.0])
        w = np.array([1e6, 1e-6, 1e-6, 1e-6])
        out = weighted_conformal_quantile(r, w, 0.1)
        assert np.isfinite(out) or out == float("inf")

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_tiny_n_returns_inf(self, n: int) -> None:
        r = np.arange(float(n))
        w = np.ones(n)
        assert weighted_conformal_quantile(r, w, 0.1) == float("inf")

    def test_all_zero_weights_is_inf(self) -> None:
        assert weighted_conformal_quantile(
            np.array([1.0, 2.0, 3.0]),
            np.zeros(3),
            0.1,
        ) == float("inf")

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            weighted_conformal_quantile(np.array([1.0, 2.0]), np.array([1.0, -1.0]), 0.1)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="weights"):
            weighted_conformal_quantile(np.array([1.0, 2.0, 3.0]), np.array([1.0]), 0.1)

    def test_nonfinite_residuals_dropped(self) -> None:
        r = np.array([1.0, np.nan, 2.0, np.inf, 3.0])
        w = np.ones(5)
        # Drops the 2 non-finite -> 3 finite residuals, no crash.
        out = weighted_conformal_quantile(r, w, 0.5)
        assert np.isfinite(out)


# --------------------------------------------------------------------------- #
# Estimator-bound calibrate / predict.
# --------------------------------------------------------------------------- #
def _fit(seed: int, n: int = 600):
    rng = np.random.default_rng(seed)
    b = rng.normal(0.0, 1.0, n)
    f = rng.normal(0.0, 1.0, n)
    y = b + 0.5 * f + rng.normal(0.0, 1.0, n)
    X = pd.DataFrame({"b": b, "feat": f})
    est = CompositeTargetEstimator(
        base_estimator=LinearRegression(),
        transform_name="linear_residual",
        base_column="b",
    ).fit(X, y)
    return est


class TestWeightedBound:
    def test_uniform_weights_match_unweighted_band(self) -> None:
        est = _fit(0)
        rng = np.random.default_rng(5)
        Xc = pd.DataFrame({"b": rng.normal(size=400), "feat": rng.normal(size=400)})
        yc = Xc["b"].to_numpy() + 0.5 * Xc["feat"].to_numpy() + rng.normal(size=400)
        # Weighted conformal uses the absolute-residual score, so compare against the
        # absolute (constant-width) unweighted band -- the default is now normalized.
        est.calibrate_conformal(Xc, yc, 0.1, score="absolute")
        est.calibrate_conformal_weighted(Xc, yc, 0.1, weights=None)
        assert est._weighted_conformal_q_[round(0.1, 6)] == pytest.approx(est._conformal_q_[round(0.1, 6)])
        lo_u, hi_u = est.predict_interval(Xc, 0.1)
        lo_w, hi_w = est.predict_interval_weighted(Xc, 0.1)
        np.testing.assert_allclose(lo_u, lo_w)
        np.testing.assert_allclose(hi_u, hi_w)

    def test_callable_density_ratio_estimator(self) -> None:
        est = _fit(1)
        rng = np.random.default_rng(6)
        Xc = pd.DataFrame({"b": rng.normal(size=400), "feat": rng.normal(size=400)})
        yc = Xc["b"].to_numpy() + rng.normal(size=400)
        # Callable returns one density ratio per cal row from X_cal.
        est.calibrate_conformal_weighted(
            Xc,
            yc,
            0.1,
            weights=lambda X: np.exp(0.3 * X["b"].to_numpy()),
        )
        assert round(0.1, 6) in est._weighted_conformal_q_
        lo, hi = est.predict_interval_weighted(Xc.iloc[:3], 0.1)
        assert lo.shape == (3,) and hi.shape == (3,) and np.all(hi >= lo)

    def test_predict_without_calibration_raises(self) -> None:
        est = _fit(0)
        with pytest.raises(RuntimeError, match="no weighted conformal radius"):
            est.predict_interval_weighted(
                pd.DataFrame({"b": [0.0], "feat": [0.0]}),
                0.1,
            )

    def test_calibrate_before_fit_raises(self) -> None:
        from sklearn.exceptions import NotFittedError

        est = CompositeTargetEstimator(
            base_estimator=LinearRegression(),
            transform_name="linear_residual",
            base_column="b",
        )
        X = pd.DataFrame({"b": [1.0, 2.0], "feat": [3.0, 4.0]})
        with pytest.raises(NotFittedError):
            est.calibrate_conformal_weighted(X, np.array([1.0, 2.0]), 0.1, weights=None)

    def test_single_cal_row_no_crash(self) -> None:
        est = _fit(0)
        Xc = pd.DataFrame({"b": [0.3], "feat": [0.1]})
        est.calibrate_conformal_weighted(Xc, np.array([0.5]), 0.1, weights=np.array([1.0]))
        assert est._weighted_conformal_q_[round(0.1, 6)] == float("inf")
        lo, hi = est.predict_interval_weighted(Xc, 0.1)
        assert lo.shape == (1,) and hi.shape == (1,) and hi[0] >= lo[0]

    def test_weight_length_mismatch_raises(self) -> None:
        est = _fit(0)
        Xc = pd.DataFrame({"b": [0.3, 0.4], "feat": [0.1, 0.2]})
        with pytest.raises(ValueError, match="entries but"):
            est.calibrate_conformal_weighted(
                Xc,
                np.array([0.5, 0.6]),
                0.1,
                weights=np.array([1.0]),
            )


# --------------------------------------------------------------------------- #
# biz_value: weighted band restores coverage under covariate shift.
# --------------------------------------------------------------------------- #
def _shifted_synthetic(seed: int):
    """Heteroscedastic-in-``b`` target with a covariate shift between the
    calibration law (``b`` centred low) and the test law (``b`` centred high).

    Residual scale grows with ``b``, so when the test set lives at higher ``b``
    its residuals are larger than the (low-``b``) calibration residuals; the
    unweighted band -- sized on small cal residuals -- under-covers. The exact
    density ratio dP_test/dP_cal is known in closed form (both Gaussians in
    ``b``), so the importance weights are exact."""
    rng = np.random.default_rng(seed)
    n_cal, n_test = 8000, 8000
    mu_cal, mu_test, sd = -1.2, 1.2, 1.0

    def gen(n, mu):
        b = rng.normal(mu, sd, n)
        f = rng.normal(0.0, 1.0, n)
        # residual scale rises MONOTONICALLY with b (not |b|): low-b calibration
        # rows have small residuals, high-b test rows have large ones, so the
        # unweighted band sized on cal residuals genuinely under-covers test.
        scale = np.maximum(0.2 + 0.8 * (b + 2.0), 0.2)
        y = b + 0.5 * f + rng.normal(0.0, 1.0, n) * scale
        return pd.DataFrame({"b": b, "feat": f}), y

    Xc, yc = gen(n_cal, mu_cal)
    Xt, yt = gen(n_test, mu_test)
    # Exact density ratio of two equal-variance Gaussians in b.
    bc = Xc["b"].to_numpy()
    w = np.exp(((bc - mu_cal) ** 2 - (bc - mu_test) ** 2) / (2.0 * sd**2))
    return Xc, yc, Xt, yt, w


class TestWeightedBizValue:
    def test_biz_weighted_coverage_closer_under_shift(self) -> None:
        """Across seeds, the weighted band's marginal coverage on the shifted
        test set must be CLOSER to 1-alpha than the unweighted band, AND the
        unweighted band must actually under-cover (proving the shift bites)."""
        alpha = 0.1
        target = 1.0 - alpha
        unw_errs, wtd_errs = [], []
        unw_covs = []
        for seed in range(4):
            Xc, yc, Xt, yt, w = _shifted_synthetic(seed)
            # Train on the calibration law so test is genuinely shifted.
            ntr = len(yc) // 2
            est = CompositeTargetEstimator(
                base_estimator=LinearRegression(),
                transform_name="linear_residual",
                base_column="b",
            ).fit(Xc.iloc[:ntr], yc[:ntr])
            Xcal, ycal, wcal = Xc.iloc[ntr:], yc[ntr:], w[ntr:]

            est.calibrate_conformal(Xcal, ycal, alpha)
            est.calibrate_conformal_weighted(Xcal, ycal, alpha, weights=wcal)

            lo_u, hi_u = est.predict_interval(Xt, alpha)
            cov_u = float(np.mean((yt >= lo_u) & (yt <= hi_u)))
            lo_w, hi_w = est.predict_interval_weighted(Xt, alpha)
            cov_w = float(np.mean((yt >= lo_w) & (yt <= hi_w)))

            unw_errs.append(abs(cov_u - target))
            wtd_errs.append(abs(cov_w - target))
            unw_covs.append(cov_u)

        mean_unw = float(np.mean(unw_errs))
        mean_wtd = float(np.mean(wtd_errs))
        # The shift must actually hurt the unweighted band on average.
        assert float(np.mean(unw_covs)) < target - 0.01, f"unweighted band did not under-cover under shift: mean cov={np.mean(unw_covs):.3f}"
        # Weighted must be measurably closer to the target.
        assert mean_wtd < mean_unw - 0.005, f"weighted not closer to target under shift: wtd_err={mean_wtd:.3f} unw_err={mean_unw:.3f}"
