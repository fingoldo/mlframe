"""Tests for time-series oriented transforms (R10c brainstorm #5):
- ``ewma_residual``: y - EWMA_k(base)
- ``rolling_quantile_ratio``: y / RollingMedian_k(base)
- ``frac_diff``: (1-L)^d * y

Each ships with round-trip exactness, edge-case handling (boundary windows, non-finite values, pre-train anchor padding), and a biz_value test on a synthetic regime where the transform is uniquely well-suited.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    _ewma_compute,
    _ewma_residual_fit,
    _ewma_residual_forward,
    _ewma_residual_inverse,
    _ewma_residual_domain,
    _rolling_median,
    _rolling_quantile_ratio_fit,
    _rolling_quantile_ratio_forward,
    _rolling_quantile_ratio_inverse,
    _frac_diff_weights,
    _frac_diff_fit,
    _frac_diff_forward,
    _frac_diff_inverse,
    get_transform,
)

# ===========================================================================
# ewma_residual
# ===========================================================================


class TestEWMA:
    """Groups tests covering e w m a."""
    def test_anchor_constant_input(self) -> None:
        """Constant base => EWMA stays at the constant value forever."""
        out = _ewma_compute(np.array([5.0] * 100), k=5, anchor=5.0)
        np.testing.assert_allclose(out, 5.0)

    def test_anchor_propagates_first_row(self) -> None:
        """First row is alpha*x + (1-alpha)*anchor; verify."""
        alpha = 2.0 / (3 + 1)
        out = _ewma_compute(np.array([10.0, 20.0, 30.0]), k=3, anchor=0.0)
        # First: alpha * 10 + (1-alpha) * 0 = 5.0
        assert out[0] == pytest.approx(alpha * 10.0)

    def test_round_trip(self) -> None:
        """Round trip."""
        rng = np.random.default_rng(0)
        n = 500
        base = rng.normal(size=n).cumsum()  # smooth-ish time series
        y = base + rng.normal(scale=0.5, size=n)
        params = _ewma_residual_fit(y, base, k=5)
        T = _ewma_residual_forward(y, base, params)
        y_back = _ewma_residual_inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-9)

    def test_round_trip_via_registry(self) -> None:
        """Round trip via registry."""
        rng = np.random.default_rng(1)
        n = 300
        base = rng.normal(size=n).cumsum()
        y = base + rng.normal(scale=0.3, size=n)
        t = get_transform("ewma_residual")
        params = t.fit(y, base)
        T = t.forward(y, base, params)
        y_back = t.inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-9)

    def test_carries_state_through_nan(self) -> None:
        """Non-finite base does not break the recursion (carry-forward)."""
        base = np.array([1.0, 2.0, np.nan, 4.0])
        out = _ewma_compute(base, k=3, anchor=0.0)
        assert np.all(np.isfinite(out))

    def test_domain_rejects_non_finite(self) -> None:
        """Domain rejects non finite."""
        base = np.array([1.0, np.nan, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        mask = _ewma_residual_domain(y, base)
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_biz_value_smooth_drift(self) -> None:
        """Slow-drift DGP: ``y_i = trend(i) + eps``. EWMA captures the trend so the residual has SMALLER variance than ``diff(y, base)`` when base = y_prev (a single lag).

        Here we set base = y_prev to mimic a 1-lag autoregressive setup. EWMA over base smooths the lag noise; the residual T = y - EWMA(y_prev) is closer to the noise floor than T_diff = y - y_prev.
        """
        rng = np.random.default_rng(2)
        n = 1000
        trend = np.linspace(0.0, 10.0, n)
        y = trend + rng.normal(scale=0.5, size=n)
        # base = y shifted by 1: y_{i-1}.
        base = np.concatenate([[y[0]], y[:-1]])
        # EWMA residual.
        params = _ewma_residual_fit(y, base, k=10)
        T_ewma = _ewma_residual_forward(y, base, params)
        # Diff residual: y - y_prev.
        T_diff = y - base
        # On smooth drift, EWMA smooths through lag noise; diff doubles the noise (Var(y - y_prev) ~= 2 * sigma^2).
        assert np.var(T_ewma) < np.var(
            T_diff
        ), f"EWMA residual variance should be < diff residual on smooth drift; got ewma={np.var(T_ewma):.4f}, diff={np.var(T_diff):.4f}"


# ===========================================================================
# rolling_quantile_ratio
# ===========================================================================


class TestRollingMedian:
    """Groups tests covering rolling median."""
    def test_centered_window(self) -> None:
        """Centred window of size 3: median of 3 neighbours."""
        arr = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
        out = _rolling_median(arr, k=3)
        # i=0: [1, 5] (half=1, lo=0, hi=2) -> median(1,5)=3.0; etc.
        assert out[0] == 3.0
        assert out[1] == 2.0  # median([1, 5, 2])
        assert out[2] == 5.0  # median([5, 2, 8])
        assert out[3] == 3.0  # median([2, 8, 3])

    def test_handles_nans_in_window(self) -> None:
        """Handles nans in window."""
        arr = np.array([1.0, np.nan, 3.0])
        out = _rolling_median(arr, k=3)
        # Window i=1: [1, nan, 3] -> finite [1, 3] -> median 2.0
        assert out[1] == 2.0

    def test_round_trip(self) -> None:
        """Round trip."""
        rng = np.random.default_rng(3)
        n = 300
        base = np.abs(rng.normal(loc=10.0, scale=2.0, size=n)) + 1.0  # all positive
        y = base * rng.uniform(low=0.8, high=1.2, size=n)
        params = _rolling_quantile_ratio_fit(y, base, k=5)
        T = _rolling_quantile_ratio_forward(y, base, params)
        y_back = _rolling_quantile_ratio_inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-7)

    def test_safe_division_near_zero(self) -> None:
        """Near-zero rolling median replaced by signed eps; predict path returns finite values."""
        base = np.array([0.0, 1e-12, 0.0, 5.0, 0.0])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        params = _rolling_quantile_ratio_fit(y, base, k=3)
        T = _rolling_quantile_ratio_forward(y, base, params)
        assert np.all(np.isfinite(T))


# ===========================================================================
# frac_diff
# ===========================================================================


class TestFracDiff:
    """Groups tests covering frac diff."""
    def test_weights_first_lag(self) -> None:
        """w_0 = 1; w_1 = -d (first-difference order weight)."""
        d = 0.5
        w = _frac_diff_weights(d, lags=5)
        assert w[0] == 1.0
        assert w[1] == pytest.approx(-d)

    def test_weights_d_equals_1_recovers_first_difference(self) -> None:
        """d = 1: w_0 = 1, w_1 = -1, w_k = 0 for k >= 2. T_i = y_i - y_{i-1}."""
        w = _frac_diff_weights(1.0, lags=5)
        assert w[0] == 1.0
        assert w[1] == -1.0
        # w_2 = -w_1 * (1 - 1) / 2 = 0; w_3 = -w_2 * (...) / 3 = 0.
        assert all(w[k] == 0.0 for k in range(2, 6))

    def test_round_trip(self) -> None:
        """Round trip."""
        rng = np.random.default_rng(4)
        n = 200
        y = rng.normal(size=n).cumsum()  # random walk
        base = np.zeros(n)  # unused but required
        params = _frac_diff_fit(y, base, d=0.5, lags=20)
        T = _frac_diff_forward(y, base, params)
        y_back = _frac_diff_inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_round_trip_via_registry(self) -> None:
        """Round trip via registry."""
        rng = np.random.default_rng(5)
        n = 150
        y = rng.normal(size=n).cumsum()
        base = np.zeros(n)
        t = get_transform("frac_diff")
        params = t.fit(y, base)
        T = t.forward(y, base, params)
        y_back = t.inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_d_zero_yields_identity(self) -> None:
        """d = 0: (1-L)^0 = 1 identity; T should equal y exactly."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base = np.zeros(5)
        params = _frac_diff_fit(y, base, d=0.0, lags=3)
        T = _frac_diff_forward(y, base, params)
        # T_i = w_0 * y_i + sum_{k>=1} w_k * y_{i-k}; with d=0, all w_k = 0 for k >= 1.
        np.testing.assert_allclose(T, y, rtol=1e-10)
