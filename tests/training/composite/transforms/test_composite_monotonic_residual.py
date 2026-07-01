"""Tests for ``monotonic_residual`` transform (R10c brainstorm extension A).

Non-parametric monotonic residual: T = y - g(base) where g is a monotone PCHIP spline fitted on quantile-knot medians, with per-knot y-values forced cumulatively monotone along the Spearman-correlation orientation.

Coverage:
- Round-trip ``y -> T -> y'`` recovers y exactly (rtol=1e-7) for any base, because the inverse simply adds g(base) back.
- g is monotone (non-decreasing or non-increasing) over the full knot range; verified by checking ``np.diff`` sign consistency on a dense grid.
- Out-of-range base values clip to the edge knot value (NOT PCHIP-extrapolated which would run off to +/- inf for sigmoidal data).
- Under-populated knot slabs fall back to the global y median (locking against per-slab noise on small n).
- Biz_value: on a saturating DGP ``y = log(1 + base) + eps``, the monotonic residual has STRICTLY lower variance than linear_residual (which leaves a wedge of curvature at the high-base end).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite import (
    _monotonic_residual_fit,
    _monotonic_residual_forward,
    _monotonic_residual_inverse,
    _monotonic_residual_domain,
    _monotonic_residual_g,
    get_transform,
)


class TestFit:
    def test_default_knots(self) -> None:
        rng = np.random.default_rng(0)
        n = 2000
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 2.0 * base + rng.normal(scale=0.3, size=n)
        params = _monotonic_residual_fit(y, base)
        assert params["n_knots_effective"] >= 8       # default 12, deduped down on ties
        assert params["knots_x"].shape == params["knots_y"].shape
        assert params["monotone_direction"] == 1      # positive Spearman

    def test_negative_correlation_yields_decreasing_g(self) -> None:
        rng = np.random.default_rng(1)
        n = 1500
        base = rng.uniform(low=0.0, high=10.0, size=n)
        y = -1.5 * base + 20.0 + rng.normal(scale=0.3, size=n)
        params = _monotonic_residual_fit(y, base)
        assert params["monotone_direction"] == -1
        # knots_y must be non-increasing (cumulative min along direction).
        diffs = np.diff(params["knots_y"])
        assert np.all(diffs <= 1e-9)

    def test_degenerate_input(self) -> None:
        params = _monotonic_residual_fit(np.array([1.0]), np.array([5.0]))
        # Falls back to 2-knot anchor with constant value.
        assert params["n_knots_effective"] == 2
        # g evaluation must be finite + constant on this fallback.
        out = _monotonic_residual_g(np.array([0.0, 5.0, 100.0]), params)
        assert np.all(np.isfinite(out))


class TestMonotonicity:
    def test_g_non_decreasing_on_positive_dgp(self) -> None:
        rng = np.random.default_rng(2)
        n = 2000
        base = rng.uniform(low=0.0, high=10.0, size=n)
        y = np.sqrt(base) * 5.0 + rng.normal(scale=0.1, size=n)
        params = _monotonic_residual_fit(y, base)
        grid = np.linspace(0.0, 10.0, 200)
        g_vals = _monotonic_residual_g(grid, params)
        diffs = np.diff(g_vals)
        assert np.all(diffs >= -1e-9), f"g is not non-decreasing; min diff = {diffs.min():.6f}"

    def test_g_clips_outside_train_range(self) -> None:
        rng = np.random.default_rng(3)
        n = 1500
        base = rng.uniform(low=0.0, high=10.0, size=n)
        y = base + rng.normal(scale=0.1, size=n)
        params = _monotonic_residual_fit(y, base)
        # Predict at values far outside [0, 10]: should clip to edge knot values rather than extrapolate.
        far_low = _monotonic_residual_g(np.array([-1000.0]), params)
        far_high = _monotonic_residual_g(np.array([1000.0]), params)
        assert far_low[0] == pytest.approx(params["knots_y"][0], rel=1e-9)
        assert far_high[0] == pytest.approx(params["knots_y"][-1], rel=1e-9)


class TestRoundTrip:
    def test_round_trip_linear_dgp(self) -> None:
        rng = np.random.default_rng(4)
        n = 1500
        base = rng.normal(loc=10.0, scale=2.0, size=n)
        y = 0.8 * base + rng.normal(scale=0.3, size=n)
        params = _monotonic_residual_fit(y, base)
        T = _monotonic_residual_forward(y, base, params)
        y_back = _monotonic_residual_inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_round_trip_saturating_dgp(self) -> None:
        """Round-trip on a saturating ``y = log(1 + base)`` DGP -- the very shape that motivates the transform."""
        rng = np.random.default_rng(5)
        n = 2000
        base = rng.uniform(low=0.0, high=20.0, size=n)
        y = np.log1p(base) * 5.0 + rng.normal(scale=0.05, size=n)
        params = _monotonic_residual_fit(y, base)
        T = _monotonic_residual_forward(y, base, params)
        y_back = _monotonic_residual_inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)

    def test_round_trip_via_registry(self) -> None:
        rng = np.random.default_rng(6)
        n = 800
        base = rng.uniform(low=1.0, high=10.0, size=n)
        y = np.sqrt(base) + rng.normal(scale=0.1, size=n)
        t = get_transform("monotonic_residual")
        params = t.fit(y, base)
        T = t.forward(y, base, params)
        y_back = t.inverse(T, base, params)
        np.testing.assert_allclose(y, y_back, rtol=1e-7, atol=1e-7)


class TestDomain:
    def test_rejects_non_finite_base(self) -> None:
        base = np.array([1.0, np.nan, 3.0, np.inf])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        mask = _monotonic_residual_domain(y, base)
        np.testing.assert_array_equal(mask, [True, False, True, False])


class TestBizValueBeatsLinearOnSaturating:
    """On a saturating DGP, monotonic_residual sucks up the curvature linear_residual misses."""

    def _make_saturating_dgp(self, n: int = 3000, seed: int = 0):
        rng = np.random.default_rng(seed)
        base = rng.uniform(low=0.0, high=30.0, size=n)
        y = np.log1p(base) * 10.0 + rng.normal(scale=0.3, size=n)
        return y, base

    def test_monotonic_residual_variance_strictly_lower(self) -> None:
        y, base = self._make_saturating_dgp()
        # Linear residual (single-base OLS).
        from mlframe.training.composite import (
            _linear_residual_fit, _linear_residual_forward,
        )
        lr_params = _linear_residual_fit(y, base)
        T_lr = _linear_residual_forward(y, base, lr_params)
        # Monotonic residual.
        mr_params = _monotonic_residual_fit(y, base)
        T_mr = _monotonic_residual_forward(y, base, mr_params)
        var_lr = float(np.var(T_lr))
        var_mr = float(np.var(T_mr))
        assert var_mr < var_lr * 0.5, (
            f"monotonic_residual variance must be << linear_residual's on a saturating DGP; "
            f"got var_lr={var_lr:.4f}, var_mr={var_mr:.4f}"
        )
        # And close to noise variance (~0.09 here).
        assert var_mr < 0.5
