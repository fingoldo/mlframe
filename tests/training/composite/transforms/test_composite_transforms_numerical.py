"""Regression sensors for the 2026-06-10 composite-transforms numerical audit.

Each test pins ONE confirmed bug so the failure mode cannot return:

- T1: Yeo-Johnson inverse produced NaN past its lambda-dependent asymptote.
- T2: linear_residual_multi condition gate was scale-variant (independent
  mixed-unit bases falsely rejected as collinear).
- T3: smoothing_spline_residual smoothing factor had wrong units
  (m*std(signal) instead of m*var(noise)) -> systematic oversmoothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.transforms.unary import (
    _yj_inverse_numpy,
    _yj_inverse_scalar,
    _yj_forward_numpy,
)
from mlframe.training.composite.transforms.linear import (
    _linear_residual_multi_fit,
)
from mlframe.training.composite.transforms.extended import (
    _smoothing_spline_residual_fit,
    _smoothing_spline_g,
)


class TestYeoJohnsonInverseFinite:
    @pytest.mark.parametrize("lam", [-2.0, -1.5, -0.87, 2.5, 3.0, 4.0])
    def test_inverse_finite_past_asymptote(self, lam: float) -> None:
        """T1: inverse must be finite for ALL t, including past the
        asymptote at t = -1/lam (lam<0) / the lam>2 mirror. Pre-fix this
        returned NaN, which np.clip could not repair."""
        t = np.linspace(-50.0, 50.0, 1001)
        y = _yj_inverse_numpy(t, lam)
        assert np.all(np.isfinite(y)), f"YJ inverse produced non-finite output for lam={lam}: {int((~np.isfinite(y)).sum())} of {y.size} rows"

    def test_scalar_matches_numpy_past_asymptote(self) -> None:
        lam = -0.87
        ts = [0.0, 1.0, 1.14, 1.15, 1.16, 5.0, 50.0]
        for t in ts:
            scal = _yj_inverse_scalar(t, lam)
            vec = float(_yj_inverse_numpy(np.array([t]), lam)[0])
            assert np.isfinite(scal) and np.isfinite(vec)
            np.testing.assert_allclose(scal, vec, rtol=1e-9, atol=1e-9)

    def test_roundtrip_still_exact_on_valid_domain(self) -> None:
        """The base floor must not perturb the forward/inverse round-trip on
        realistic y (the floor only bites within ~1e-12 of the asymptote)."""
        rng = np.random.default_rng(0)
        for lam in (-1.5, -0.87, 0.5, 1.0, 2.5):
            y = rng.normal(0.0, 3.0, size=2000)
            t = _yj_forward_numpy(y, lam)
            y_back = _yj_inverse_numpy(t, lam)
            np.testing.assert_allclose(y_back, y, rtol=1e-7, atol=1e-7)


class TestLinearResidualMultiScaleInvariance:
    def test_independent_mixed_unit_bases_not_flagged_collinear(self) -> None:
        """T2: two INDEPENDENT bases differing only in scale (N(0,1) and
        N(0,1e6)) must NOT be rejected as collinear. Pre-fix the unscaled
        condition number read ~1e6 > gate -> alphas=[0,0] fallback."""
        rng = np.random.default_rng(1)
        n = 4000
        b1 = rng.normal(0.0, 1.0, size=n)
        b2 = rng.normal(0.0, 1.0e6, size=n)
        base = np.column_stack([b1, b2])
        y = 1.5 * b1 + 2.0e-6 * b2 + rng.normal(0.0, 0.01, size=n)
        params = _linear_residual_multi_fit(y, base)
        assert not params["collinear_fallback"], f"independent mixed-unit bases falsely flagged collinear; cond={params['condition_number']}"
        # Both slopes recovered (not the all-zero fallback).
        alphas = params["alphas"]
        np.testing.assert_allclose(alphas[0], 1.5, rtol=0.05)
        np.testing.assert_allclose(alphas[1], 2.0e-6, rtol=0.05)

    def test_genuinely_collinear_still_flagged(self) -> None:
        """The fix must NOT make the gate blind: a truly collinear pair
        (b2 = 2*b1 + tiny) is still rejected."""
        rng = np.random.default_rng(2)
        n = 4000
        b1 = rng.normal(0.0, 1.0, size=n)
        b2 = 2.0 * b1 + rng.normal(0.0, 1e-9, size=n)
        base = np.column_stack([b1, b2])
        y = b1 + rng.normal(0.0, 0.1, size=n)
        params = _linear_residual_multi_fit(y, base)
        assert params["collinear_fallback"], f"genuinely collinear bases not flagged; cond={params['condition_number']}"


class TestSmoothingSplineNoiseUnits:
    def test_s_in_noise_variance_units_not_signal_std(self) -> None:
        """T3: the smoothing factor must scale with m*var(noise) (~m*0.01=30),
        NOT m*std(signal) (~m*3.5=10000). This is the unambiguous bug
        signature -- pre-fix s was ~362x too large."""
        rng = np.random.default_rng(3)
        n = 3000
        base = np.sort(rng.uniform(-3.0, 3.0, size=n))
        signal = 5.0 * np.sin(3.0 * base)
        y = signal + rng.normal(0.0, 0.1, size=n)
        params = _smoothing_spline_residual_fit(y, base)
        m = int(params["knots_b"].size)
        # var(noise)=0.01 -> s ~= m*0.01; allow generous headroom but well
        # below the m*std(signal) ~= m*3.5 the bug produced.
        assert params["s"] < 0.2 * m, f"smoothing factor s={params['s']:.1f} is in signal-std units (m={m}); expected ~m*var(noise) << {0.2 * m:.0f}"

    def test_residual_approaches_noise_floor(self) -> None:
        """T3: with the correct s = m*var(noise), the spline recovers the
        higher-frequency signal so the residual std is near the noise floor
        (0.1), not ~1.9. Pre-fix the spline oversmoothed (corr 0.886)."""
        rng = np.random.default_rng(3)
        n = 3000
        base = np.sort(rng.uniform(-3.0, 3.0, size=n))
        signal = 5.0 * np.sin(3.0 * base)
        noise_sd = 0.1
        y = signal + rng.normal(0.0, noise_sd, size=n)
        params = _smoothing_spline_residual_fit(y, base)
        g = _smoothing_spline_g(base, params)
        resid_sd = float(np.std(y - g))
        assert resid_sd < 0.5, f"spline oversmoothed: residual std={resid_sd:.3f} (noise floor={noise_sd}); the bug left ~1.9"
        signal_recovery_corr = float(np.corrcoef(g, signal)[0, 1])
        assert signal_recovery_corr > 0.99, f"spline did not recover the signal: corr={signal_recovery_corr:.3f}"
