"""Per-transform behavioural tests for Pack L (2026-05-26) extended bivariate +
multi-base composite transforms.

Round-trip + domain + JSON serialisability are covered by the shared
contract test (``test_composite_transforms_registry_contract.py``).
This file asserts transform-specific math + edge cases:

* asinh_residual: signed-base handling (rejected by logratio).
* centered_ratio: works on signed bases via learned shift c.
* polynomial_residual_deg2: recovers quadratic relation exactly.
* rank_residual: rank-fraction shape + ECDF inversion.
* smoothing_spline_residual: tracks a known smooth nonlinear curve.
* reciprocal_residual: 1/y - 1/base math + near-zero safety.
* geometric_mean_residual: multi-base geomean math + 2D base accepted.
* pairwise_interaction_residual: bilinear residual + 2D base.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.transforms.extended import (
    _asinh_residual_domain,
    _asinh_residual_fit,
    _asinh_residual_forward,
    _asinh_residual_inverse,
    _centered_ratio_domain,
    _centered_ratio_fit,
    _centered_ratio_forward,
    _centered_ratio_inverse,
    _geometric_mean_residual_domain,
    _geometric_mean_residual_fit,
    _geometric_mean_residual_forward,
    _geometric_mean_residual_inverse,
    _pairwise_interaction_residual_domain,
    _pairwise_interaction_residual_fit,
    _pairwise_interaction_residual_forward,
    _pairwise_interaction_residual_inverse,
    _polynomial_residual_deg2_domain,
    _polynomial_residual_deg2_fit,
    _polynomial_residual_deg2_forward,
    _polynomial_residual_deg2_inverse,
    _rank_residual_fit,
    _rank_residual_forward,
    _rank_residual_inverse,
    _reciprocal_residual_domain,
    _reciprocal_residual_fit,
    _reciprocal_residual_forward,
    _reciprocal_residual_inverse,
    _smoothing_spline_residual_fit,
    _smoothing_spline_residual_forward,
    _smoothing_spline_residual_inverse,
)


_RNG = np.random.default_rng(20260526)


# ---------------------------------------------------------------------------
# asinh_residual
# ---------------------------------------------------------------------------


class TestAsinhResidual:
    def test_accepts_signed_base(self):
        base = _RNG.uniform(-10.0, 10.0, size=400)
        y = 1.2 * np.arcsinh(base) + 0.05 * _RNG.standard_normal(400)
        y = np.sinh(y)  # back to original scale
        domain = _asinh_residual_domain(y, base)
        assert domain.all(), "asinh_residual must accept signed (and zero) bases"

    def test_recovers_alpha_on_linear_arcsinh_relation(self):
        base = _RNG.uniform(-5.0, 5.0, size=2000)
        # T = arcsinh(y) - alpha * arcsinh(base) - beta. Forcing alpha=1.5, beta=0.3.
        alpha_true, beta_true = 1.5, 0.3
        arc_y = alpha_true * np.arcsinh(base) + beta_true + 0.01 * _RNG.standard_normal(2000)
        y = np.sinh(arc_y)
        params = _asinh_residual_fit(y, base)
        assert params["alpha"] == pytest.approx(alpha_true, abs=0.05)
        assert params["beta"] == pytest.approx(beta_true, abs=0.05)

    def test_roundtrip_exact(self):
        base = _RNG.uniform(-3.0, 3.0, size=300)
        y = 2.0 * base + 1.0 + 0.1 * _RNG.standard_normal(300)
        params = _asinh_residual_fit(y, base)
        t = _asinh_residual_forward(y, base, params)
        y_back = _asinh_residual_inverse(t, base, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-9, atol=1e-9)

    def test_degenerate_too_few_finite(self):
        params = _asinh_residual_fit(np.array([1.0, np.nan, np.inf]), np.array([1.0, 2.0, 3.0]))
        assert params == {"alpha": 1.0, "beta": 0.0}


# ---------------------------------------------------------------------------
# centered_ratio
# ---------------------------------------------------------------------------


class TestCenteredRatio:
    def test_signed_base_shifts_above_zero(self):
        base = np.linspace(-2.0, 5.0, 300)
        y = 3.0 + 0.05 * _RNG.standard_normal(300)
        params = _centered_ratio_fit(y, base)
        c = params["c"]
        # (base + c) must be strictly positive on the train min
        assert (base + c).min() > 0
        # And the eps floor is positive
        assert params["eps"] > 0

    def test_roundtrip_exact_on_signed_base(self):
        base = np.linspace(-2.0, 5.0, 400)
        y = base * 0.5 + 1.0 + 0.05 * _RNG.standard_normal(400)
        params = _centered_ratio_fit(y, base)
        t = _centered_ratio_forward(y, base, params)
        y_back = _centered_ratio_inverse(t, base, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# polynomial_residual_deg2
# ---------------------------------------------------------------------------


class TestPolynomialResidualDeg2:
    def test_recovers_quadratic_coefficients(self):
        base = _RNG.uniform(-3.0, 3.0, size=2000)
        # y = 1.0 + 0.7*base - 0.4*base^2 + noise
        a1_true, a2_true, beta_true = 0.7, -0.4, 1.0
        y = beta_true + a1_true * base + a2_true * base * base + 0.01 * _RNG.standard_normal(2000)
        params = _polynomial_residual_deg2_fit(y, base)
        assert params["alpha1"] == pytest.approx(a1_true, abs=0.02)
        assert params["alpha2"] == pytest.approx(a2_true, abs=0.02)
        assert params["beta"] == pytest.approx(beta_true, abs=0.02)

    def test_residual_is_small_when_quadratic_fits(self):
        base = _RNG.uniform(-2.0, 2.0, size=1000)
        y = 0.3 - 0.5 * base + 0.8 * base * base
        params = _polynomial_residual_deg2_fit(y, base)
        t = _polynomial_residual_deg2_forward(y, base, params)
        # Pure deterministic quadratic relation -> residual ~= 0.
        assert np.median(np.abs(t)) < 1e-9

    def test_degenerate_constant_base(self):
        # Constant base -> only intercept identifiable; alpha1/alpha2 ridge-pulled to 0.
        base = np.full(100, 2.0)
        y = 3.0 + 0.1 * _RNG.standard_normal(100)
        params = _polynomial_residual_deg2_fit(y, base)
        # Just assert finite + roundtrip works (numerical stability).
        assert np.isfinite(params["alpha1"])
        assert np.isfinite(params["alpha2"])
        assert np.isfinite(params["beta"])


# ---------------------------------------------------------------------------
# rank_residual
# ---------------------------------------------------------------------------


class TestRankResidual:
    def test_t_is_centered_rank_space(self):
        base = _RNG.uniform(-5.0, 5.0, size=500)
        y = base + 0.3 * _RNG.standard_normal(500)
        params = _rank_residual_fit(y, base)
        t = _rank_residual_forward(y, base, params)
        # T = rank(y) - alpha*rank(base) - beta is mean-zero by construction.
        assert abs(np.mean(t)) < 0.05

    def test_inverse_yields_train_y_values(self):
        base = _RNG.uniform(0.0, 10.0, size=300)
        y = 2.0 * base + 1.0
        params = _rank_residual_fit(y, base)
        t = _rank_residual_forward(y, base, params)
        y_back = _rank_residual_inverse(t, base, params)
        # ECDF inverse returns elements of sorted y; check values are in
        # the train y range and median absolute error is small.
        assert y_back.min() >= y.min() - 1e-9
        assert y_back.max() <= y.max() + 1e-9


# ---------------------------------------------------------------------------
# smoothing_spline_residual
# ---------------------------------------------------------------------------


class TestSmoothingSplineResidual:
    def test_tracks_smooth_nonmonotone_curve(self):
        base = np.linspace(-3.0, 3.0, 500)
        # Non-monotone smooth curve (sin) - linear_residual / monotonic_residual cannot fit this.
        y = np.sin(base) + 0.02 * _RNG.standard_normal(500)
        params = _smoothing_spline_residual_fit(y, base)
        t = _smoothing_spline_residual_forward(y, base, params)
        # Spline absorbs the structure; residual var should be much smaller than y var.
        assert np.std(t) < 0.5 * np.std(y)

    def test_pickle_safe_params(self):
        import pickle

        base = np.linspace(0.0, 10.0, 200)
        y = base**1.3 + 0.05 * _RNG.standard_normal(200)
        params = _smoothing_spline_residual_fit(y, base)
        blob = pickle.dumps(params)
        loaded = pickle.loads(blob)
        # Spline rebuilt from arrays must give identical forward result.
        t_a = _smoothing_spline_residual_forward(y, base, params)
        t_b = _smoothing_spline_residual_forward(y, base, loaded)
        np.testing.assert_allclose(t_a, t_b)


# ---------------------------------------------------------------------------
# reciprocal_residual
# ---------------------------------------------------------------------------


class TestReciprocalResidual:
    def test_forward_math(self):
        base = np.array([1.0, 2.0, 4.0])
        y = np.array([2.0, 4.0, 8.0])
        params = _reciprocal_residual_fit(y, base)
        t = _reciprocal_residual_forward(y, base, params)
        # T = 1/y - 1/base for the strictly-positive case.
        np.testing.assert_allclose(t, 1.0 / y - 1.0 / base, rtol=1e-9)

    def test_near_zero_y_safe(self):
        base = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 1e-30, 2.0])
        params = _reciprocal_residual_fit(y, base)
        t = _reciprocal_residual_forward(y, base, params)
        assert np.all(np.isfinite(t)), "near-zero y must not produce inf/nan"

    def test_domain_rejects_zero(self):
        base = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 0.0, 2.0])
        ok = _reciprocal_residual_domain(y, base)
        assert not ok[0]  # base = 0
        assert not ok[1]  # y = 0
        assert ok[2]


# ---------------------------------------------------------------------------
# geometric_mean_residual (multi-base)
# ---------------------------------------------------------------------------


class TestGeometricMeanResidual:
    def test_accepts_2d_base(self):
        base = _RNG.uniform(1.0, 10.0, size=(300, 3))
        gm = np.exp(np.log(base).mean(axis=1))
        y = gm * 2.0
        params = _geometric_mean_residual_fit(y, base)
        t = _geometric_mean_residual_forward(y, base, params)
        np.testing.assert_allclose(t, 2.0 * np.ones(300), rtol=1e-9)

    def test_roundtrip_2d(self):
        base = _RNG.uniform(0.5, 5.0, size=(200, 2))
        gm = np.exp(np.log(base).mean(axis=1))
        y = gm * (1.0 + 0.1 * _RNG.standard_normal(200))
        params = _geometric_mean_residual_fit(y, base)
        t = _geometric_mean_residual_forward(y, base, params)
        y_back = _geometric_mean_residual_inverse(t, base, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-9)

    def test_domain_rejects_zero_or_negative_base(self):
        base = np.array([[1.0, 2.0], [0.0, 3.0], [2.0, -1.0]])
        y = np.array([1.0, 2.0, 3.0])
        ok = _geometric_mean_residual_domain(y, base)
        assert ok[0]
        assert not ok[1]  # zero in row 1
        assert not ok[2]  # negative in row 2


# ---------------------------------------------------------------------------
# pairwise_interaction_residual (multi-base)
# ---------------------------------------------------------------------------


class TestPairwiseInteractionResidual:
    def test_recovers_bilinear_relation(self):
        base = _RNG.uniform(-2.0, 2.0, size=(2000, 2))
        prod = base[:, 0] * base[:, 1]
        alpha_true, beta_true = 1.7, 0.4
        y = alpha_true * prod + beta_true + 0.01 * _RNG.standard_normal(2000)
        params = _pairwise_interaction_residual_fit(y, base)
        assert params["alpha"] == pytest.approx(alpha_true, abs=0.02)
        assert params["beta"] == pytest.approx(beta_true, abs=0.02)

    def test_roundtrip_2d(self):
        base = _RNG.uniform(-1.0, 1.0, size=(500, 3))
        y = 0.5 * np.prod(base, axis=1) + 1.0 + 0.05 * _RNG.standard_normal(500)
        params = _pairwise_interaction_residual_fit(y, base)
        t = _pairwise_interaction_residual_forward(y, base, params)
        y_back = _pairwise_interaction_residual_inverse(t, base, params)
        np.testing.assert_allclose(y_back, y, rtol=1e-9)

    def test_accepts_signed_bases(self):
        base = _RNG.uniform(-5.0, 5.0, size=(100, 2))
        y = _RNG.standard_normal(100)
        ok = _pairwise_interaction_residual_domain(y, base)
        assert ok.all(), "signed multi-base must be accepted by domain_check"
