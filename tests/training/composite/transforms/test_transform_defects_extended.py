"""Regression tests for scale-, offset- and domain-dependent defects in the extended bivariate transforms (``extended.py``).

Each test reproduces a concrete wrong output on the pre-fix code: a constant ``reciprocal_residual`` prediction for large-scale targets, a
``centered_ratio`` sign flip just below the train range, an ill-conditioned quadratic fit on an offset base, silently swallowed spline failures,
an over-restrictive ``geometric_mean_residual`` domain and O(n) ``rank_residual`` params.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np
import pytest

from mlframe.training.composite.transforms import get_transform
from mlframe.utils.log_throttle import reset_throttle_counts


def _round_trip_max_err(name: str, y: np.ndarray, base: np.ndarray) -> float:
    """Fit ``name`` on (y, base) and return the max abs error of ``inverse(forward(y))`` over every row."""
    tr = get_transform(name)
    params = tr.fit(y, base)
    t = tr.forward(y, base, params)
    return float(np.max(np.abs(tr.inverse(t, base, params) - y)))


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3, 1e6])
def test_reciprocal_residual_round_trip_exact_at_every_scale(scale: float) -> None:
    """The inverse's z-floor was in base units, so every prediction collapsed to ``1/eps_b`` once |y| exceeded ~1e6/median|base| (scale >= 1e3)."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.9, 1.1, 500) * scale
    y = base * rng.uniform(1.5, 2.5, 500)
    assert _round_trip_max_err("reciprocal_residual", y, base) < 1e-9 * np.max(np.abs(y))


def test_reciprocal_residual_old_params_without_eps_z_use_the_y_unit_floor() -> None:
    """Params pickled before ``eps_z`` existed must still load and invert with a y-unit floor (derived from ``eps_y``), not the base-unit clamp."""
    rng = np.random.default_rng(1)
    base = rng.uniform(900, 1100, 300)
    y = 2.0 * base
    tr = get_transform("reciprocal_residual")
    params = tr.fit(y, base)
    legacy = {k: v for k, v in params.items() if k != "eps_z"}
    t = tr.forward(y, base, legacy)
    np.testing.assert_allclose(tr.inverse(t, base, legacy), y, rtol=1e-9)


def test_centered_ratio_positive_base_below_train_min_keeps_sign() -> None:
    """A strictly positive base used to be shifted so its train min sat next to the pole; a base 3% below that min then inverted to a negative y."""
    rng = np.random.default_rng(0)
    base = rng.uniform(100, 200, 400)
    y = 3.0 * base + rng.normal(0, 1, 400)
    tr = get_transform("centered_ratio")
    params = tr.fit(y, base)
    t_med = float(np.median(tr.forward(y, base, params)))
    pred = tr.inverse(np.full(3, t_med), np.array([97.0, 98.5, 99.0]), params)
    assert np.all(pred > 0)
    np.testing.assert_allclose(pred, 3.0 * np.array([97.0, 98.5, 99.0]), rtol=0.1)


def test_centered_ratio_fitted_domain_rejects_rows_past_the_pole() -> None:
    """``_centered_ratio_domain_fitted`` accepted any ``|base + c| >= eps``, so a sign-flipped denominator reached the inverse instead of the fallback."""
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 1.0, 400)
    y = 2.0 * base + 10.0
    tr = get_transform("centered_ratio")
    params = tr.fit(y, base)
    c = float(params["c"])
    probe = np.array([-c - 5.0, -c - 0.5, -c + 5.0])
    ok = tr.domain_check_fitted(None, probe, params)
    assert ok.tolist() == [False, False, True]


@pytest.mark.parametrize("center,spread", [(1e4, 1.0), (1e5, 1.0), (1e6, 10.0)])
def test_polynomial_residual_deg2_absorbs_curvature_on_offset_base(center: float, spread: float) -> None:
    """Raw uncentred normal equations left about a third of the signal in T once the base was offset from zero (std(T) ~0.6 vs noise 0.01)."""
    rng = np.random.default_rng(0)
    b = center + spread * rng.standard_normal(2000)
    x = (b - center) / spread
    y = 0.5 * x * x + 2.0 * x + rng.normal(0, 0.01, 2000)
    tr = get_transform("polynomial_residual_deg2")
    params = tr.fit(y, b)
    assert float(np.std(tr.forward(y, b, params))) < 0.05
    np.testing.assert_allclose(tr.inverse(tr.forward(y, b, params), b, params), y, atol=1e-9 * max(1.0, float(np.max(np.abs(y)))))


def test_polynomial_residual_deg2_small_n_fallback_is_mean_not_diff() -> None:
    """The < 10 finite-row fallback was ``alpha1 = 1`` (``T = y - base``); every sibling fallback is zero slope around the mean."""
    tr = get_transform("polynomial_residual_deg2")
    y = np.array([100.0, 101.0, 99.0, 100.0])
    base = np.array([1.0, 2.0, 3.0, 4.0])
    params = tr.fit(y, base)
    np.testing.assert_allclose(tr.inverse(np.zeros(2), np.array([50.0, -50.0]), params), [100.0, 100.0])


def test_polynomial_residual_deg2_legacy_raw_params_still_invert() -> None:
    """Params pickled before centring carry only ``alpha1/alpha2/beta``; they must keep evaluating on the raw base."""
    tr = get_transform("polynomial_residual_deg2")
    legacy = {"alpha1": 2.0, "alpha2": 0.5, "beta": 1.0}
    b = np.array([0.0, 1.0, 2.0])
    np.testing.assert_allclose(tr.inverse(np.zeros(3), b, legacy), [1.0, 3.5, 7.0])


def test_smoothing_spline_build_failure_is_flagged_and_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A spline build failure was swallowed at DEBUG and turned the transform into ``T = y - mean(y)`` with no signal to discovery."""
    import scipy.interpolate

    def _boom(*args: object, **kwargs: object) -> None:
        """Stand-in for ``UnivariateSpline`` that always fails to build."""
        raise ValueError("forced spline failure")

    rng = np.random.default_rng(0)
    base = rng.uniform(0, 10, 300)
    y = np.sin(base) + rng.normal(0, 0.1, 300)
    monkeypatch.setattr(scipy.interpolate, "UnivariateSpline", _boom)
    reset_throttle_counts("smoothing_spline_build_failed")
    with caplog.at_level(logging.WARNING):
        params = get_transform("smoothing_spline_residual").fit(y, base)
    assert params["is_degenerate"] is True
    assert any("UnivariateSpline build failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


def test_smoothing_spline_healthy_fit_is_not_flagged() -> None:
    """The degeneracy flag must stay False on a normal fit, otherwise discovery would drop every smoothing-spline spec."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 10, 300)
    y = np.sin(base) + rng.normal(0, 0.1, 300)
    assert get_transform("smoothing_spline_residual").fit(y, base)["is_degenerate"] is False


def test_geometric_mean_residual_accepts_zero_and_negative_y() -> None:
    """``T = y / geomean(bases)`` is defined for any finite y; the domain used to drop every ``y <= 0`` row (a zero-inflated target lost all zeros)."""
    rng = np.random.default_rng(0)
    base = rng.uniform(1.0, 5.0, (200, 2))
    y = rng.normal(0.0, 3.0, 200)
    y[:20] = 0.0
    tr = get_transform("geometric_mean_residual")
    assert tr.domain_check(y, base).all()
    params = tr.fit(y, base)
    np.testing.assert_allclose(tr.inverse(tr.forward(y, base, params), base, params), y, atol=1e-12)


def test_rank_residual_params_size_is_bounded() -> None:
    """``rank_residual`` stored both full sorted train arrays, so the pickled params grew linearly with n (10x from 1e4 to 1e5 rows)."""
    rng = np.random.default_rng(0)
    sizes = []
    for n in (10_000, 100_000):
        base = rng.standard_normal(n)
        y = base + rng.standard_normal(n)
        sizes.append(len(pickle.dumps(get_transform("rank_residual").fit(y, base))))
    assert sizes[1] <= 2 * sizes[0]
    assert sizes[1] < 1_000_000


@pytest.mark.parametrize("n", [50, 5000, 50_000])
def test_rank_residual_round_trip_exact_with_bounded_knots(n: int) -> None:
    """The bounded knot table must still invert every train row exactly, including tied values."""
    rng = np.random.default_rng(n)
    base = rng.standard_normal(n)
    y = np.round(base + rng.standard_normal(n), 1)
    assert _round_trip_max_err("rank_residual", y, base) < 1e-9 * max(1.0, float(np.max(np.abs(y))))


def test_rank_residual_legacy_sorted_array_params_still_invert() -> None:
    """Params pickled with ``y_sorted`` / ``b_sorted`` keep their original nearest-bucket lookup."""
    tr = get_transform("rank_residual")
    y_sorted = np.arange(10.0)
    legacy = {"y_sorted": y_sorted, "b_sorted": np.arange(10.0), "alpha": 1.0, "beta": 0.0}
    t = tr.forward(y_sorted, y_sorted, legacy)
    np.testing.assert_allclose(tr.inverse(t, y_sorted, legacy), y_sorted)
