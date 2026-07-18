"""biz_value: ``fe_hermite_l2_penalty`` (+ its saturation regime) -- the coefficient-magnitude penalty that lets a
genuine high-coefficient separable-polynomial reconstruction win over a deceptive small-||c|| plateau.

``fe_hermite_l2_penalty`` weights the coefficient penalty subtracted from the raw MI objective in the smart_polynom
pair optimiser (wired: ``_mrmr_fe_step/_step_core.py`` -> ``polynom_pair_fe`` -> ``optimise_hermite_pair`` ->
``hermite_fe._l2_penalty_value``). The DOCUMENTED win (pre_distortion suite F-POLY fix) is the SATURATING regime:
``lambda * s/(s+sat)`` rises toward a constant ceiling ``lambda`` as ``s = ||c||^2`` grows, so a genuine separable
reconstruction of ``(a**3-2a)(b**2-b)`` (``||c||^2 ~ 86``, MI ~ 1.5) is NOT crushed, while the legacy RAW penalty
``lambda * ||c||^2`` grows without bound and DOES crush it -- driving the optimiser to a low-||c|| atan2/div plateau.

These gates pin the penalty mechanism's quantitative win directly on ``_l2_penalty_value`` (the full smart_polynom A/B
is seed-sensitive + heavy because the optimiser L2-normalises coefficients at that layer -- the mechanism IS the lever,
so it is pinned here deterministically, the same shape the project uses for other penalty/gate mechanisms):
  A. on a genuine high-coefficient solution (``||c||^2 ~ 86``) the SATURATING penalty stays BELOW the MI peak (~1.5) so
     the solution survives, while the legacy RAW penalty EXCEEDS it many-fold (would crush it).
  B. on noise-scale coefficients (``||c||^2`` small) both regimes pay a small, comparable penalty (no spurious split).
  C. ``fe_hermite_l2_penalty <= 0`` disables the penalty entirely (0.0) in both regimes.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import _l2_penalty_value


LAMBDA = 0.05  # the fe_hermite_l2_penalty default
SAT_DEFAULT = 1.0  # _L2_PENALTY_SATURATION_DEFAULT
MI_PEAK = 1.5  # measured MI of the genuine F-POLY reconstruction


def _genuine_high_coef():
    # ||c_a||^2 + ||c_b||^2 ~ 86, the measured magnitude of the separable Chebyshev reconstruction of (a**3-2a)(b**2-b).
    """Genuine high coef."""
    ca = np.array([0.0, -2.0, 0.0, 6.0, 0.0, 0.0], dtype=np.float64)  # ||.||^2 = 40
    cb = np.array([0.0, -1.0, 6.78, 0.0, 0.0, 0.0], dtype=np.float64)  # ||.||^2 ~ 46
    s = float(np.sum(ca**2) + np.sum(cb**2))
    assert 80.0 <= s <= 92.0, s
    return ca, cb


def _noise_scale_coef():
    """Noise scale coef."""
    rng = np.random.default_rng(0)
    ca = 0.1 * rng.standard_normal(6)
    cb = 0.1 * rng.standard_normal(6)
    return ca, cb


def test_saturating_penalty_preserves_genuine_high_coef_solution_raw_crushes_it():
    """Saturating penalty preserves genuine high coef solution raw crushes it."""
    ca, cb = _genuine_high_coef()
    pen_sat = _l2_penalty_value(ca, cb, LAMBDA, l2_penalty_saturation=SAT_DEFAULT)
    pen_raw = _l2_penalty_value(ca, cb, LAMBDA, l2_penalty_saturation=-1.0)
    # Saturating penalty stays well below the MI peak -> the genuine solution survives the objective.
    assert pen_sat < 0.3 * MI_PEAK, f"saturating penalty {pen_sat:.4f} too large to preserve genuine high-coef solution"
    # Legacy raw penalty exceeds the MI peak many-fold -> would crush it.
    assert pen_raw > 2.0 * MI_PEAK, f"raw penalty {pen_raw:.4f} should dwarf MI peak {MI_PEAK} (the historical crush)"
    assert pen_raw > 8.0 * pen_sat, f"raw/sat separation too small: raw={pen_raw:.4f} sat={pen_sat:.4f}"


def test_noise_scale_coef_both_regimes_pay_small_comparable_penalty():
    """Noise scale coef both regimes pay small comparable penalty."""
    ca, cb = _noise_scale_coef()
    pen_sat = _l2_penalty_value(ca, cb, LAMBDA, l2_penalty_saturation=SAT_DEFAULT)
    pen_raw = _l2_penalty_value(ca, cb, LAMBDA, l2_penalty_saturation=-1.0)
    # Both tiny relative to the MI peak; no spurious split that would mis-rank a noise candidate.
    assert pen_sat < 0.05 * MI_PEAK and pen_raw < 0.05 * MI_PEAK, (pen_sat, pen_raw)
    assert abs(pen_sat - pen_raw) < 0.02, f"noise-scale penalties should be comparable: sat={pen_sat:.5f} raw={pen_raw:.5f}"


def test_penalty_disabled_when_lambda_nonpositive():
    """Penalty disabled when lambda nonpositive."""
    ca, cb = _genuine_high_coef()
    assert _l2_penalty_value(ca, cb, 0.0, l2_penalty_saturation=SAT_DEFAULT) == 0.0
    assert _l2_penalty_value(ca, cb, -1.0, l2_penalty_saturation=-1.0) == 0.0


def test_saturating_penalty_monotone_and_ceiling_bounded():
    # The saturating penalty rises monotonically with ||c||^2 but never reaches the lambda ceiling.
    """Saturating penalty monotone and ceiling bounded."""
    prev = -1.0
    for scale in (0.1, 1.0, 5.0, 50.0, 500.0):
        ca = np.full(6, scale, dtype=np.float64)
        cb = np.zeros(6, dtype=np.float64)
        pen = _l2_penalty_value(ca, cb, LAMBDA, l2_penalty_saturation=SAT_DEFAULT)
        assert pen > prev, "saturating penalty must be monotone increasing in ||c||^2"
        assert pen < LAMBDA, f"saturating penalty {pen} must stay below the lambda ceiling {LAMBDA}"
        prev = pen
