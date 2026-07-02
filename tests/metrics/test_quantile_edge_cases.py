"""Edge-case coverage for ``mlframe.metrics.quantile``.

Complements ``test_quantile_degenerate_input_guards`` (which pins the input-validation raises) and
``test_crps_fused_per_alpha_identity`` (which pins the fused-kernel identity) with VALUE-level checks:
pinball at extreme alpha + empty, coverage inclusive-boundary + integer dtype + empty, Winkler known
value + alpha-range validation, CRPS tail-integration branches (hand-computed), and PIT interpolation /
clamping / quantile-crossing warning. Expected numbers are computed independently (by hand and via
sklearn's ``mean_pinball_loss``), never by asserting the function equals itself.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.metrics.quantile import (
    coverage,
    crps_from_quantiles,
    mean_interval_width,
    pinball_loss,
    pit_values,
    winkler_score,
)


# ----------------------------------------------------------------------------
# pinball_loss
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "alpha,expected",
    [
        # L = mean(max(alpha*e, (alpha-1)*e)), e = y - q.
        # y=[1,2,3], q=1.5 -> e=[-0.5,0.5,1.5].
        (0.5, (0.5 + 0.5 + 1.5) / 2 / 3),   # 0.5*mean(|e|) = 0.5*(2.5/3)
        (0.0, (0.5 + 0.0 + 0.0) / 3),        # mean(relu(q-y)) = relu([0.5,-0.5,-1.5]) -> [0.5,0,0]
        (1.0, (0.0 + 0.5 + 1.5) / 3),        # mean(relu(y-q)) = relu([-0.5,0.5,1.5]) -> [0,0.5,1.5]
    ],
)
def test_pinball_loss_value_incl_extreme_alpha(alpha, expected):
    y = np.array([1.0, 2.0, 3.0])
    q = np.full(3, 1.5)
    got = pinball_loss(y, q, alpha)
    assert got == pytest.approx(expected, abs=1e-12)
    # Cross-check against sklearn's reference implementation (independent).
    from sklearn.metrics import mean_pinball_loss

    assert got == pytest.approx(mean_pinball_loss(y, q, alpha=alpha), abs=1e-12)


def test_pinball_loss_empty_returns_zero_and_shape_mismatch_raises():
    assert pinball_loss(np.array([]), np.array([]), 0.5) == 0.0
    with pytest.raises(ValueError, match="shape"):
        pinball_loss(np.array([1.0, 2.0]), np.array([1.0]), 0.5)


# ----------------------------------------------------------------------------
# coverage
# ----------------------------------------------------------------------------


def test_coverage_inclusive_boundary_integer_dtype():
    # coverage is INCLUSIVE: y == q_lo and y == q_hi both count. Integer inputs must work
    # (the wrapper dropped the unconditional float64 cast in iter610).
    y = np.array([1, 9, 10], dtype=np.int64)
    lo = np.array([1, 1, 1], dtype=np.int64)
    hi = np.array([9, 9, 9], dtype=np.int64)
    # row0 y==lo (covered), row1 y==hi (covered), row2 y=10>hi (not) -> 2/3.
    assert coverage(y, lo, hi) == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_coverage_all_inside_and_empty():
    y = np.array([2.0, 5.0, 8.0])
    assert coverage(y, np.full(3, 1.0), np.full(3, 9.0)) == 1.0
    # Empty input -> the n==0 kernel branch returns 0.0 (documented contract, NOT NaN).
    assert coverage(np.array([]), np.array([]), np.array([])) == 0.0


# ----------------------------------------------------------------------------
# winkler_score
# ----------------------------------------------------------------------------


def test_winkler_score_known_value():
    # S = width + (2/alpha)*(q_lo - y)*I(y<q_lo) + (2/alpha)*(y - q_hi)*I(y>q_hi).
    # alpha_miscov=0.2 -> 2/alpha=10; width=8 everywhere.
    y = np.array([0.0, 5.0, 10.0])
    lo = np.full(3, 1.0)
    hi = np.full(3, 9.0)
    # row0: below by 1 -> 8 + 10*1 = 18; row1: inside -> 8; row2: above by 1 -> 8 + 10*1 = 18.
    expected = (18.0 + 8.0 + 18.0) / 3.0
    assert winkler_score(y, lo, hi, 0.2) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
def test_winkler_score_rejects_out_of_range_alpha(bad_alpha):
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="alpha_miscov"):
        winkler_score(y, np.array([0.0, 0.0]), np.array([3.0, 3.0]), bad_alpha)


def test_winkler_score_empty_returns_zero():
    assert winkler_score(np.array([]), np.array([]), np.array([]), 0.2) == 0.0


# ----------------------------------------------------------------------------
# mean_interval_width
# ----------------------------------------------------------------------------


def test_mean_interval_width_value_and_shape_mismatch():
    assert mean_interval_width(np.array([1.0, 2.0]), np.array([3.0, 5.0])) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="mean_interval_width"):
        mean_interval_width(np.array([1.0, 2.0]), np.array([3.0]))


# ----------------------------------------------------------------------------
# crps_from_quantiles
# ----------------------------------------------------------------------------


def test_crps_from_quantiles_tail_integration_known_value():
    """Exercises BOTH constant-extrapolation tail branches (a[0]>0 and a[-1]<1).

    y=[0], preds=[[-1, 1]], alphas=[0.25, 0.75].
    per-alpha pinball = [0.25, 0.25]; inner trapezoid over [0.25,0.75] = 0.125.
    lo-tail (alpha 0..0.25, q=-1): pinball at alpha=0 is 0 -> 0.5*0.25*(0+0.25) = 0.03125.
    hi-tail (alpha 0.75..1, q=1): pinball at alpha=1 is 0 -> 0.5*0.25*(0.25+0) = 0.03125.
    CRPS = 2*(0.125+0.03125+0.03125) = 0.375.
    """
    y = np.array([0.0])
    preds = np.array([[-1.0, 1.0]])
    got = crps_from_quantiles(y, preds, alphas=[0.25, 0.75])
    assert got == pytest.approx(0.375, abs=1e-12)


def test_crps_from_quantiles_empty_is_nan_and_nonincreasing_alphas_raise():
    assert np.isnan(crps_from_quantiles(np.array([]), np.empty((0, 2)), alphas=[0.25, 0.75]))
    with pytest.raises(ValueError, match="strictly increasing"):
        crps_from_quantiles(
            np.array([1.0, 2.0]), np.array([[0.0, 1.0], [0.0, 1.0]]), alphas=[0.75, 0.25]
        )


# ----------------------------------------------------------------------------
# pit_values
# ----------------------------------------------------------------------------


def test_pit_values_interpolation_and_clamping():
    # alphas=[0.2,0.8], quantile curve q(0.2)=0, q(0.8)=10 for every row.
    # y=-1 clamps to alpha[0]=0.2; y=5 interpolates to 0.5; y=11 clamps to alpha[-1]=0.8.
    alphas = [0.2, 0.8]
    preds = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    y = np.array([-1.0, 5.0, 11.0])
    out = pit_values(y, preds, alphas)
    np.testing.assert_allclose(out, [0.2, 0.5, 0.8], atol=1e-12)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_pit_values_k_much_greater_than_n_runs_and_stays_in_unit_interval():
    rng = np.random.default_rng(0)
    n, k = 2, 20
    alphas = np.linspace(0.02, 0.98, k)
    # Monotone (non-crossing) rows so no warning; K >> N stresses the per-row insertion sort.
    preds = np.sort(rng.normal(size=(n, k)), axis=1)
    y = rng.normal(size=n)
    out = pit_values(y, preds, alphas)
    assert out.shape == (n,)
    assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_pit_values_warns_on_quantile_crossing(caplog):
    # Ascending alphas -> predicted quantiles should be non-decreasing across columns.
    # Row 0 decreases (crossing); row 1 is monotone. Exactly 1/2 rows should be flagged.
    alphas = [0.1, 0.5, 0.9]
    preds = np.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])
    y = np.array([2.0, 2.0])
    with caplog.at_level(logging.WARNING, logger="mlframe.metrics.quantile"):
        pit_values(y, preds, alphas)
    assert "crossing" in caplog.text
    assert "1/2" in caplog.text
