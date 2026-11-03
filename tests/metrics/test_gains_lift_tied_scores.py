"""Gains, lift and the gains table average tied scores, so row order cannot manufacture lift."""

from __future__ import annotations

import numpy as np

from mlframe.metrics.classification._gains_lift import cumulative_gains_curve, gains_table, lift_curve


def _constant_score_case():
    """10 rows, the 2 positives first, one constant score."""
    return np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), np.full(10, 0.5)


def test_lift_curve_of_a_constant_score_is_one():
    """The first point used to read 5.0 (both positives in the top 10%); an uninformative score has lift 1."""
    y, s = _constant_score_case()
    _frac, lift = lift_curve(y, s)
    np.testing.assert_allclose(lift[np.isfinite(lift)], 1.0)


def test_gains_curve_of_a_constant_score_is_the_diagonal():
    """Random targeting: gain equals the fraction targeted."""
    y, s = _constant_score_case()
    frac, gain = cumulative_gains_curve(y, s)
    np.testing.assert_allclose(gain, frac)


def test_row_order_does_not_change_the_gains_table_for_tied_scores():
    """Reversing the rows leaves the table's cumulative capture unchanged when scores tie."""
    rng = np.random.default_rng(0)
    s = np.round(rng.random(300), 1)
    y = (rng.random(300) < s).astype(int)
    a = gains_table(y, s, n_bins=10)
    b = gains_table(y[::-1], s[::-1], n_bins=10)
    for key in a:
        if isinstance(a[key], np.ndarray) and a[key].dtype.kind == "f":
            np.testing.assert_allclose(a[key], b[key])


def test_tie_free_scores_are_unchanged():
    """Distinct scores keep the historical curve (negative control)."""
    y = np.array([1, 0, 1, 0, 0])
    _frac, gain = cumulative_gains_curve(y, np.array([0.9, 0.8, 0.7, 0.2, 0.1]))
    np.testing.assert_allclose(gain, [0.0, 0.5, 0.5, 1.0, 1.0, 1.0])
