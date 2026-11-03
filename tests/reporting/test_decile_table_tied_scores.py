"""The decile gain / lift table treats a block of equal scores as indistinguishable rows, so row order cannot manufacture lift."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts._binary_decile_table import binary_decile_table


def test_constant_score_over_label_sorted_rows_has_lift_one_everywhere():
    """1000 rows, score 0.5 everywhere, the 100 positives first: decile 1 used to capture all of them (lift 10.0); an uninformative score has lift 1.0."""
    y = np.r_[np.ones(100), np.zeros(900)]
    out = binary_decile_table(y, np.full(1000, 0.5))
    np.testing.assert_allclose(out["lift"], 1.0)
    np.testing.assert_allclose(out["gain"], np.arange(1, 11) / 10.0)


def test_row_order_does_not_change_the_table_for_tied_scores():
    """Reversing the rows leaves every column unchanged when scores tie."""
    rng = np.random.default_rng(0)
    score = np.round(rng.random(500), 1)
    y = (rng.random(500) < score).astype(float)
    a = binary_decile_table(y, score)
    b = binary_decile_table(y[::-1], score[::-1])
    for key in ("gain", "lift", "cum_ks"):
        np.testing.assert_allclose(a[key], b[key])


def test_tie_free_scores_are_unchanged():
    """Distinct scores keep the historical integer result (negative control)."""
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float)
    out = binary_decile_table(y, np.linspace(1.0, 0.0, 10))
    assert out["positives"].tolist() == [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
