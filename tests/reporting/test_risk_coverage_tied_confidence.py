"""The risk-coverage curve averages loss inside blocks of equal confidence, so input row order cannot change AURC."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.risk_coverage import compute_risk_coverage


def _case():
    """Binary scores in {0.2, 0.8}: confidence |p - 0.5| * 2 ties for every row; 20% of the rows are wrong."""
    y = np.r_[np.zeros(40), np.ones(40), np.zeros(10), np.ones(10)]
    p = np.r_[np.full(40, 0.2), np.full(40, 0.8), np.full(10, 0.8), np.full(10, 0.2)]
    return y, p


def test_aurc_does_not_depend_on_row_order_for_tied_confidence():
    """Misclassified rows last versus first used to move AURC; with block-averaged loss both orders agree."""
    y, p = _case()
    a = compute_risk_coverage(y, p, task="binary")
    b = compute_risk_coverage(y[::-1], p[::-1], task="binary")
    assert abs(a[3] - b[3]) < 1e-12
