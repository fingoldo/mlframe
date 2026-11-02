"""A constant column carries no information about a time-ordered split and must not be flagged by the temporal leak scan."""

from __future__ import annotations

import numpy as np

from mlframe.evaluation.leak_scan import _rank_columns, _spearman_against


def test_constant_column_has_zero_correlation_with_a_row_ordered_split():
    """Train rows first, test rows after: a constant column correlates 0 (it was 0.866 with row-order tie breaking)."""
    n = 1000
    x = np.ones((n, 1))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    assert abs(float(_spearman_against(x, y)[0])) < 1e-12


def test_tied_values_get_the_average_rank():
    """Three equal values share the mean rank, so row order cannot leak in."""
    ranks = _rank_columns(np.array([[5.0], [5.0], [5.0], [1.0]]))
    np.testing.assert_allclose(ranks[:, 0], [2.0, 2.0, 2.0, 0.0])
