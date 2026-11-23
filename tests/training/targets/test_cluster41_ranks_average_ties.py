"""Cluster 41 regression: tied values share their average rank, as in scipy (audit 2026-11-20, finding 41-F12)."""

import numpy as np
from scipy.stats import rankdata

from mlframe.training.targets.regression_residual_audit import _ranks_via_scatter


def test_ties_get_average_rank_matching_scipy():
    """Equal predictions share the mean of their positions, not a positional order (41-F12)."""
    x = np.array([3.0, 1.0, 3.0, 2.0, 3.0])
    np.testing.assert_allclose(_ranks_via_scatter(x), rankdata(x) - 1.0)


def test_tie_ranks_do_not_depend_on_row_order():
    """Permuting rows that tie must not change the rank assigned to each value (41-F12)."""
    x = np.array([5.0, 5.0, 1.0, 5.0])
    ranks_a = _ranks_via_scatter(x)
    ranks_b = _ranks_via_scatter(x[::-1])
    np.testing.assert_allclose(ranks_a[::-1], ranks_b)


def test_no_ties_is_plain_ordinal_ranking():
    """Control: distinct values keep ordinal 0-based ranks (41-F12)."""
    x = np.array([0.3, 0.1, 0.2])
    np.testing.assert_allclose(_ranks_via_scatter(x), np.array([2.0, 0.0, 1.0]))
