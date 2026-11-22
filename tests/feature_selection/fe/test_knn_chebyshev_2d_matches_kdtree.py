"""``knn_chebyshev_2d`` returns what ``sklearn.neighbors.KDTree(metric='chebyshev').query`` returns for points without ties, and the KSG estimators that use it keep their values."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._ksg_cc_fast import knn_chebyshev_2d


@pytest.mark.parametrize("n,k", [(60, 3), (500, 5), (4000, 1), (4000, 8)])
def test_distances_and_indices_equal_kdtree(n, k):
    """Same ascending distances and the same neighbour indices, the row itself first."""
    from sklearn.neighbors import KDTree

    rng = np.random.default_rng(n + k)
    P = rng.standard_normal((n, 2))
    P[:, 1] += 0.6 * P[:, 0]
    d_ref, i_ref = KDTree(P, metric="chebyshev").query(P, k=k + 1)
    d, i = knn_chebyshev_2d(P, k)
    np.testing.assert_array_equal(d, d_ref)
    np.testing.assert_array_equal(i, i_ref)


def test_fewer_points_than_neighbours_returns_every_point():
    """With n <= k the answer has n columns."""
    P = np.array([[0.0, 0.0], [1.0, 0.5], [3.0, 1.0]])
    d, i = knn_chebyshev_2d(P, 5)
    assert d.shape == (3, 3) and sorted(i[0].tolist()) == [0, 1, 2]


def test_ksg_estimators_run_on_the_kernel():
    """The mixed KSG and the LNC estimators give a positive MI for a dependent pair and about zero for an independent one."""
    from mlframe.feature_selection.filters._ksg import ksg_lnc_mi, mixed_ksg_mi

    rng = np.random.default_rng(1)
    x = rng.standard_normal(1500)
    dep = x + 0.3 * rng.standard_normal(1500)
    ind = rng.standard_normal(1500)
    for fn in (mixed_ksg_mi, ksg_lnc_mi):
        assert fn(x, dep, k=5) > 0.3
        assert fn(x, ind, k=5) < 0.1
