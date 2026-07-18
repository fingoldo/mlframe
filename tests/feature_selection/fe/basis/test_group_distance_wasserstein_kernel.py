"""Regression sensor for the linear-time Wasserstein-1 kernel in
``_group_distance_fe`` (Layer 95 PART B perf optimization).

``_wasserstein1`` originally called ``scipy.stats.wasserstein_distance`` per
(group, num_col), re-sorting the large global array on every group. The njit
``_wasserstein1_sorted_kernel`` exploits the pre-sorted global to compute scipy's
EXACT integral in O(nu+nv). These tests pin numerical equivalence to scipy
(reduction-order FP noise only) on continuous, discrete-tied and pathological
inputs, so a future "just rewrite the merge" cannot silently change the value.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _group_distance_fe as gd

scipy_stats = pytest.importorskip("scipy.stats")
wasserstein_distance = scipy_stats.wasserstein_distance


@pytest.mark.skipif(not gd._HAVE_NUMBA, reason="numba kernel not available")
@pytest.mark.parametrize("seed", range(6))
def test_wasserstein1_matches_scipy_continuous(seed):
    """Wasserstein1 matches scipy continuous."""
    rng = np.random.default_rng(seed)
    glob = np.sort(rng.normal(size=rng.integers(200, 3000)))
    for _ in range(40):
        group = rng.normal(size=int(rng.integers(8, 300)))
        got = gd._wasserstein1(group, glob)
        ref = wasserstein_distance(group, glob)
        assert abs(got - ref) <= 1e-9 + 1e-9 * abs(ref), (got, ref)


@pytest.mark.skipif(not gd._HAVE_NUMBA, reason="numba kernel not available")
@pytest.mark.parametrize("seed", range(4))
def test_wasserstein1_matches_scipy_discrete_ties(seed):
    """Wasserstein1 matches scipy discrete ties."""
    rng = np.random.default_rng(100 + seed)
    glob = np.sort(rng.integers(0, 6, size=rng.integers(200, 3000)).astype(np.float64))
    for _ in range(40):
        group = rng.integers(0, 6, size=int(rng.integers(8, 300))).astype(np.float64)
        got = gd._wasserstein1(group, glob)
        ref = wasserstein_distance(group, glob)
        assert abs(got - ref) <= 1e-9 + 1e-9 * abs(ref), (got, ref)


@pytest.mark.skipif(not gd._HAVE_NUMBA, reason="numba kernel not available")
def test_wasserstein1_edge_cases():
    """Wasserstein1 edge cases."""
    glob = np.sort(np.arange(50, dtype=np.float64))
    # identical distributions -> 0
    assert gd._wasserstein1(np.arange(50, dtype=np.float64), glob) == pytest.approx(0.0, abs=1e-12)
    # single-value group
    g = np.full(10, 7.0)
    assert gd._wasserstein1(g, glob) == pytest.approx(wasserstein_distance(g, glob), abs=1e-9)
    # empty inputs short-circuit to 0
    assert gd._wasserstein1(np.array([]), glob) == 0.0
    assert gd._wasserstein1(g, np.array([])) == 0.0
