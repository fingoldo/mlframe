"""Wave 9.1 loop-iter-6 regression: ``_edges_from_quantiles`` MUST
de-duplicate inner edges.

Pre-fix: ``np.nanpercentile`` on a constant column emits ``n_bins + 1``
identical values; on a heavily skewed / sparse column (e.g. 99% zeros)
it emits a run of identical inner edges. ``np.searchsorted`` then sees
duplicates and ``K_x = len(edges) + 1`` becomes phantom-inflated.

Effect downstream:
- ``_plug_in_mi_njit`` (`_adaptive_nbins.py:362`) over-corrects Miller-Madow
  by ``(K_x_phantom - 1) * (K_y - 1) / (2N)`` on every constant /
  near-constant column. Bias becomes negative instead of zero.
- SU normaliser denominator ``log(K_x)`` is inflated, shrinking the SU
  ranking of sparse features.
- Joint-histogram allocation over-allocates ``K_x * K_y`` cells
  (in extreme cases forces a wider dtype).

``edges_qs`` already de-duplicates via ``np.unique`` at line 170; the fix
brings ``_edges_from_quantiles`` in line so the FD / Sturges /
OptimalJoint paths inherit it (they all funnel through this helper).
"""

from __future__ import annotations

import numpy as np
import pytest


from mlframe.feature_selection.filters._adaptive_nbins import (
    _edges_from_quantiles,
    edges_freedman_diaconis,
    edges_qs,
    per_feature_edges,
)


def test_edges_from_quantiles_dedups_constant_column():
    """Pre-fix: returned 9 duplicate 1.0 edges. Post-fix: empty."""
    x_const = np.ones(500)
    edges = _edges_from_quantiles(x_const, n_bins=10)
    assert edges.size == 0, f"constant column must produce 0 inner edges (K_x=1); got {edges.size} edges with values {edges[:5]}..."


def test_edges_from_quantiles_dedups_sparse_column():
    """1% positive sparse column: 99% zeros + 1% ones."""
    x_sparse = np.zeros(1000)
    x_sparse[:10] = 1.0
    edges = _edges_from_quantiles(x_sparse, n_bins=10)
    # Pre-fix: emitted 10 duplicate zero edges. Post-fix: at most
    # 1 unique inner edge (zero or one quantile boundary), so size <= 1.
    assert edges.size <= 1, f"sparse 1% column must collapse duplicate-zero edges; got {edges.size} edges with values {edges[:5]}..."
    assert len(np.unique(edges)) == edges.size


def test_edges_from_quantiles_normal_continuous_unaffected():
    """Continuous data MUST keep all unique edges - the fix only collapses
    ties, not legitimate cuts.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)
    edges = _edges_from_quantiles(x, n_bins=20)
    # 19 inner edges for n_bins=20, all distinct under N(0,1) at n=2000
    assert edges.size == 19
    assert len(np.unique(edges)) == edges.size


def test_freedman_diaconis_constant_returns_one_bin():
    """Through the FD wrapper (calls ``_edges_from_quantiles`` internally)."""
    K = int(np.searchsorted(edges_freedman_diaconis(np.ones(500)), np.ones(500), side="right").max()) + 1
    assert K == 1


def test_qs_and_freedman_diaconis_agree_on_constant():
    """QS de-duplicates correctly; FD now matches under iter-6 fix."""
    x = np.ones(500)
    assert edges_qs(x).size == edges_freedman_diaconis(x).size == 0


def test_per_feature_edges_freedman_diaconis_constant_yields_kx1():
    """The bug fingerprint at the API boundary - ``per_feature_edges`` for
    a constant column under FD must give ``K_x = 1`` (single-bin), not
    the phantom ``K_x = 10`` the pre-fix code produced.
    """
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(500), rng.standard_normal(500)])
    edges = per_feature_edges(X, method="freedman_diaconis")
    binned0 = np.searchsorted(edges[0], X[:, 0], side="right")
    Kx0 = int(binned0.max()) + 1
    assert Kx0 == 1, (
        f"constant column must collapse to K_x=1 under freedman_diaconis; "
        f"got K_x={Kx0}. This breaks Miller-Madow correction "
        f"(over-correction by (K_x-1)*(K_y-1)/(2N)) and SU normalization "
        f"(log(K_x) denominator inflated)."
    )
