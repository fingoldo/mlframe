"""The block analytic null of the pair admission equals the per-pair ``_conditional_perm_null`` wherever the analytic branch applies."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_cmi_redundancy_gate import _conditional_perm_null
from mlframe.feature_selection.filters._mrmr_fe_step import _pair_admission_batch as A


@pytest.mark.parametrize("n,k_x,k_z", [(20000, 4, 3), (20000, 32, 16), (3000, 64, 32)])
def test_block_analytic_null_equals_per_pair(n, k_x, k_z):
    """Same floor and null mean per candidate; where the block declines, the per-pair call is the permutation one (not compared)."""
    rng = np.random.default_rng(n + k_x)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    z = rng.integers(0, k_z, size=n).astype(np.int64)
    xs = [rng.integers(0, k, size=n).astype(np.int64) for k in (2, k_x // 2 or 1, k_x)]
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import _entropy_from_classes, _renumber_joint

    cards = []
    for x in xs:
        cards.append((_renumber_joint(x, z)[1], _renumber_joint(x, y, z)[1]))
    k_zc = _entropy_from_classes(z)[1]
    k_yz = _renumber_joint(y, z)[1]
    floor, mean, applies = A._block_analytic_nulls(n, k_zc, [c[0] for c in cards], k_yz, [c[1] for c in cards], A._CMI_QUANTILE)
    for j, x in enumerate(xs):
        if not applies[j]:
            continue
        f1, m1 = _conditional_perm_null(x, y, z, seed=1, precomp_cards=(int(k_zc), int(cards[j][0]), int(k_yz), int(cards[j][1])))
        assert floor[j] == f1 and mean[j] == m1
