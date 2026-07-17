"""Wave 9.1 loop-iter-23 regression: DCD ``distance='vi'`` must be a
real Variation-of-Information formula, not a silent SU alias.

Pre-fix at ``_dynamic_cluster_discovery.py:275-286``: the ``vi`` branch
called ``symmetric_uncertainty`` and returned its result unchanged. The
in-source comment confessed "Here we approximate via SU and transform:
VI_proxy = 1 - SU, distance = SU still." -- so the documented opt-in to
VI distance silently became SU. Live confirmation: two DCDState objects
differing only in ``.distance=('su' vs 'vi')`` produced BIT-IDENTICAL
pair_su outputs (both 0.007656 etc).

Effect: ``MRMR(dcd_enable=True, dcd_distance='vi')`` got SU clustering
semantics regardless. Affects every user who chose VI for clustering -
the cluster_anchors / cluster pruning behaved as if SU was chosen.

Fix: implement the documented formula::
  VI(X, Y) = H(X) + H(Y) - 2 I(X; Y)
  score = 1 - VI / log(K_x * K_y)     [Meila 2007 normalization]

The score stays in [0, 1] with "1 = identical, 0 = independent" so
the existing ``score > tau_cluster`` membership rule is semantically
consistent across distance choices.
"""

from __future__ import annotations

import numpy as np
import pytest


def _state_for(distance: str, fd, fn):
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        make_dcd_state,
    )

    return make_dcd_state(
        X_raw=None,
        factors_data=fd,
        factors_nbins=fn,
        cols=[f"c{i}" for i in range(fd.shape[1])],
        nbins=fn,
        target_indices=None,
        distance=distance,
    )


def test_vi_differs_from_su_on_noise_pair():
    """On a randomly-correlated pair, SU and VI must produce different
    scores (proves VI is not a silent SU alias).
    """
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import pair_su

    rng = np.random.default_rng(0)
    n = 1000
    fd = rng.integers(0, 4, (n, 2)).astype(np.int32)
    fn = np.array([4, 4], dtype=np.int64)
    score_su = pair_su(_state_for("su", fd, fn), 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    score_vi = pair_su(_state_for("vi", fd, fn), 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    assert score_su != score_vi, f"SU and VI gave identical scores ({score_su}) - VI branch is still aliasing SU."


def test_vi_identical_columns_returns_one():
    """X = Y -> VI = 0 -> score = 1."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import pair_su

    rng = np.random.default_rng(1)
    n = 800
    col = rng.integers(0, 4, n).astype(np.int32)
    fd = np.column_stack([col, col])
    fn = np.array([4, 4], dtype=np.int64)
    score = pair_su(_state_for("vi", fd, fn), 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    assert score > 0.99


def test_vi_independent_columns_score_near_zero():
    """X independent of Y -> VI near max -> score near 0."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import pair_su

    rng = np.random.default_rng(2)
    n = 2000
    fd = np.column_stack(
        [
            rng.integers(0, 4, n).astype(np.int32),
            rng.integers(0, 4, n).astype(np.int32),
        ]
    )
    fn = np.array([4, 4], dtype=np.int64)
    score = pair_su(_state_for("vi", fd, fn), 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    assert score < 0.05


def test_vi_score_in_unit_interval():
    """Output of the iter-23 VI similarity must stay in [0, 1] for
    arbitrary integer inputs.
    """
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import pair_su

    rng = np.random.default_rng(3)
    n = 500
    fd = rng.integers(0, 5, (n, 4)).astype(np.int32)
    fn = np.array([5, 5, 5, 5], dtype=np.int64)
    state = _state_for("vi", fd, fn)
    for a in range(4):
        for b in range(a + 1, 4):
            score = pair_su(state, a, b, entropy_cache=None, factors_data=fd, factors_nbins=fn)
            assert 0.0 <= score <= 1.0, f"VI score out of [0,1]: pair=({a},{b}), score={score}"
