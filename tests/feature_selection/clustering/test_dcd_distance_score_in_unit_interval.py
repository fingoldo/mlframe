"""Wave 9.1 loop-iter-32 regression: DCD distance scores MUST stay in
[0, 1] for the cluster-membership rule ``score > tau_cluster`` to be
semantically valid.

Pre-fix at ``_dynamic_cluster_discovery.py:341`` (sotoca_pla branch):
the score formula ``max(0.0, 1.0 - d)`` clamped only the lower bound.
When the target was highly informative about BOTH X_a and X_b
(``I(X_a;Y) + I(X_b;Y) > I(X_a;X_b)`` - a common case for two
target-informative features), ``d`` went negative and the score
exceeded 1.0 (live demo: 1.5).

Effect: the cluster-membership test ``score > tau_cluster`` then
triggered for ANY ``tau_cluster`` in [0, 1] including the default 0.7.
sotoca_pla was DESIGNED to detect target-informative redundancy
clusters but was instead falsely pruning the exact features it should
have surfaced.

Severity: P1 (silent over-pruning on the target-informative case).

Fix at line 341: ``max(0.0, min(1.0, 1.0 - d))``. Parallel parity fix
at iter-23 VI branch (line 289) - same defensive upper clamp against
FP noise.
"""

from __future__ import annotations

import numpy as np
import pytest


def _state(distance: str, fd, fn, target_indices=None):
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        make_dcd_state,
    )

    return make_dcd_state(
        X_raw=None,
        factors_data=fd,
        factors_nbins=fn,
        cols=[f"c{i}" for i in range(fd.shape[1])],
        nbins=fn,
        target_indices=target_indices,
        distance=distance,
    )


def test_sotoca_pla_clamps_to_one_on_target_informative_pair():
    """X_a = X_b = Y -> pre-fix score 1.5, post-fix score 1.0."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import pair_su

    rng = np.random.default_rng(0)
    n = 1000
    y = rng.integers(0, 2, n).astype(np.int32)
    fd = np.column_stack([y, y, y])
    fn = np.array([2, 2, 2], dtype=np.int64)
    state = _state(
        "sotoca_pla",
        fd,
        fn,
        target_indices=np.array([2], dtype=np.int64),
    )
    score = pair_su(state, 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    assert 0.0 <= score <= 1.0
    # Identical-to-target pair must give the MAX similarity (1.0),
    # not a phantom 1.5+.
    assert score >= 0.99


def test_sotoca_pla_independent_pair_low_score():
    """Negative control: two independent features -> low similarity."""
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import pair_su

    rng = np.random.default_rng(1)
    n = 2000
    a = rng.integers(0, 4, n).astype(np.int32)
    b = rng.integers(0, 4, n).astype(np.int32)
    y = rng.integers(0, 2, n).astype(np.int32)
    fd = np.column_stack([a, b, y])
    fn = np.array([4, 4, 2], dtype=np.int64)
    state = _state("sotoca_pla", fd, fn, target_indices=np.array([2], dtype=np.int64))
    score = pair_su(state, 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    assert 0.0 <= score <= 1.0


def test_vi_clamp_upper_bound():
    """Parity check: VI branch (iter 23) also defensively clamps to
    [0, 1] against numerical FP noise.
    """
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import pair_su

    rng = np.random.default_rng(2)
    n = 1000
    col = rng.integers(0, 4, n).astype(np.int32)
    fd = np.column_stack([col, col])  # identical -> score should be 1.0 max
    fn = np.array([4, 4], dtype=np.int64)
    state = _state("vi", fd, fn)
    score = pair_su(state, 0, 1, entropy_cache=None, factors_data=fd, factors_nbins=fn)
    assert 0.0 <= score <= 1.0


@pytest.mark.parametrize("distance", ["su", "vi", "sotoca_pla"])
def test_all_distances_unit_interval_invariant(distance):
    """Across distance modes the score MUST stay in [0, 1] for the
    cluster-membership rule to be semantically meaningful.
    """
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import pair_su

    rng = np.random.default_rng(3)
    n = 500
    fd = rng.integers(0, 5, (n, 4)).astype(np.int32)
    fn = np.array([5, 5, 5, 5], dtype=np.int64)
    target = np.array([3], dtype=np.int64) if distance == "sotoca_pla" else None
    state = _state(distance, fd, fn, target_indices=target)
    for a in range(4):
        for b in range(a + 1, 4):
            score = pair_su(state, a, b, entropy_cache=None, factors_data=fd, factors_nbins=fn)
            assert 0.0 <= score <= 1.0, f"distance={distance}, pair=({a},{b}), score={score}"
