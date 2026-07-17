"""Wave 9.1 loop-iter-44 regression: ``discover_cluster_members`` MUST
refuse to anchor on a swap-pruned column.

Pre-fix at ``_dynamic_cluster_discovery.py:433``::

    anchors = state.cluster_anchors.setdefault(anchor, set())

``setdefault`` resurrected a dead (swap-pruned) anchor as an empty
entry. The discover loop then proceeded to add arbitrary candidates as
its "members" - inflating ``MRMR.dcd_["n_anchors"]`` and silently
corrupting the published anchor->member graph.

Live demo (direct API user, not the live MRMR fit path which only
passes freshly-selected vars):

  state.pool_pruned_mask[0] = True   # anchor 0 was swap-pruned earlier
  discover_cluster_members(state, just_selected=0, candidate_pool=[1,2,3])
  pre-fix: state.cluster_anchors gets key 0 with {1,2,3} as members
  post-fix: cluster_anchors stays empty, function returns empty set

Severity: medium. ``_screen_predictors._dcd_discover`` only passes
freshly-selected ``var`` indices in the live MRMR fit path so this was
latent. But ``discover_cluster_members`` is in ``__all__`` and the
docstring promises set-default semantics; direct API consumers
(notebook audit code, downstream re-screening) tripped it.

Fix at line 432: refuse to anchor on a column with
``pool_pruned_mask[anchor] == True`` - return empty set immediately.
"""

from __future__ import annotations

import numpy as np


def _state(n_cols=4):
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        make_dcd_state,
    )

    rng = np.random.default_rng(0)
    fd = rng.integers(0, 4, (200, n_cols)).astype(np.int32)
    fn = np.array([4] * n_cols, dtype=np.int64)
    return (
        make_dcd_state(
            X_raw=None,
            factors_data=fd,
            factors_nbins=fn,
            cols=[f"c{i}" for i in range(n_cols)],
            nbins=fn,
            target_indices=None,
            distance="su",
        ),
        fd,
        fn,
    )


def test_pruned_anchor_returns_empty_set_no_resurrection():
    """Anchor on a column with ``pool_pruned_mask=True`` must yield no
    members AND not resurrect the key in ``cluster_anchors``.
    """
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        discover_cluster_members,
    )

    state, fd, fn = _state()
    state.pool_pruned_mask[0] = True
    result = discover_cluster_members(
        state,
        just_selected=0,
        candidate_pool=[1, 2, 3],
        entropy_cache=None,
        factors_data=fd,
        factors_nbins=fn,
    )
    assert result == set()
    assert 0 not in state.cluster_anchors


def test_non_pruned_anchor_works_normally():
    """Negative control: a fresh anchor on a non-pruned column behaves
    as documented (may add some members based on SU).
    """
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        discover_cluster_members,
    )

    state, fd, fn = _state()
    # tau_cluster default may not accept on random data; just verify
    # the function completes and creates the anchor key.
    result = discover_cluster_members(
        state,
        just_selected=0,
        candidate_pool=[1, 2, 3],
        entropy_cache=None,
        factors_data=fd,
        factors_nbins=fn,
    )
    # The key 0 must exist (even if empty set) so the in-loop code
    # path of MRMR.fit can append to it later.
    assert 0 in state.cluster_anchors
    assert isinstance(result, set)


def test_multiple_pruned_anchors_all_refused():
    """Several pruned anchors in sequence must all be refused without
    polluting cluster_anchors.
    """
    from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
        discover_cluster_members,
    )

    state, fd, fn = _state()
    state.pool_pruned_mask[0] = True
    state.pool_pruned_mask[2] = True
    for pruned_anchor in (0, 2):
        result = discover_cluster_members(
            state,
            just_selected=pruned_anchor,
            candidate_pool=[1, 3],
            entropy_cache=None,
            factors_data=fd,
            factors_nbins=fn,
        )
        assert result == set()
        assert pruned_anchor not in state.cluster_anchors
