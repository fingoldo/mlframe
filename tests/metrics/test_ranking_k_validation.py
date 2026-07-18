"""Regression: ranking metrics must reject k<=0 with a CLEAR ValueError.

Pre-fix behaviour (the bug):
    - ``map_at_k(k=0)`` raised an opaque ``ZeroDivisionError`` from deep in
      the numba kernel (denom = min(k, n_rel) = 0).
    - ``ndcg_at_k(k=0)`` / ``ndcg_at_k(k=-1)`` SILENTLY returned NaN.
    - ``compute_ranking_summary(eval_at=(0,))`` SILENTLY returned NaN.
The ``_ranking_extras`` siblings (dcg_at_k / hit_at_k / precision_at_k /
expected_reciprocal_rank) already validate ``k>=1``; ranking.py did not.
A bad config must raise clearly, not silently misbehave or crash opaquely.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.ranking import ndcg_at_k, map_at_k, compute_ranking_summary

_YT = np.array([1.0, 1.0, 0.0, 0.0])
_YS = np.array([0.9, 0.8, 0.7, 0.6])
_G = np.zeros(4)


@pytest.mark.parametrize("bad_k", [0, -1, -10])
def test_ndcg_at_k_rejects_nonpositive_k(bad_k):
    """Ndcg at k rejects nonpositive k."""
    with pytest.raises(ValueError, match="k must be >= 1"):
        ndcg_at_k(_YT, _YS, _G, k=bad_k)


@pytest.mark.parametrize("bad_k", [0, -1, -10])
def test_map_at_k_rejects_nonpositive_k(bad_k):
    """Map at k rejects nonpositive k."""
    with pytest.raises(ValueError, match="k must be >= 1"):
        map_at_k(_YT, _YS, _G, k=bad_k)


@pytest.mark.parametrize("eval_at", [(0,), (0, 5), (5, -1), (1, 0, 10)])
def test_ranking_summary_rejects_nonpositive_k(eval_at):
    """Ranking summary rejects nonpositive k."""
    with pytest.raises(ValueError, match="must be >= 1"):
        compute_ranking_summary(_YT, _YS, _G, eval_at=eval_at)


def test_valid_k_still_works():
    # Sanity: a valid config is unchanged (do NOT reject currently-valid configs).
    """Valid k still works."""
    assert ndcg_at_k(_YT, _YS, _G, k=2) == pytest.approx(1.0)
    assert map_at_k(_YT, _YS, _G, k=2) == pytest.approx(1.0)
    s = compute_ranking_summary(_YT, _YS, _G, eval_at=(1, 5))
    assert s["ndcg@1"] == pytest.approx(1.0)
    assert s["map@5"] == pytest.approx(1.0)
