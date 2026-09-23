"""The base ranking must not let the redundancy term win on scale alone.

``mrmr_rank_bases`` scores ``relevance - beta * mean redundancy`` with ``beta`` defaulting to 1. The two are different mutual-information quantities:
relevance is feature-to-target, redundancy is feature-to-feature, and the second is routinely far larger, especially against a low-cardinality
target. Subtracting them raw meant the order was decided almost entirely by diversity, which is the opposite of trading a little relevance for
diversity. Both terms are rescaled to the pool's own maximum before they are combined.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._mrmr_base_rank import mrmr_rank_bases


def test_the_most_relevant_candidate_survives_a_scale_mismatch():
    """Relevance in [0, 0.05] against redundancy in [0, 2.0]: the strongest non-duplicate must still rank near the top."""
    names = ["strong", "mid", "weak", "dup"]
    relevance = [0.050, 0.030, 0.010, 0.049]  # binary-target scale
    # Feature-to-feature MI between continuous columns, a scale up: "dup" is near-identical to "strong".
    red = np.array(
        [
            [0.0, 0.40, 0.30, 1.95],
            [0.40, 0.0, 0.35, 0.45],
            [0.30, 0.35, 0.0, 0.32],
            [1.95, 0.45, 0.32, 0.0],
        ]
    )
    ranked = mrmr_rank_bases(names, relevance, red, k=4)
    assert ranked[0] == "strong", f"the first pick is pure relevance and must be the strongest: {ranked}"
    assert "strong" in ranked[:2] and ranked.index("strong") < ranked.index("dup"), f"the duplicate outranked the original: {ranked}"


def test_beta_zero_is_still_pure_relevance_order():
    """With no diversity weight the ranking is relevance alone, scaling or not."""
    names = ["a", "b", "c"]
    red = np.array([[0.0, 5.0, 5.0], [5.0, 0.0, 5.0], [5.0, 5.0, 0.0]])
    assert mrmr_rank_bases(names, [0.1, 0.3, 0.2], red, k=3, beta=0.0) == ["b", "c", "a"]


def test_diversity_still_breaks_a_relevance_tie():
    """Rescaling must not disable the redundancy term: among equally relevant candidates the least redundant wins."""
    names = ["a", "b", "c"]
    red = np.array([[0.0, 0.9, 0.1], [0.9, 0.0, 0.5], [0.1, 0.5, 0.0]])
    ranked = mrmr_rank_bases(names, [0.5, 0.5, 0.5], red, k=3)
    assert ranked[0] == "a", ranked
    assert ranked[1] == "c", f"after picking a, the least redundant with it should come next: {ranked}"


def test_an_all_zero_redundancy_pool_does_not_divide_by_zero():
    """Perfectly independent candidates give a zero redundancy scale, which must not produce NaN ordering."""
    names = ["a", "b", "c"]
    red = np.zeros((3, 3))
    assert mrmr_rank_bases(names, [0.1, 0.9, 0.5], red, k=3) == ["b", "c", "a"]


def test_an_all_zero_relevance_pool_is_ordered_by_diversity_alone():
    """No relevance to trade means the ranking falls back to diversity without dividing by zero."""
    names = ["a", "b", "c"]
    red = np.array([[0.0, 0.8, 0.2], [0.8, 0.0, 0.4], [0.2, 0.4, 0.0]])
    ranked = mrmr_rank_bases(names, [0.0, 0.0, 0.0], red, k=3)
    assert set(ranked) == set(names) and len(ranked) == 3


@pytest.mark.parametrize("k", [0, 1, 2, 5])
def test_k_is_respected(k):
    """The pool size still bounds the result, and a non-positive k returns nothing."""
    names = ["a", "b", "c"]
    red = np.zeros((3, 3))
    assert len(mrmr_rank_bases(names, [0.3, 0.2, 0.1], red, k=k)) == max(0, min(k, 3))
