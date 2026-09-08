"""Regression tests for how a tied ranking is cut at K.

This is the bug that produced the benchmark's most striking result and its most embarrassing one. RFECV
assigns rank 1 to every survivor, so an arm that keeps all columns has a completely tied ranking. The cut
used a stable sort, a stable sort preserves the input order, and the generated beds declare their
informative columns first -- so cutting a tied ranking at K returned exactly the answer key, and the arm
scored a perfect recovery it never earned, on every seed, which then read as perfect stability too.

The fix has two halves and both are tested here: ties are permuted with a per-cell seed, so a tied arm
scores at chance rather than at column order, and the ranking reports the size of its largest tie group so a
consumer can tell a ranking from a coin flip.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from mlframe.feature_selection._benchmarks.fs_hybrid._matched_k import cut_at_k, ranking_from_arm_result

NAMES: List[str] = ["p0", "p1", "p2", "n0", "n1", "n2", "n3", "n4"]
ANSWER = {"p0", "p1", "p2"}


class _Result:
    """Minimal stand-in for an ArmResult with a chosen score vector."""

    def __init__(self, score: np.ndarray, kind: str = "ordinal") -> None:
        """Store the score and declare the kind the ranking layer should read it as."""
        self.support = np.ones(score.shape[0], dtype=bool)
        self.score = score
        self.score_kind = kind


class TestFullyTiedRanking:
    """An arm that expresses no preference must not inherit one from the spec."""

    def test_the_cut_is_not_the_declaration_order(self) -> None:
        """The exact failure: a stable sort handed back the first K columns, which are the answer key."""
        seeds_hitting_the_answer = 0
        for seed in range(30):
            ranking = ranking_from_arm_result(_Result(np.zeros(len(NAMES))), NAMES, tie_break_seed=seed)
            if set(cut_at_k(ranking, 3) or []) == ANSWER:
                seeds_hitting_the_answer += 1
        # Chance of drawing the three answer columns from eight is 1/56; thirty seeds landing on it every
        # time is what the bug looked like, and a handful of hits is what chance looks like.
        assert seeds_hitting_the_answer <= 3, f"{seeds_hitting_the_answer}/30 seeds recovered the answer key from a tie"

    def test_different_seeds_give_different_cuts(self) -> None:
        """A tied ranking that is identical across seeds is a constant, and a constant is not a selection."""
        cuts = {tuple(cut_at_k(ranking_from_arm_result(_Result(np.zeros(len(NAMES))), NAMES, tie_break_seed=seed), 3) or ()) for seed in range(10)}
        assert len(cuts) > 1

    def test_the_same_seed_reproduces_the_cut(self) -> None:
        """Randomised does not mean irreproducible: a resumed cell must land where the first run did."""
        first = cut_at_k(ranking_from_arm_result(_Result(np.zeros(len(NAMES))), NAMES, tie_break_seed=7), 3)
        second = cut_at_k(ranking_from_arm_result(_Result(np.zeros(len(NAMES))), NAMES, tie_break_seed=7), 3)
        assert first == second

    def test_the_tie_is_reported(self) -> None:
        """A consumer must be able to tell a ranking from a coin flip without re-deriving the scores."""
        ranking = ranking_from_arm_result(_Result(np.zeros(len(NAMES))), NAMES, tie_break_seed=0)
        assert ranking.largest_tie_group == len(NAMES)


class TestGenuineRankingIsUntouched:
    """The fix must not randomise anything an arm actually ordered."""

    def test_a_strict_ranking_is_returned_in_score_order(self) -> None:
        """Distinct scores leave nothing to break, so the order is the arm's and only the arm's."""
        scores = np.array([9.0, 8.0, 7.0, 1.0, 0.5, 0.4, 0.3, 0.2])
        for seed in (0, 1, 2):
            ranking = ranking_from_arm_result(_Result(scores, kind="continuous"), NAMES, tie_break_seed=seed)
            assert list(ranking.order[:3]) == ["p0", "p1", "p2"]
            assert ranking.largest_tie_group == 1

    def test_only_the_tied_block_is_permuted(self) -> None:
        """A partial tie must not disturb the columns the arm did separate."""
        scores = np.array([9.0, 8.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        for seed in (0, 1, 2):
            ranking = ranking_from_arm_result(_Result(scores, kind="continuous"), NAMES, tie_break_seed=seed)
            assert list(ranking.order[:2]) == ["p0", "p1"]
            assert ranking.largest_tie_group == 6

    def test_a_non_finite_score_sorts_last_rather_than_crashing(self) -> None:
        """An arm that could not score a column has not ranked it first."""
        scores = np.array([np.nan, 8.0, 7.0, 1.0, 0.5, 0.4, 0.3, 0.2])
        ranking = ranking_from_arm_result(_Result(scores, kind="continuous"), NAMES, tie_break_seed=0)
        assert ranking.order[-1] == "p0"


def test_a_selection_order_arm_keeps_its_own_prefix() -> None:
    """Tie-breaking is a scored-ranking concern; an arm that reports an order is not touched by it."""

    class _Ordered:
        """An arm reporting the order it selected in."""

        support = np.array([True, True, True, False, False, False, False, False])
        score = None
        score_kind = "selection_order"
        ranked_prefix = (2, 0, 1)

    ranking: Any = ranking_from_arm_result(_Ordered(), NAMES, tie_break_seed=3)
    assert list(ranking.order) == ["p2", "p0", "p1"]
