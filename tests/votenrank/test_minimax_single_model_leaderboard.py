"""VOTENRANK-7 regression test: minimax_election on a 1-model leaderboard must return that model.

Pre-fix, minimax_ranking's ``models_scores.drop(model).max()`` on a 1-model leaderboard drops the model's
own score, leaving an empty Series whose ``.max()`` is NaN. ``ranking2top`` (``ranking == ranking.max()``)
never matches NaN, so ``minimax_election`` silently returned an empty winner list instead of the one
trivially-correct model.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard


@pytest.mark.parametrize("score_type", ["winning_votes", "margins", "pairwise_opposition"])
def test_minimax_election_single_model_leaderboard_returns_that_model(score_type):
    """A leaderboard with exactly one model must elect that model under every score_type, not an empty list."""
    tbl = pd.DataFrame({"t0": [1.0], "t1": [2.0]}, index=["only_model"])
    lb = Leaderboard(table=tbl)

    ranking = lb.minimax_ranking(score_type=score_type)
    assert not ranking.isna().any(), f"minimax_ranking produced NaN for a 1-model leaderboard: {ranking}"

    winners = lb.minimax_election(score_type=score_type)
    assert winners == ["only_model"], f"expected minimax_election to elect the sole model, got {winners}"


def test_minimax_ranking_multi_model_unaffected_by_the_single_model_guard():
    """Sanity: the new empty-opponents guard must not change results on a real multi-model leaderboard."""
    rng = np.random.default_rng(7)
    tbl = pd.DataFrame(rng.normal(size=(5, 4)), index=[f"m{i}" for i in range(5)], columns=[f"t{j}" for j in range(4)])
    lb = Leaderboard(table=tbl)
    ranking = lb.minimax_ranking(score_type="winning_votes")
    assert not ranking.isna().any()
    assert len(ranking) == 5
