"""Regression: minimax_ranking winning_votes redundant-pass removal stays identical.

Pins that dropping the duplicated ((ranks < ranks.loc[model]) * weights).sum(axis=1)
computation in minimax_ranking does not change the output. The reference recomputes
the score with the explicit duplicate, mirroring the pre-optimization code.
"""

import numpy as np
import pandas as pd

from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard


def _reference_winning_votes(lb):
    out = []
    for model in lb.models:
        models_scores = ((lb.ranks < lb.ranks.loc[model]) * lb.weights).sum(axis=1)
        does_win = ((lb.ranks < lb.ranks.loc[model]) * lb.weights).sum(axis=1) > ((lb.ranks > lb.ranks.loc[model]) * lb.weights).sum(axis=1)
        models_scores = models_scores * does_win
        out.append(models_scores.drop(model).max())
    return (-pd.Series(data=out, index=pd.Series(lb.models, name="Name"))).sort_values(ascending=False)


def test_minimax_winning_votes_identical_to_duplicated_reference():
    rng = np.random.default_rng(42)
    tbl = pd.DataFrame(
        rng.normal(size=(30, 12)),
        index=[f"m{i}" for i in range(30)],
        columns=[f"t{j}" for j in range(12)],
    )
    lb = Leaderboard(table=tbl)

    got = lb.minimax_ranking(score_type="winning_votes")
    expected = _reference_winning_votes(lb)

    assert got.index.tolist() == expected.index.tolist()
    np.testing.assert_array_equal(got.to_numpy(), expected.to_numpy())
