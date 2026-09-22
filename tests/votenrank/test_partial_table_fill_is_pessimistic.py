"""A model that did not run on a task must not be credited with a median result there.

`mean_ranking` / `optimality_gap_ranking` filled missing cells with the task median, so a model that crashed or timed
out on the two hardest tasks beat a model that ran everywhere and scored below median on them - and nothing in the
result said which cells were imputed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.votenrank import Leaderboard


def _table():
    # Higher is better. B failed on the two hard tasks; A ran everywhere but scored below median on them.
    return pd.DataFrame(
        # A and B are identical on the tasks both ran, so the fill rule alone decides their order.
        {"easy1": [0.90, 0.90, 0.92], "easy2": [0.90, 0.90, 0.90], "hard1": [0.40, np.nan, 0.70], "hard2": [0.40, np.nan, 0.70]},
        index=["A", "B", "C"],
    )


def test_a_failed_model_does_not_outrank_one_that_ran():
    """Filled with the task's worst score, B can at best tie A - it cannot be credited above a model that ran."""
    ranking = Leaderboard(table=_table()).mean_ranking()
    assert ranking["B"] <= ranking["A"] + 1e-12


def test_the_imputed_cells_are_reported():
    lb = Leaderboard(table=_table())
    assert sorted(lb.imputed_cells_) == [("B", "hard1"), ("B", "hard2")]


def test_the_historical_median_fill_is_still_available():
    ranking = Leaderboard(table=_table(), partial_fill="median").mean_ranking()
    assert list(ranking.index).index("B") < list(ranking.index).index("A"), "median credit is what put B ahead"


def test_an_unknown_fill_policy_is_refused():
    with pytest.raises(ValueError, match="partial_fill"):
        Leaderboard(table=_table(), partial_fill="mean")
