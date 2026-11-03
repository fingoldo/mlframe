"""elect_all honours the caller's parameter grid, and the rank splice keeps main-model order among tied specialist scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard
from mlframe.votenrank.rank_splice import segment_rank_splice


def test_elect_all_runs_only_the_requested_parameter_grid():
    """use_methods={'minimax': {'score_type': ['margins']}} must produce rows for 'margins' only, not for every default score_type."""
    rng = np.random.default_rng(0)
    table = pd.DataFrame(rng.random((6, 5)), index=[f"m{i}" for i in range(6)], columns=[f"t{j}" for j in range(5)])
    lb = Leaderboard(table)
    res = lb.elect_all(use_methods={"minimax": {"score_type": ["margins"]}})
    labels = [str(x) for x in res.iloc[:, 0]]
    assert labels and all("margins" in label for label in labels)


def test_constant_specialist_scores_keep_the_main_models_order():
    """A constant specialist score ranks nothing: the segment must come back sorted ascending in row order (stable ties), as the main order."""
    rng = np.random.default_rng(1)
    n = 600
    main = rng.random(n)
    mask = np.ones(n, dtype=bool)
    out = segment_rank_splice(main, np.full(n, 0.5), mask)
    np.testing.assert_array_equal(out, np.sort(main))
