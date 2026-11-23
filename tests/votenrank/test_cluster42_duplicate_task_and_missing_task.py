"""Cluster 42 regression: duplicate task columns are refused, and a missing task is named as such (audit 2026-11-20, findings 42-F4, 42-F7)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard


def test_duplicate_task_columns_are_refused():
    """A table with two columns named t1 raises instead of silently addressing only the first (42-F4)."""
    table = pd.DataFrame(np.ones((3, 3)), index=["m1", "m2", "m3"], columns=["t1", "t1", "t2"])
    with pytest.raises(ValueError, match="duplicate task column"):
        Leaderboard(table)


def test_all_nan_task_is_reported_as_missing_not_zero_score():
    """A fully-NaN task column produces a missing-task error, not the zero/negative-score message (42-F7)."""
    table = pd.DataFrame({"t1": [1.0, 2.0], "t2": [np.nan, np.nan]}, index=["m1", "m2"])
    lb = Leaderboard(table, allow_partial=True)
    with pytest.raises(ValueError, match="have no scores at all"):
        lb.mean_ranking(mean_type="geometric")


def test_clean_table_still_ranks():
    """Control: a complete table with unique tasks still ranks (42-F4)."""
    table = pd.DataFrame({"t1": [1.0, 2.0], "t2": [3.0, 1.0]}, index=["m1", "m2"])
    assert len(Leaderboard(table).mean_ranking(mean_type="arithmetic")) == 2
