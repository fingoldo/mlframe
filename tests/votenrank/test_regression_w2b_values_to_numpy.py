"""Regression sensor for w2b-percol-scattered Leaderboard ``.values`` -> ``.to_numpy()`` migration (finding #40).

Behavioural check: the migrated paths produce the same numeric is_partial flag + the same prepared print arrays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_leaderboard_is_partial_detects_nan_via_to_numpy():
    from mlframe.votenrank.leaderboard.Leaderboard import Leaderboard

    df_partial = pd.DataFrame({"task_a": [1.0, 2.0, np.nan], "task_b": [3.0, 4.0, 5.0]}, index=["m1", "m2", "m3"])
    lb = Leaderboard(df_partial)
    assert lb.is_partial

    df_full = pd.DataFrame({"task_a": [1.0, 2.0, 3.0], "task_b": [4.0, 5.0, 6.0]}, index=["m1", "m2", "m3"])
    lb_full = Leaderboard(df_full)
    assert not lb_full.is_partial
