"""The two-step meta table is float64 with one row per model, even when a model has no data in a group."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.votenrank import Leaderboard


def _table():
    """Four models, four tasks in two groups; model ``d`` has no scores in the second group."""
    t = pd.DataFrame(
        {"t1": [3.0, 2.0, 1.0, 0.5], "t2": [3.0, 1.0, 2.0, 0.4], "t3": [1.0, 3.0, 2.0, np.nan], "t4": [2.0, 3.0, 1.0, np.nan]},
        index=["a", "b", "c", "d"],
    )
    return t, {"g1": ["t1", "t2"], "g2": ["t3", "t4"]}


def test_meta_table_is_float64_and_has_every_model():
    """All meta columns are float64 (not object) and the index is the full model list."""
    t, groups = _table()
    meta = Leaderboard(t).get_meta_leaderboard("mean", groups).table
    assert list(meta.index) == ["a", "b", "c", "d"]
    assert all(dt == np.float64 for dt in meta.dtypes)


def test_model_without_data_in_a_group_is_nan_there():
    """Model d has no data in g2, so its g2 meta score is NaN while the other models keep theirs."""
    t, groups = _table()
    meta = Leaderboard(t).get_meta_leaderboard("mean", groups).table
    assert np.isnan(meta.loc["d", "g2"])
    assert not np.isnan(meta.loc[["a", "b", "c"], "g2"]).any()
