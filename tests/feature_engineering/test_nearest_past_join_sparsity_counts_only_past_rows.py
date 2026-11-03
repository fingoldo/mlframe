"""The fallback tier's min_group_size floor counts only right rows at or before the left row, never rows from its future."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.nearest_past_join import nearest_past_join


def _frames():
    """Group 'a' has 1 past right row (t=1) and 9 future ones (t=100..108); group 'b' adds six past rows so the global tier has 7 historical rows."""
    right = pd.DataFrame({"t": [1, *range(100, 109), 2, 3, 4, 5, 6, 7], "g": ["a"] * 10 + ["b"] * 6, "v": [10.0] + [99.0] * 9 + [5.0] * 6})
    left = pd.DataFrame({"t": [50], "g": ["a"]})
    return left, right


def test_group_with_one_past_and_nine_future_rows_is_sparse_for_min_group_size_five():
    """Fine tier ['g'] has only 1 historical row for t=50 (< 5): the row must fall through to the global tier, not be accepted because 10 rows exist overall."""
    left, right = _frames()
    out = nearest_past_join(left, right, "t", by=["g"], right_value_cols=["v"], fallback_by_chain=[None], min_group_size=5, tier_col="tier")
    assert int(out["tier"].iloc[0]) == 1


def test_group_with_enough_past_rows_stays_on_the_fine_tier():
    """Five past rows satisfy the floor (negative control)."""
    right = pd.DataFrame({"t": [1, 2, 3, 4, 5], "g": ["a"] * 5, "v": np.arange(5.0)})
    left = pd.DataFrame({"t": [50], "g": ["a"]})
    out = nearest_past_join(left, right, "t", by=["g"], right_value_cols=["v"], fallback_by_chain=[None], min_group_size=5, tier_col="tier")
    assert int(out["tier"].iloc[0]) == 0
