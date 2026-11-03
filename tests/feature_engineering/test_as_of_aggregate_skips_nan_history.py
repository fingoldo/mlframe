"""leakage_safe_aggregate skips NaN history values like pandas: one NaN neither poisons later sums / means nor counts as a value."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.as_of_aggregate import leakage_safe_aggregate


def _run():
    """One entity with history [1, NaN, 3] at t = 1, 2, 3 and queries at cutoff 10 (all history) and 3 (first two rows)."""
    hist = pd.DataFrame({"e": ["a"] * 3, "t": [1, 2, 3], "v": [1.0, np.nan, 3.0]})
    query = pd.DataFrame({"e": ["a", "a"], "as_of": [10, 3]})
    return leakage_safe_aggregate(hist, "e", "t", query, {"v": ["sum", "mean", "count"]})


def test_sum_mean_and_count_ignore_the_nan_value():
    """Full history: sum 4, mean 2, count 2 (it was NaN, NaN, 3)."""
    out = _run()
    assert out.loc[0, "v_sum"] == 4.0 and out.loc[0, "v_mean"] == 2.0 and out.loc[0, "v_count"] == 2.0


def test_cutoff_inside_the_history_uses_only_earlier_non_nan_rows():
    """Cutoff 3 sees rows t=1 and t=2: sum 1, mean 1, count 1."""
    out = _run()
    assert out.loc[1, "v_sum"] == 1.0 and out.loc[1, "v_mean"] == 1.0 and out.loc[1, "v_count"] == 1.0
