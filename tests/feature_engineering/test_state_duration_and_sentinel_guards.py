"""Missing states are not possession (24-F7); skewed binary and categorical columns are not sentinels (24-F8)."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.sentinel_missing_count import detect_column_sentinel
from mlframe.feature_engineering.state_duration import time_since_state_change


def test_missing_state_is_not_counted_as_possession():
    """A NaN state used to cast to True, so the row was counted as possession (duration 1 instead of NaN)."""
    state = np.array([0.0, np.nan, 1.0, 1.0], dtype=np.float64)
    group = np.zeros(4, dtype=np.int64)
    out = time_since_state_change(state, group)
    assert np.isnan(out["possession_duration"][1])
    assert out["possession_duration"][2] == 1.0


def test_skewed_binary_label_is_not_flagged_as_sentinel():
    """A 5:1 binary label has one other level; it is an imbalance, not a placeholder value."""
    s = pd.Series([0] * 50 + [1] * 10)
    assert detect_column_sentinel(s) is None


def test_skewed_categorical_is_not_flagged_as_sentinel():
    """Categorical levels are real values, never placeholders."""
    s = pd.Series(["a"] * 80 + ["b", "c", "d"] * 3)
    assert detect_column_sentinel(s) is None


def test_numeric_placeholder_is_still_detected():
    """A spike of -999 among many distinct real values is still a sentinel."""
    rng = np.random.default_rng(0)
    real = rng.normal(size=300)
    s = pd.Series(np.concatenate([real, np.full(60, -999.0)]))
    assert detect_column_sentinel(s) == -999.0
