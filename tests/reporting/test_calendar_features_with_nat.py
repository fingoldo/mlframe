"""A NaT timestamp no longer breaks calendar-feature detection (audit 2026-11-20, finding 20-F4)."""
import numpy as np
import pandas as pd

from mlframe.reporting.charts._calendar_features import calendar_feature_names


def test_hour_encoding_is_detected_with_a_missing_timestamp():
    """A column that is the hour of day must still be flagged when one timestamp is NaT."""
    n = 400
    ts = pd.date_range("2024-01-01", periods=n, freq="37min")
    ts = pd.Series(ts)
    ts.iloc[5] = pd.NaT
    hour = ts.dt.hour.fillna(0).astype(float).to_numpy()
    frame = pd.DataFrame({"hour_of_day": hour, "noise": np.random.default_rng(0).standard_normal(n)})
    names = calendar_feature_names(frame, ts.to_numpy())
    assert "hour_of_day" in names
