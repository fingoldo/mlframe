"""Timezone-aware timestamps no longer make calendar-feature detection return nothing (tracker 2026-11-02 wave 9, TIME9-3)."""

import numpy as np
import pandas as pd

from mlframe.reporting.charts._calendar_features import calendar_feature_names


def test_tz_aware_series_still_finds_the_hour_column():
    """An hour-of-day column derived in UTC matches tz-aware UTC timestamps; before the fix the tz-aware dtype skipped the whole check."""
    ts = pd.Series(pd.date_range("2024-01-01", periods=400, freq="37min", tz="UTC"))
    frame = pd.DataFrame({"hour": ts.dt.hour.astype(float), "noise": np.random.default_rng(0).normal(size=400)})
    assert "hour" in calendar_feature_names(frame, ts)
