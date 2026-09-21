"""Calendar encodings of the timestamp must be recognised by value so drift charts can set them aside."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.reporting.charts._calendar_features import calendar_feature_names


def test_detects_calendar_encodings_not_real_features():
    rng = np.random.default_rng(0)
    n = 5000
    ts = np.datetime64("2026-03-01") + np.sort(rng.integers(0, 180 * 86400, n)).astype("timedelta64[s]")
    s = pd.Series(ts)
    frame = pd.DataFrame(
        {
            "posted_day_sin": np.sin(2 * np.pi * s.dt.day / 31),
            "posted_weekday": s.dt.weekday.astype(float),
            "posted_hour_cos": np.cos(2 * np.pi * s.dt.hour / 24),
            "budget": rng.lognormal(size=n),
            "trend": np.arange(n, dtype=float) + rng.normal(scale=50, size=n),
        }
    )
    found = set(calendar_feature_names(frame, ts.astype("datetime64[us]")))
    assert found == {"posted_day_sin", "posted_weekday", "posted_hour_cos"}


def test_numeric_timestamps_detect_nothing():
    frame = pd.DataFrame({"a": np.arange(100.0)})
    assert calendar_feature_names(frame, np.arange(100.0)) == []
