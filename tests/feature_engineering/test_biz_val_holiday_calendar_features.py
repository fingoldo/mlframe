"""biz_value test for ``feature_engineering.holiday_calendar_features.holiday_calendar_features``.

Source: av_top3_mini_datahack_timeseries2016.md -- Christmas Eve sales ~70x median attributed to "Christmas
Sales"/"Year end sales adjustment"; av_top3_xtreme_mlhack_datafest2017.md -- Spanish holiday binary flags plus
a days-since-last-holiday metric. A generic day-of-week feature carries essentially no information about
WHICH specific calendar date a demand spike lands on; the holiday/eve flags should.
"""
from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.holiday_calendar_features import holiday_calendar_features


def test_biz_val_holiday_flags_separate_spike_days_better_than_day_of_week():
    dates = pd.Series(pd.date_range("2020-01-01", "2024-12-31", freq="D"))
    feats = holiday_calendar_features(dates, country="US")

    # spike label: demand spike happens on holidays and their eves (the source's own claimed mechanism).
    label = (feats["holiday_is_holiday"] | feats["holiday_is_eve"]).astype(int)
    assert 0.02 < label.mean() < 0.15  # sanity: spikes are a real minority class, not degenerate.

    dow = dates.dt.dayofweek
    auc_dow = roc_auc_score(label, dow)
    holiday_signal = feats["holiday_is_holiday"].astype(int) + feats["holiday_is_eve"].astype(int)
    auc_holiday = roc_auc_score(label, holiday_signal)

    assert auc_holiday >= 0.99, f"expected holiday/eve flags to near-perfectly identify spike days by construction, got auc={auc_holiday:.4f}"
    assert auc_holiday > auc_dow + 0.4, f"expected holiday flags to massively beat day-of-week, got holiday={auc_holiday:.4f} dow={auc_dow:.4f}"


def test_holiday_calendar_features_flags_known_us_holidays():
    dates = pd.Series(pd.to_datetime(["2024-12-24", "2024-12-25", "2024-12-26", "2025-01-01"]))
    feats = holiday_calendar_features(dates, country="US")

    assert feats["holiday_is_eve"].tolist() == [True, False, False, False]
    assert feats["holiday_is_holiday"].tolist() == [False, True, False, True]


def test_holiday_calendar_features_days_since_until_are_consistent():
    dates = pd.Series(pd.to_datetime(["2024-12-20", "2024-12-25", "2024-12-27"]))
    feats = holiday_calendar_features(dates, country="US")

    assert feats.loc[0, "holiday_days_until"] == 5.0  # Dec 20 -> Dec 25.
    assert feats.loc[1, "holiday_days_since"] == 0.0  # the holiday itself.
    assert feats.loc[2, "holiday_days_since"] == 2.0  # Dec 27, 2 days after Dec 25.
