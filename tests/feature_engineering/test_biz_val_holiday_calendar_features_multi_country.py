"""biz_value test for ``holiday_calendar_features(..., countries=[...])`` (opt-in multi-region blend mode).

Source: same av_top3_mini_datahack_timeseries2016.md / av_top3_xtreme_mlhack_datafest2017.md motivation as the
single-country test -- a demand spike lands on a KNOWN calendar date. For a multinational entity operating
across several regions, the spike-causing holiday can be a LOCAL one that only exists in a subsidiary's
calendar (e.g. Canada Day, July 1) and never appears on the parent country's calendar (e.g. US). A feature
pipeline hard-coded to one fixed ``country`` structurally cannot see those spikes -- it isn't a modeling
weakness, it's a missing column. The new ``countries=[...]`` blend mode's combined ``{prefix}_any_is_holiday``
flag is built to catch a spike driven by EITHER region's calendar.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score

from mlframe.feature_engineering.holiday_calendar_features import holiday_calendar_features


def test_biz_val_holiday_calendar_features_multi_country_any_flag_catches_subsidiary_only_spikes():
    dates = pd.Series(pd.date_range("2020-01-01", "2024-12-31", freq="D"))

    # Single fixed-country pipeline (default/legacy behavior): parent entity's own calendar only.
    single = holiday_calendar_features(dates, country="US")

    # Opt-in multi-region blend: same entity now also tags a Canadian subsidiary's regional calendar.
    multi = holiday_calendar_features(dates, country="US", countries=["US", "CA"])

    # Ground-truth spike label: demand spikes on a holiday/eve in EITHER region (the real multinational
    # mechanism -- whichever subsidiary's local holiday it is, the group sees elevated activity that day).
    label = (multi["holiday_any_is_holiday"] | multi["holiday_any_is_eve"]).astype(int)
    assert 0.03 < label.mean() < 0.20  # sanity: a real minority class, not degenerate.

    # The single-country flag can only ever see the US calendar -- CA-only holidays (e.g. Canada Day, Victoria
    # Day) are invisible to it, so it should systematically under-catch the combined-region spike label.
    single_signal = single["holiday_is_holiday"].astype(int) + single["holiday_is_eve"].astype(int)
    auc_single = roc_auc_score(label, single_signal)

    multi_signal = multi["holiday_any_is_holiday"].astype(int) + multi["holiday_any_is_eve"].astype(int)
    auc_multi = roc_auc_score(label, multi_signal)

    assert auc_multi >= 0.99, f"expected the combined any-region flag to near-perfectly identify spike days by construction, got auc={auc_multi:.4f}"
    assert auc_multi > auc_single + 0.03, (
        f"expected the multi-country combined flag to beat the single fixed-country flag on cross-region "
        f"spikes, got multi={auc_multi:.4f} single={auc_single:.4f}"
    )

    # Per-country columns are present alongside the combined ones, so the caller can still see WHICH region.
    assert "holiday_US_is_holiday" in multi.columns and "holiday_CA_is_holiday" in multi.columns


def test_holiday_calendar_features_countries_default_omitted_matches_single_country_output():
    dates = pd.Series(pd.date_range("2023-01-01", "2023-12-31", freq="D"))

    baseline = holiday_calendar_features(dates, country="US")
    explicit_none = holiday_calendar_features(dates, country="US", countries=None)

    pd.testing.assert_frame_equal(baseline, explicit_none)


def test_holiday_calendar_features_multi_country_any_flag_matches_manual_or():
    dates = pd.Series(pd.to_datetime(["2024-07-01", "2024-07-04", "2024-12-25", "2024-03-29"]))
    multi = holiday_calendar_features(dates, country="US", countries=["US", "CA"])

    # 2024-07-01 is Canada Day (CA-only), 2024-07-04 is US Independence Day (US-only), 2024-12-25 is a
    # holiday in both, 2024-03-29 is Good Friday (CA-only per the ``holidays`` package's federal CA calendar,
    # not observed as a US federal holiday).
    manual_any = multi["holiday_US_is_holiday"] | multi["holiday_CA_is_holiday"]
    assert (multi["holiday_any_is_holiday"] == manual_any).all()
    assert multi["holiday_any_is_holiday"].tolist() == [True, True, True, True]
    assert multi["holiday_US_is_holiday"].tolist() == [False, True, True, False]
    assert multi["holiday_CA_is_holiday"].tolist() == [True, False, True, True]
