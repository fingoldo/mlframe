"""biz_value test for ``holiday_calendar_features(..., include_nearest_name=True)``.

Source: av_top3_xtreme_mlhack_datafest2017.md -- Mark Landry's Spanish holiday flags distinguished
national/local/observance TIERS rather than treating every holiday as one blanket event; the ``holidays``
package's ``country_holidays()`` calendar already carries each holiday's NAME (e.g. "Christmas Day" vs a
minor observance) in its dict values, which the existing ``is_holiday``/``is_eve`` flags collapse to one
bit. Different holidays cause genuinely different-magnitude demand spikes in practice (Christmas Eve ~70x
median per the source, a minor observance far less) -- a blanket ``is_holiday`` flag can only ever let a
model learn ONE average effect size across all holidays, while target-encoding the new per-row holiday NAME
lets it learn a per-holiday magnitude. Uses the existing leakage-safe
``mlframe.training.feature_handling.ordered_target_encoder.ordered_target_encode`` for the encoding step
(not reimplemented here).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering.holiday_calendar_features import holiday_calendar_features
from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode

# Real per-holiday multiplier assumption for the synthetic: a major shopping holiday spikes hard, a minor
# observance barely moves demand -- exactly the distinction a blanket is_holiday flag cannot represent.
_MAJOR_HOLIDAYS = {"Christmas Day", "Thanksgiving"}
_MAJOR_MULTIPLIER = 3.0
_MINOR_MULTIPLIER = 1.2
_BASELINE = 100.0


def _true_multiplier(name: str) -> float:
    if name == "none":
        return 1.0
    return _MAJOR_MULTIPLIER if any(major in name for major in _MAJOR_HOLIDAYS) else _MINOR_MULTIPLIER


def test_biz_val_nearest_holiday_name_target_encoding_beats_blanket_is_holiday_flag():
    dates = pd.Series(pd.date_range("2005-01-01", "2024-12-31", freq="D"))
    feats = holiday_calendar_features(dates, country="US", include_nearest_name=True, name_window_days=0)

    assert feats["holiday_nearest_holiday_name"].nunique() >= 5  # sanity: real distinct holiday names present.

    rng = np.random.default_rng(0)
    true_mult = feats["holiday_nearest_holiday_name"].map(_true_multiplier).to_numpy()
    y = _BASELINE * true_mult * (1.0 + rng.normal(0.0, 0.03, size=len(feats)))

    order = np.arange(len(feats))
    encoded_name = ordered_target_encode(feats["holiday_nearest_holiday_name"].to_numpy(), y, order=order, smoothing=1.0)
    encoded_blanket = ordered_target_encode(feats["holiday_is_holiday"].to_numpy(), y, order=order, smoothing=1.0)

    # Evaluate on HOLIDAY rows only, past a decade-long causal warm-up (each of the ~10 distinct named
    # holidays needs several PRIOR occurrences before its expanding-mean encoding stabilizes) -- the ~99%
    # non-holiday rows are baseline for both encodings and would wash out the per-holiday-magnitude signal
    # this feature is specifically about if included in the RMSE.
    is_holiday_row = feats["holiday_is_holiday"].to_numpy()
    warm = (dates.dt.year >= dates.dt.year.min() + 10).to_numpy()
    eval_mask = is_holiday_row & warm
    rmse_name = mean_squared_error(y[eval_mask], encoded_name[eval_mask]) ** 0.5
    rmse_blanket = mean_squared_error(y[eval_mask], encoded_blanket[eval_mask]) ** 0.5

    assert rmse_name < rmse_blanket * 0.5, (
        f"expected per-name target encoding to roughly halve RMSE vs blanket flag, got name={rmse_name:.2f} blanket={rmse_blanket:.2f}"
    )
    assert rmse_name < 12.0, f"expected per-name encoding RMSE close to the {_BASELINE * 0.03:.1f} noise floor, got {rmse_name:.2f}"


def test_holiday_calendar_features_nearest_holiday_name_matches_known_dates():
    dates = pd.Series(pd.to_datetime(["2024-12-25", "2024-07-04", "2024-03-15"]))
    feats = holiday_calendar_features(dates, country="US", include_nearest_name=True)

    assert feats.loc[0, "holiday_nearest_holiday_name"] == "Christmas Day"
    assert feats.loc[1, "holiday_nearest_holiday_name"] == "Independence Day"


def test_holiday_calendar_features_nearest_holiday_name_window_and_sentinel():
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2024-08-01"]))
    feats = holiday_calendar_features(dates, country="US", include_nearest_name=True, none_sentinel="NONE", name_window_days=5)

    assert feats.loc[0, "holiday_nearest_holiday_name"] == "New Year's Day"
    assert feats.loc[1, "holiday_nearest_holiday_name"] == "NONE"  # early August, far from any US holiday.


def test_holiday_calendar_features_include_nearest_name_default_off_keeps_old_columns():
    dates = pd.Series(pd.to_datetime(["2024-12-25"]))
    feats = holiday_calendar_features(dates, country="US")

    assert "holiday_nearest_holiday_name" not in feats.columns
    assert list(feats.columns) == ["holiday_is_holiday", "holiday_is_eve", "holiday_days_since", "holiday_days_until"]
