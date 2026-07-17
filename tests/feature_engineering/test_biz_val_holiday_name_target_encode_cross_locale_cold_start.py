"""biz_value test for ``holiday_name_target_encode_cross_locale``'s opt-in cross-locale shrinkage mode.

Source: extends av_top3_xtreme_mlhack_datafest2017.md's per-holiday-name target-encoding recipe (see
``test_biz_val_holiday_calendar_features_nearest_holiday_name.py``) to the cold-start problem: a country/
locale with little or no PAST history of a given holiday falls back to the flat global prior under
same-country-only encoding, even when other countries already carry rich history for a holiday of the SAME
name (e.g. "Christmas Day" is observed with a similar demand-spike magnitude across dozens of ``holidays``-
package locales). This synthetic reproduces that: a RICH country ("US") observes every holiday every year for
30 years; a COLD-START country ("XX") only ever observes "Christmas Day" twice, near the END of the timeline
(after the rich country has already built up 29 years of "Christmas Day" history). Same-country-only encoding
has zero local history for XX's Christmas Day rows and collapses to the grand mean (a blend of ALL holidays'
multipliers); cross-locale shrinkage borrows the already-converged cross-country "Christmas Day" mean instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering.holiday_locale_target_encoding import holiday_name_target_encode_cross_locale
from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode

_BASELINE = 100.0
_MULTIPLIERS = {
    "Christmas Day": 3.0,
    "New Year's Day": 2.0,
    "Labor Day": 1.5,
    "Minor Observance": 1.1,
}
_RICH_YEARS = 30
_COLD_YEARS = (28, 29)  # the cold-start country's only two Christmas Day occurrences, near the end of the timeline.


def _build_synthetic():
    """Helper: Build synthetic."""
    rng = np.random.default_rng(0)
    names = []
    countries = []
    y = []
    order = []

    # Rich country "US": every holiday, every year, in chronological (year-major) order.
    for year in range(_RICH_YEARS):
        for name, mult in _MULTIPLIERS.items():
            names.append(name)
            countries.append("US")
            y.append(_BASELINE * mult * (1.0 + rng.normal(0.0, 0.04)))
            order.append(year * 10)  # 10-slot-per-year timeline; leaves room for the cold country's events.

    # Cold-start country "XX": only "Christmas Day", only twice, both near the end of the timeline -- by
    # then the rich country has already built ~28 years of "Christmas Day" history for the cross-locale prior.
    for year in _COLD_YEARS:
        names.append("Christmas Day")
        countries.append("XX")
        y.append(_BASELINE * _MULTIPLIERS["Christmas Day"] * (1.0 + rng.normal(0.0, 0.04)))
        order.append(year * 10 + 1)  # placed just after that year's US events in the shared timeline.

    return np.asarray(names), np.asarray(countries), np.asarray(y, dtype=np.float64), np.asarray(order)


def test_biz_val_holiday_name_target_encode_cross_locale_cold_start_country_beats_same_country_only():
    """Biz val holiday name target encode cross locale cold start country beats same country only."""
    names, countries, y, order = _build_synthetic()

    same_country_only = holiday_name_target_encode_cross_locale(names, countries, y, order=order, smoothing=1.0)
    cross_locale = holiday_name_target_encode_cross_locale(names, countries, y, order=order, smoothing=1.0, cross_locale_shrinkage=5.0)

    cold_mask = countries == "XX"
    true_y = y[cold_mask]

    rmse_same_country = mean_squared_error(true_y, same_country_only[cold_mask]) ** 0.5
    rmse_cross_locale = mean_squared_error(true_y, cross_locale[cold_mask]) ** 0.5

    assert rmse_cross_locale < rmse_same_country * 0.35, (
        f"expected cross-locale shrinkage to cut cold-start RMSE by >65%, got same_country={rmse_same_country:.2f} cross_locale={rmse_cross_locale:.2f}"
    )
    # The true Christmas-Day multiplier is 3.0x baseline (300); a converged 28-year cross-country prior
    # should land close to it, not merely closer-than-baseline.
    assert rmse_cross_locale < 20.0, f"expected cross-locale RMSE near the ~4% noise floor, got {rmse_cross_locale:.2f}"


def test_holiday_name_target_encode_cross_locale_default_is_bit_identical_to_composite_key_encoding():
    """Holiday name target encode cross locale default is bit identical to composite key encoding."""
    names, countries, y, order = _build_synthetic()

    default_out = holiday_name_target_encode_cross_locale(names, countries, y, order=order, smoothing=1.0)

    composite_key = np.char.add(np.char.add(countries.astype(str), "\x00"), names.astype(str))
    reference_out = ordered_target_encode(composite_key, y, order=order, smoothing=1.0)

    np.testing.assert_array_equal(default_out, reference_out)


def test_holiday_name_target_encode_cross_locale_rejects_non_positive_shrinkage():
    """Holiday name target encode cross locale rejects non positive shrinkage."""
    names, countries, y, order = _build_synthetic()
    import pytest

    with pytest.raises(ValueError):
        holiday_name_target_encode_cross_locale(names, countries, y, order=order, cross_locale_shrinkage=0.0)
