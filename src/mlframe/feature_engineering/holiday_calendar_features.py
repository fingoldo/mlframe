"""``holiday_calendar_features``: country-aware holiday/eve flags and days-to-nearest-holiday features.

Source: av_top3_mini_datahack_timeseries2016.md -- Christmas Eve sales ~70x median attributed to "Christmas
Sales"/"Year end sales adjustment"; av_top3_xtreme_mlhack_datafest2017.md -- Mark Landry incorporated "Spanish
holiday binary flags (national, local, observance)" plus a "percentile-based days-since-last-holiday metric."
A demand/traffic spike around a specific KNOWN calendar date (a holiday, or the day before it) is easy for a
model to miss entirely from generic day-of-week/month features -- an explicit is-holiday/is-eve flag plus
days-to-nearest-holiday lets the model localize the effect directly, rather than approximating it through
many higher-order date-part interactions.

Uses the ``holidays`` package (optional dependency, ``pip install mlframe[feature_engineering]``) for the
actual per-country calendar; distinct from :func:`mlframe.feature_engineering.event_proximity_decay.event_proximity_decay_features`,
which takes an arbitrary caller-supplied event-date list and produces a continuous distance-decayed "force"
feature -- this module is specifically holiday-calendar-sourced and produces discrete is-holiday/is-eve flags
plus raw days-to-nearest-holiday (the source's own explicit feature shapes), not a decay curve.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@lru_cache(maxsize=64)
def _cached_holiday_dates(country: str, years: Tuple[int, ...]) -> np.ndarray:
    """Build (and cache) a country's holiday-date array for a given year range.

    Rebuilding the ``holidays`` calendar (a real parsing/rule-evaluation pass per country/year) is the
    dominant cost of ``holiday_calendar_features`` when called repeatedly with the same country/year range
    (e.g. once per CV fold or per feature-engineering chunk in a pipeline) -- measured as ~55% of total
    cProfile time on a repeated-call benchmark. Caching keyed on the (small, hashable) inputs avoids it.
    """
    import holidays as holidays_pkg  # optional dependency; imported lazily so the base package has no hard dep.

    calendar = holidays_pkg.country_holidays(country, years=list(years))
    return np.asarray(pd.DatetimeIndex(sorted(pd.Timestamp(d) for d in calendar.keys())).to_numpy())


def holiday_calendar_features(dates: pd.Series, country: str, years: Optional[range] = None, column_prefix: str = "holiday") -> pd.DataFrame:
    """Per-row holiday-calendar features: is-holiday, is-eve, days-since/until nearest holiday.

    Parameters
    ----------
    dates
        ``(n,)`` datetime-like series.
    country
        ISO country code understood by the ``holidays`` package (e.g. ``"US"``, ``"ES"``).
    years
        Year range to build the holiday calendar over; defaults to ``[dates.year.min()-1, dates.year.max()+1]``
        (a 1-year pad on each side so "days until next holiday" is defined near the series' edges).
    column_prefix
        Output column-name prefix.

    Returns
    -------
    pd.DataFrame
        ``{prefix}_is_holiday`` (bool), ``{prefix}_is_eve`` (bool, the day immediately before a holiday),
        ``{prefix}_days_since`` / ``{prefix}_days_until`` (float, nearest holiday in either direction; NaN
        only possible at the very edges of the padded calendar range).
    """
    dates_dt = pd.to_datetime(dates)
    if years is None:
        years = range(int(dates_dt.dt.year.min()) - 1, int(dates_dt.dt.year.max()) + 2)

    holiday_dates = _cached_holiday_dates(country, tuple(years))

    dates_np = dates_dt.to_numpy()
    is_holiday = np.isin(dates_np, holiday_dates)

    eve_dates = holiday_dates - np.timedelta64(1, "D")
    is_eve = np.isin(dates_np, eve_dates)

    # searchsorted gives the insertion point -> nearest holiday on/after and on/before each date. Using
    # side="left" for "after" and side="right"-1 for "before" makes both inclusive of the date ITSELF when
    # it is a holiday, so days_since == days_until == 0 on the holiday (symmetric, not just days_until).
    idx_after = np.searchsorted(holiday_dates, dates_np, side="left")
    idx_before = np.searchsorted(holiday_dates, dates_np, side="right") - 1

    days_until = np.full(len(dates_np), np.nan)
    has_after = idx_after < len(holiday_dates)
    days_until[has_after] = (holiday_dates[idx_after[has_after]] - dates_np[has_after]) / np.timedelta64(1, "D")

    days_since = np.full(len(dates_np), np.nan)
    has_before = idx_before >= 0
    days_since[has_before] = (dates_np[has_before] - holiday_dates[idx_before[has_before]]) / np.timedelta64(1, "D")

    out: Dict[str, np.ndarray] = {
        f"{column_prefix}_is_holiday": is_holiday,
        f"{column_prefix}_is_eve": is_eve,
        f"{column_prefix}_days_since": days_since,
        f"{column_prefix}_days_until": days_until,
    }
    return pd.DataFrame(out, index=dates.index if isinstance(dates, pd.Series) else None)


__all__ = ["holiday_calendar_features"]
