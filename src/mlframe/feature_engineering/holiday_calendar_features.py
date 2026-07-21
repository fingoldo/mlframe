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
from typing import Dict, Optional, Sequence, Tuple

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


@lru_cache(maxsize=64)
def _cached_holiday_dates_and_names(country: str, years: Tuple[int, ...]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Same coverage as :func:`_cached_holiday_dates` plus each date's holiday NAME, sorted in lockstep.

    A separate cache (rather than deriving names from the plain-dates cache) because ``country_holidays()``
    dict values are the whole point here -- a date can carry multiple stacked names (e.g. a national holiday
    that coincides with a local observance); those are joined with ``" / "`` so each date keeps exactly one
    name string aligned 1:1 with its slot in the sorted date array.
    """
    import holidays as holidays_pkg  # optional dependency; imported lazily so the base package has no hard dep.

    calendar = holidays_pkg.country_holidays(country, years=list(years))
    merged: Dict[pd.Timestamp, str] = {}
    for d, name in calendar.items():
        ts = pd.Timestamp(d)
        merged[ts] = f"{merged[ts]} / {name}" if ts in merged else name
    ordered = sorted(merged.items())
    dates = np.asarray(pd.DatetimeIndex([ts for ts, _ in ordered]).to_numpy())
    names = tuple(name for _, name in ordered)
    return dates, names


def _single_country_arrays(
    dates_np: np.ndarray,
    country: str,
    years: range,
    include_nearest_name: bool,
    none_sentinel: str,
    name_window_days: Optional[float],
) -> Dict[str, np.ndarray]:
    """Core per-country computation shared by the single- and multi-country entry points.

    Factored out so :func:`holiday_calendar_features`'s opt-in ``countries`` blend mode can compute each
    region's flags with the exact same logic as the single-``country`` path (bit-identical per-region
    output), rather than a parallel reimplementation that could silently drift.
    """
    if include_nearest_name:
        holiday_dates, holiday_names = _cached_holiday_dates_and_names(country, tuple(years))
    else:
        holiday_dates = _cached_holiday_dates(country, tuple(years))

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
        "is_holiday": is_holiday,
        "is_eve": is_eve,
        "days_since": days_since,
        "days_until": days_until,
    }

    if include_nearest_name:
        n = len(dates_np)
        nearest_name = np.full(n, none_sentinel, dtype=object)
        # Break the since/until tie in favor of whichever side is actually closer; a NaN distance
        # (edge of the padded calendar range) loses to any real distance on the other side.
        since_dist = np.where(has_before, days_since, np.inf)
        until_dist = np.where(has_after, days_until, np.inf)
        use_before = since_dist <= until_dist
        nearest_idx = np.where(use_before, idx_before, idx_after)
        nearest_dist = np.where(use_before, since_dist, until_dist)
        has_any = has_before | has_after
        if name_window_days is None:
            in_window = has_any
        else:
            in_window = has_any & (nearest_dist <= name_window_days)
        names_arr = np.asarray(holiday_names, dtype=object)
        nearest_name[in_window] = names_arr[nearest_idx[in_window]]
        out["nearest_holiday_name"] = nearest_name

    return out


def holiday_calendar_features(
    dates: pd.Series,
    country: str,
    years: Optional[range] = None,
    column_prefix: str = "holiday",
    include_nearest_name: bool = False,
    none_sentinel: str = "none",
    name_window_days: Optional[float] = None,
    countries: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Per-row holiday-calendar features: is-holiday, is-eve, days-since/until nearest holiday.

    Parameters
    ----------
    dates
        ``(n,)`` datetime-like series.
    country
        ISO country code understood by the ``holidays`` package (e.g. ``"US"``, ``"ES"``). Still required
        (and used verbatim for the default single-region output) even when ``countries`` is also given.
    years
        Year range to build the holiday calendar over; defaults to ``[dates.year.min()-1, dates.year.max()+1]``
        (a 1-year pad on each side so "days until next holiday" is defined near the series' edges).
    column_prefix
        Output column-name prefix.
    include_nearest_name
        When ``True``, add ``{prefix}_nearest_holiday_name`` -- unlike the blanket ``is_holiday`` flag (one
        average effect size across every holiday), this lets a downstream target-encoder learn a PER-HOLIDAY
        magnitude (e.g. Christmas vs a minor observance) since the ``holidays`` package's own dict values
        already carry each holiday's name and were otherwise discarded.
    none_sentinel
        Category emitted for rows outside ``name_window_days`` of any holiday.
    name_window_days
        Rows with the nearest holiday further than this many days away (in either direction) get
        ``none_sentinel`` instead of a name; ``None`` means always name the nearest holiday, however far.
    countries
        Opt-in multi-region blend mode, ``None`` by default (single fixed ``country``, output unchanged).
        When given a list of ISO country codes (e.g. a multinational entity's operating subsidiaries), emits
        PER-COUNTRY ``{prefix}_{code}_is_holiday`` / ``{prefix}_{code}_is_eve`` flags plus combined
        ``{prefix}_any_is_holiday`` / ``{prefix}_any_is_eve`` (OR across all listed countries) and
        ``{prefix}_any_days_since`` / ``{prefix}_any_days_until`` (nearest holiday in ANY listed region).
        A single fixed ``country`` misses a spike driven by a subsidiary region's local holiday that isn't
        also a holiday in the entity's primary country -- the combined flag catches it without the caller
        having to enumerate every region's columns downstream. ``include_nearest_name``/``none_sentinel``/
        ``name_window_days`` are ignored in this mode (per-region holiday names would need per-region
        cross-locale handling, already covered separately by
        :func:`mlframe.feature_engineering.holiday_locale_target_encoding.holiday_name_target_encode_cross_locale`).

    Returns
    -------
    pd.DataFrame
        Single-region mode (default): ``{prefix}_is_holiday`` (bool), ``{prefix}_is_eve`` (bool, the day
        immediately before a holiday), ``{prefix}_days_since`` / ``{prefix}_days_until`` (float, nearest
        holiday in either direction; NaN only possible at the very edges of the padded calendar range), and,
        if ``include_nearest_name``, ``{prefix}_nearest_holiday_name`` (str category).
        Multi-region mode (``countries`` given): one ``{prefix}_{code}_is_holiday`` / ``{prefix}_{code}_is_eve``
        pair per listed country plus the four combined ``{prefix}_any_*`` columns described above.
    """
    dates_dt = pd.to_datetime(dates)
    if years is None:
        years = range(int(dates_dt.dt.year.min()) - 1, int(dates_dt.dt.year.max()) + 2)
    # Normalize to midnight before any holiday-array comparison: holiday_dates (built from the
    # `holidays` package's date objects) are always midnight-normalized, so an un-normalized
    # timestamped input (e.g. 2024-12-25 14:30:00) would silently never match via np.isin --
    # is_holiday/is_eve always False with no error.
    dates_np = dates_dt.dt.normalize().to_numpy()

    if countries is not None:
        n = len(dates_np)
        out_multi: Dict[str, np.ndarray] = {}
        any_is_holiday = np.zeros(n, dtype=bool)
        any_is_eve = np.zeros(n, dtype=bool)
        any_days_since = np.full(n, np.nan)
        any_days_until = np.full(n, np.nan)
        for code in countries:
            arrs = _single_country_arrays(dates_np, code, years, False, none_sentinel, name_window_days)
            out_multi[f"{column_prefix}_{code}_is_holiday"] = arrs["is_holiday"]
            out_multi[f"{column_prefix}_{code}_is_eve"] = arrs["is_eve"]
            any_is_holiday |= arrs["is_holiday"]
            any_is_eve |= arrs["is_eve"]
            any_days_since = np.fmin(any_days_since, arrs["days_since"])
            any_days_until = np.fmin(any_days_until, arrs["days_until"])
        out_multi[f"{column_prefix}_any_is_holiday"] = any_is_holiday
        out_multi[f"{column_prefix}_any_is_eve"] = any_is_eve
        out_multi[f"{column_prefix}_any_days_since"] = any_days_since
        out_multi[f"{column_prefix}_any_days_until"] = any_days_until
        return pd.DataFrame(out_multi, index=dates.index if isinstance(dates, pd.Series) else None)

    arrs = _single_country_arrays(dates_np, country, years, include_nearest_name, none_sentinel, name_window_days)
    out: Dict[str, np.ndarray] = {
        f"{column_prefix}_is_holiday": arrs["is_holiday"],
        f"{column_prefix}_is_eve": arrs["is_eve"],
        f"{column_prefix}_days_since": arrs["days_since"],
        f"{column_prefix}_days_until": arrs["days_until"],
    }
    if include_nearest_name:
        out[f"{column_prefix}_nearest_holiday_name"] = arrs["nearest_holiday_name"]

    return pd.DataFrame(out, index=dates.index if isinstance(dates, pd.Series) else None)


__all__ = ["holiday_calendar_features"]
