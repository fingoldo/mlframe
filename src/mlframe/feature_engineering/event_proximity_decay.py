"""``event_proximity_decay_features``: distance-decayed "strength" columns for known event dates.

Source: 5th_m5-forecasting-accuracy.md -- "I created features from the remaining days for the event...
values in this column refer to the 'strength' of the proximity of the event limited to 30... sum of all
these 'forces'." A per-event linear-decay kernel (``max(0, cap - |days_to_event|)``) turns a sparse set of
known event dates into a smooth, distance-aware signal a model can use even on days AWAY from the event
itself (a pre/post-event ramp), rather than a single sparse binary "is-event-day" indicator. Generalizes
beyond calendar events to any known reference-date set (holidays, corporate actions, scheduled maintenance).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd


def event_proximity_decay_features(
    dates: pd.Series,
    event_dates: Sequence,
    cap: int = 30,
    column_prefix: str = "event_proximity",
) -> pd.DataFrame:
    """Per-event and aggregate distance-decayed proximity "strength" columns.

    Parameters
    ----------
    dates
        ``(n,)`` the series' own date/day values (datetime-like or integer day index).
    event_dates
        Known event reference dates/days, same dtype family as ``dates``.
    cap
        Decay cap: a row's strength for an event is ``max(0, cap - |days_to_event|)``, zero beyond
        ``cap`` days from the event in either direction.
    column_prefix
        Output column-name prefix.

    Returns
    -------
    pd.DataFrame
        One column per event (``{prefix}_event{i}``, individual decayed strength) plus
        ``{prefix}_total_force`` (row-wise sum across all events -- the combined "force" of every nearby
        event, matching the source's own aggregate feature).
    """
    dates_arr = pd.to_datetime(dates) if not pd.api.types.is_numeric_dtype(dates) else dates.to_numpy()
    is_datetime = pd.api.types.is_datetime64_any_dtype(dates_arr) if hasattr(dates_arr, "dtype") else False

    out: Dict[str, np.ndarray] = {}
    total_force = np.zeros(len(dates), dtype=np.float64)
    for i, event_date in enumerate(event_dates):
        if is_datetime:
            days_to_event = (pd.to_datetime(dates_arr) - pd.Timestamp(event_date)).dt.days.to_numpy().astype(np.float64)
        else:
            days_to_event = (np.asarray(dates_arr, dtype=np.float64) - float(event_date))
        strength = np.maximum(0.0, cap - np.abs(days_to_event))
        out[f"{column_prefix}_event{i}"] = strength
        total_force += strength

    out[f"{column_prefix}_total_force"] = total_force
    return pd.DataFrame(out, index=dates.index if hasattr(dates, "index") else None)


__all__ = ["event_proximity_decay_features"]
