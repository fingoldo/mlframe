"""``event_proximity_decay_features``: distance-decayed "strength" columns for known event dates.

Source: 5th_m5-forecasting-accuracy.md -- "I created features from the remaining days for the event...
values in this column refer to the 'strength' of the proximity of the event limited to 30... sum of all
these 'forces'." A per-event linear-decay kernel (``max(0, cap - |days_to_event|)``) turns a sparse set of
known event dates into a smooth, distance-aware signal a model can use even on days AWAY from the event
itself (a pre/post-event ramp), rather than a single sparse binary "is-event-day" indicator. Generalizes
beyond calendar events to any known reference-date set (holidays, corporate actions, scheduled maintenance).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def event_proximity_decay_features(
    dates: pd.Series,
    event_dates: Sequence,
    cap: int = 30,
    column_prefix: str = "event_proximity",
    cap_before: Optional[int] = None,
    cap_after: Optional[int] = None,
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
        ``cap`` days from the event in either direction. Used symmetrically unless ``cap_before``/
        ``cap_after`` are given.
    column_prefix
        Output column-name prefix.
    cap_before
        Opt-in: decay cap applied to rows BEFORE the event (anticipation/ramp-up leg). When either
        ``cap_before`` or ``cap_after`` is given, the other defaults to ``cap`` -- letting the pre-event
        buildup and post-event falloff decay at different rates (e.g. a promotion's demand ramps up over
        weeks but decays within days once it ends). ``None`` (default, both) reproduces the exact
        symmetric behavior above -- bit-identical to omitting these parameters.
    cap_after
        Opt-in: decay cap applied to rows AT/AFTER the event (post-event falloff leg). See ``cap_before``.

    Returns
    -------
    pd.DataFrame
        One column per event (``{prefix}_event{i}``, individual decayed strength) plus
        ``{prefix}_total_force`` (row-wise sum across all events -- the combined "force" of every nearby
        event, matching the source's own aggregate feature).
    """
    dates_arr = pd.to_datetime(dates) if not pd.api.types.is_numeric_dtype(dates) else dates.to_numpy()
    is_datetime = pd.api.types.is_datetime64_any_dtype(dates_arr) if hasattr(dates_arr, "dtype") else False

    asymmetric = cap_before is not None or cap_after is not None
    effective_cap_before = float(cap_before if cap_before is not None else cap)
    effective_cap_after = float(cap_after if cap_after is not None else cap)

    out: Dict[str, np.ndarray] = {}
    total_force = np.zeros(len(dates), dtype=np.float64)
    for i, event_date in enumerate(event_dates):
        if is_datetime:
            days_to_event = (pd.to_datetime(dates_arr) - pd.Timestamp(event_date)).dt.days.to_numpy().astype(np.float64)
        else:
            days_to_event = (np.asarray(dates_arr, dtype=np.float64) - float(event_date))
        if asymmetric:
            # days_to_event < 0 -> row is BEFORE the event (anticipation leg); >= 0 -> AT/AFTER (falloff leg).
            row_cap = np.where(days_to_event < 0, effective_cap_before, effective_cap_after)
            strength = np.maximum(0.0, row_cap - np.abs(days_to_event))
        else:
            strength = np.maximum(0.0, cap - np.abs(days_to_event))
        out[f"{column_prefix}_event{i}"] = strength
        total_force += strength

    out[f"{column_prefix}_total_force"] = total_force
    return pd.DataFrame(out, index=dates.index if hasattr(dates, "index") else None)


__all__ = ["event_proximity_decay_features"]
