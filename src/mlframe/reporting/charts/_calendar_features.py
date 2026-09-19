"""Detect features that are calendar encodings of the row timestamp (hour, day of month, weekday, month and their sin/cos).

Drift charts bucket rows by time, and an adversarial check separates an earlier period from a later one; a calendar
encoding of the timestamp differs between such groups by construction. A production PSI heatmap showed
``job_posted_at_day_sin`` at PSI 5.6 in every bucket and ``job_posted_at_weekday_sin`` at 14.2, and the adversarial
chart ranked the same columns as the top "drift drivers" of an AUC 0.985 -- all of it the calendar, none of it drift.
Detection is by value, not by name: a feature whose values match one of those encodings (|corr| > 0.99 on a sample)
is a calendar feature whatever it is called.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from mlframe.reporting.charts._drift_shared import _frame_columns

_CORR_MIN = 0.99
_SAMPLE = 20_000


def _calendar_encodings(ts: np.ndarray) -> List[np.ndarray]:
    """Hour, day of month, weekday, month, day of year as raw integers and as sin/cos of their cycle."""
    t = np.asarray(ts).astype("datetime64[s]")
    days = t.astype("datetime64[D]")
    hour = ((t - days).astype(np.int64) // 3600).astype(np.float64)
    months = days.astype("datetime64[M]")
    dom = (days - months).astype(np.int64).astype(np.float64) + 1.0
    weekday = ((days.astype(np.int64) + 3) % 7).astype(np.float64)
    month = (months.astype(np.int64) % 12).astype(np.float64) + 1.0
    doy = (days - days.astype("datetime64[Y]")).astype(np.int64).astype(np.float64) + 1.0
    out: List[np.ndarray] = []
    for raw, period in ((hour, 24.0), (dom, 31.0), (weekday, 7.0), (month, 12.0), (doy, 366.0)):
        out.append(raw)
        angle = 2.0 * np.pi * raw / period
        out.append(np.sin(angle))
        out.append(np.cos(angle))
    return out


def calendar_feature_names(frame: Any, timestamps: np.ndarray, feature_names: Optional[Sequence[str]] = None, seed: int = 0) -> List[str]:
    """Names of the columns of ``frame`` whose values are a calendar encoding of ``timestamps`` (row-aligned)."""
    ts = np.asarray(timestamps)
    if ts.size == 0 or not np.issubdtype(ts.dtype, np.datetime64):
        return []
    cols, names = _frame_columns(frame, feature_names)
    n = min(ts.shape[0], *(len(c) for c in cols)) if cols else 0
    if n < 50:
        return []
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(n, _SAMPLE), replace=False))
    ts_s = ts[idx]
    ok_t = ~np.isnat(ts_s)
    encodings = [e[ok_t] for e in _calendar_encodings(ts_s[ok_t])]
    # Over a window shorter than the cycle (day of year / month across half a year) an encoding is just a monotone
    # function of time, and matching it would flag any trending feature as "calendar". Keep only encodings that
    # actually cycle within the window.
    t_num = ts_s[ok_t].astype("datetime64[s]").astype(np.int64).astype(np.float64)
    encodings = [e for e in encodings if np.ptp(e) > 0 and abs(float(np.corrcoef(e, t_num)[0, 1])) < 0.9]
    found: List[str] = []
    for col, name in zip(cols, names):
        try:
            v = np.asarray(col, dtype=np.float64)[idx][ok_t]
        except (TypeError, ValueError):
            continue
        fin = np.isfinite(v)
        if fin.sum() < 50 or np.ptp(v[fin]) == 0:
            continue
        for e in encodings:
            ee = e[fin]
            if np.ptp(ee) == 0:
                continue
            if abs(float(np.corrcoef(v[fin], ee)[0, 1])) > _CORR_MIN:
                found.append(str(name))
                break
    return found
