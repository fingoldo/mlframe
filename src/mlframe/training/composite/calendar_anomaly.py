"""Automated calendar-anomaly (holiday/event spike) detection for a datetime-indexed target.

Manually spotting and correcting anomalous calendar days (a sales team's found ~70x median spikes on Dec
24-25, another independently added Spanish holiday flags) doesn't scale and misses days nobody thought to
check. This scans a datetime-indexed target for days whose value deviates far from a robust LOCAL baseline
(rolling median, resistant to the very spikes being detected -- a plain rolling mean would be dragged up by
them), flags candidate anomalous days, and returns both a binary flag and a "divided-out" corrected series.

A single deviation threshold conflates two very different situations: a genuine one-off event (e.g. a
holiday) and a RECURRING calendar pattern (e.g. every-Sunday sales peak) that happens to also clear the
deviation bar at every occurrence. The former deserves the "divide it out" treatment above; the latter is
actually signal -- it deserves its own feature (day-of-week / day-of-period dummy), not correction-away.
The opt-in ``recurrence_period`` parameter classifies each flagged day by how often OTHER days sharing the
same calendar phase (``index % recurrence_period``) were also flagged, splitting ``flagged`` into
``recurring`` (repeats often enough to look seasonal) and ``rare`` (true one-off candidates).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def detect_calendar_anomalies(
    y: np.ndarray,
    window: int = 14,
    deviation_ratio_threshold: float = 3.0,
    min_periods: int = 5,
    recurrence_period: Optional[int] = None,
    recurrence_min_occurrences: int = 3,
) -> dict:
    """Flag days whose value deviates by more than ``deviation_ratio_threshold``x from a robust local baseline.

    Parameters
    ----------
    y
        ``(n,)`` numeric series in chronological (row) order, one value per calendar day (or other regular
        period).
    window
        Rolling-median window length (centered) used as the robust local baseline.
    deviation_ratio_threshold
        A day is flagged when ``max(y_t, baseline_t) / min(y_t, baseline_t) > deviation_ratio_threshold``
        (symmetric: catches both unusually high AND unusually low days).
    min_periods
        Minimum rolling-window observations required before a baseline is trusted; days before that get
        ``flagged=False`` (insufficient history to judge).
    recurrence_period
        Opt-in. When set (e.g. ``7`` for weekly seasonality), flagged days are further classified by how
        many OTHER days at the same phase (``index % recurrence_period``) are also flagged: ``recurring``
        (likely a real seasonal pattern, e.g. weekly peaks) vs ``rare`` (likely a true one-off event, e.g. a
        holiday). ``None`` (default) leaves the return dict and all other outputs exactly as before.
    recurrence_min_occurrences
        Minimum number of flagged occurrences at the same phase (including the day itself) for that phase to
        be classified ``recurring``. Only used when ``recurrence_period`` is set.

    Returns
    -------
    dict
        ``flagged`` ``(n,)`` bool, ``baseline`` ``(n,)`` the rolling-median local baseline,
        ``deviation_ratio`` ``(n,)`` float, ``corrected`` ``(n,)`` -- ``y`` with flagged days divided by
        their own deviation ratio (pulling them back toward the local baseline; unflagged days unchanged).
        When ``recurrence_period`` is set, also includes ``recurring`` ``(n,)`` bool (flagged days whose
        calendar phase recurs often enough), ``rare`` ``(n,)`` bool (flagged days that don't -- true one-off
        candidates), and ``phase_flag_count`` ``(n,)`` int (how many days share this day's phase and are
        flagged, broadcast to every day at that phase).
    """
    y = np.asarray(y, dtype=np.float64)
    series = pd.Series(y)
    baseline = series.rolling(window=window, center=True, min_periods=min_periods).median()

    baseline_arr = baseline.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        hi = np.maximum(y, baseline_arr)
        lo = np.minimum(y, baseline_arr)
        deviation_ratio = np.where(lo > 0, hi / lo, np.where(hi > 0, np.inf, 1.0))

    has_baseline = ~np.isnan(baseline_arr)
    flagged = has_baseline & (deviation_ratio > deviation_ratio_threshold)

    corrected = y.copy()
    corrected[flagged] = y[flagged] / deviation_ratio[flagged]

    result = {
        "flagged": flagged,
        "baseline": baseline_arr,
        "deviation_ratio": deviation_ratio,
        "corrected": corrected,
    }

    if recurrence_period is not None:
        n = y.shape[0]
        phase = np.arange(n) % recurrence_period
        phase_flag_count = np.zeros(n, dtype=np.int64)
        counts_per_phase = np.bincount(phase[flagged], minlength=recurrence_period)
        phase_flag_count = counts_per_phase[phase]

        recurring = flagged & (phase_flag_count >= recurrence_min_occurrences)
        rare = flagged & ~recurring

        result["recurring"] = recurring
        result["rare"] = rare
        result["phase_flag_count"] = phase_flag_count

    return result


def apply_calendar_anomaly_flag(y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: returns ``(binary_flag, corrected_series)`` for direct use as model inputs."""
    result = detect_calendar_anomalies(y, **kwargs)
    return result["flagged"].astype(np.int64), result["corrected"]


__all__ = ["detect_calendar_anomalies", "apply_calendar_anomaly_flag"]
