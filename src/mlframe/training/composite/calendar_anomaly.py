"""Automated calendar-anomaly (holiday/event spike) detection for a datetime-indexed target.

Manually spotting and correcting anomalous calendar days (a sales team's found ~70x median spikes on Dec
24-25, another independently added Spanish holiday flags) doesn't scale and misses days nobody thought to
check. This scans a datetime-indexed target for days whose value deviates far from a robust LOCAL baseline
(rolling median, resistant to the very spikes being detected -- a plain rolling mean would be dragged up by
them), flags candidate anomalous days, and returns both a binary flag and a "divided-out" corrected series.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def detect_calendar_anomalies(
    y: np.ndarray,
    window: int = 14,
    deviation_ratio_threshold: float = 3.0,
    min_periods: int = 5,
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

    Returns
    -------
    dict
        ``flagged`` ``(n,)`` bool, ``baseline`` ``(n,)`` the rolling-median local baseline,
        ``deviation_ratio`` ``(n,)`` float, ``corrected`` ``(n,)`` -- ``y`` with flagged days divided by
        their own deviation ratio (pulling them back toward the local baseline; unflagged days unchanged).
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

    return {
        "flagged": flagged,
        "baseline": baseline_arr,
        "deviation_ratio": deviation_ratio,
        "corrected": corrected,
    }


def apply_calendar_anomaly_flag(y: np.ndarray, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience wrapper: returns ``(binary_flag, corrected_series)`` for direct use as model inputs."""
    result = detect_calendar_anomalies(y, **kwargs)
    return result["flagged"].astype(np.int64), result["corrected"]


__all__ = ["detect_calendar_anomalies", "apply_calendar_anomaly_flag"]
