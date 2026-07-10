"""biz_value test for ``training.composite.detect_calendar_anomalies`` / ``apply_calendar_anomaly_flag``.

The win: reproduces the source scenario (a holiday spike ~70x the typical daily level) and confirms the
anomalous days are correctly flagged with zero false positives on normal days, and that the corrected series'
global mean is essentially the true typical level while the raw series' mean is badly skewed by the spike --
the concrete downstream consequence of not correcting for calendar anomalies before computing summary
statistics or fitting a baseline model.
"""
from __future__ import annotations

import numpy as np

from mlframe.training.composite.calendar_anomaly import apply_calendar_anomaly_flag, detect_calendar_anomalies


def test_biz_val_detect_calendar_anomalies_flags_holiday_spike_with_no_false_positives():
    rng = np.random.default_rng(0)
    n = 365
    typical = 100.0
    y = typical + rng.normal(0, 5, n)
    spike_days = [358, 359]  # e.g. Dec 24-25
    for d in spike_days:
        y[d] = typical * 70

    result = detect_calendar_anomalies(y, window=14, deviation_ratio_threshold=3.0)
    flagged_days = np.flatnonzero(result["flagged"])

    assert set(flagged_days.tolist()) == set(spike_days)

    raw_mean_error = abs(float(y.mean()) - typical)
    corrected_mean_error = abs(float(result["corrected"].mean()) - typical)
    assert corrected_mean_error < raw_mean_error * 0.1, (
        f"the corrected series' global mean should be far closer to the true typical level than the raw series: "
        f"corrected_error={corrected_mean_error:.4f} raw_error={raw_mean_error:.4f}"
    )


def test_apply_calendar_anomaly_flag_returns_binary_flag_and_corrected_series():
    rng = np.random.default_rng(1)
    n = 100
    y = 50.0 + rng.normal(0, 2, n)
    y[50] = 500.0

    flag, corrected = apply_calendar_anomaly_flag(y, window=14, deviation_ratio_threshold=3.0)
    assert flag.dtype == np.int64
    assert flag[50] == 1
    assert flag.sum() == 1
    assert corrected[50] < y[50]


def test_detect_calendar_anomalies_insufficient_history_not_flagged():
    y = np.array([100.0, 100.0])
    result = detect_calendar_anomalies(y, window=14, min_periods=5)
    assert not result["flagged"].any()
