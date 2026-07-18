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
    """Biz val detect calendar anomalies flags holiday spike with no false positives."""
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
    """Apply calendar anomaly flag returns binary flag and corrected series."""
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
    """Detect calendar anomalies insufficient history not flagged."""
    y = np.array([100.0, 100.0])
    result = detect_calendar_anomalies(y, window=14, min_periods=5)
    assert not result["flagged"].any()


def test_detect_calendar_anomalies_recurrence_period_none_is_bit_identical_to_prior_default():
    """Default behavior (recurrence_period=None) must be unchanged: no new keys, identical values."""
    rng = np.random.default_rng(0)
    n = 365
    y = 100.0 + rng.normal(0, 5, n)
    y[358] = 100.0 * 70

    baseline = detect_calendar_anomalies(y, window=14, deviation_ratio_threshold=3.0)
    extended = detect_calendar_anomalies(y, window=14, deviation_ratio_threshold=3.0, recurrence_period=None)

    assert set(extended.keys()) == set(baseline.keys()) == {"flagged", "baseline", "deviation_ratio", "corrected"}
    for key in baseline:
        np.testing.assert_array_equal(baseline[key], extended[key])


def test_biz_val_detect_calendar_anomalies_recurrence_separates_weekly_pattern_from_one_off_holidays():
    """Synthetic: a genuine weekly (every-Sunday) sales peak plus a few true one-off holiday spikes.

    A single deviation threshold flags both categories identically. The recurrence classifier must put the
    weekly-peak days into 'recurring' and the true one-off holidays into 'rare', with zero cross-contamination.
    """
    rng = np.random.default_rng(2)
    n = 365 * 2  # two years of daily data, so weekly peaks recur ~104 times
    typical = 100.0
    y = typical + rng.normal(0, 5, n)

    # Genuine recurring pattern: every 7th day (e.g. Sunday) sees a 4x peak.
    recurring_days = np.arange(0, n, 7)
    y[recurring_days] = typical * 4

    # True one-off events: a handful of isolated holiday spikes at non-weekly-aligned, well-separated days.
    one_off_days = [40, 150, 260, 500]
    for d in one_off_days:
        y[d] = typical * 70

    result = detect_calendar_anomalies(
        y,
        window=14,
        deviation_ratio_threshold=3.0,
        recurrence_period=7,
        recurrence_min_occurrences=3,
    )

    recurring_flagged = set(np.flatnonzero(result["recurring"]).tolist())
    rare_flagged = set(np.flatnonzero(result["rare"]).tolist())

    # No cross-contamination between the two categories.
    assert recurring_flagged.isdisjoint(rare_flagged)

    # The true one-offs must land in 'rare', not 'recurring'.
    assert set(one_off_days) <= rare_flagged
    assert set(one_off_days).isdisjoint(recurring_flagged)

    # The weekly-peak days must overwhelmingly land in 'recurring', not 'rare'.
    recurring_days_set = set(recurring_days.tolist())
    correctly_recurring = len(recurring_days_set & recurring_flagged)
    assert (
        correctly_recurring >= len(recurring_days_set) * 0.95
    ), f"expected >=95% of the {len(recurring_days_set)} genuine weekly-peak days classified 'recurring', got {correctly_recurring}"
    assert len(recurring_days_set & rare_flagged) == 0
