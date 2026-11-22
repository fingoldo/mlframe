"""Audit 2026-11-20 finding 32-F10: integer epoch-second timestamps are not read as nanoseconds.

Daily epoch seconds used to be classified sub-hourly, so the weekly period was never proposed. The test feeds daily integer epoch seconds
and checks the step classification is daily.
"""

import numpy as np

from mlframe.training.baselines._dummy_timeseries import _infer_ts_step_periods, _normalize_timestamps


def test_daily_epoch_seconds_are_classified_daily():
    """Integer timestamps 86400 s apart infer a daily step, not sub-hourly."""
    ts = np.arange(40, dtype=np.int64) * 86_400 + 1_700_000_000
    step, _periods = _infer_ts_step_periods(_normalize_timestamps(ts))
    assert step == "daily"
