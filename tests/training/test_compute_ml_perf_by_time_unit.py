"""Sensor for compute_ml_perf_by_time: int64 epoch-seconds timestamps
must NOT collapse the entire dataset into a single bucket.

Fixed at the same commit wave as the audit timestamp-unit fix (9a1855e + 3f11d93).
Without the coerce_timestamps_for_audit threading, ``pd.to_datetime(int64_arr)``
defaulted to nanosecond interpretation, every row landed in the first ~2 seconds
after 1970-01-01, ``pd.Grouper(freq='D')`` saw all rows in a single day bucket,
and the per-time-bin metric collapsed into a single value -- silently. The output
looked legitimate ("here's the metric over time"), but the time axis was a lie.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.evaluation import compute_ml_perf_by_time


def test_int64_epoch_seconds_produces_multiple_time_bins():
    """A 7-day span of int64 epoch-seconds at 1k rows/day must produce ~7 bins,
    not 1. Without the fix, all rows would collapse into a single bucket at 1970-01-01.
    """
    n_per_day = 1000
    n_days = 7
    rng = np.random.default_rng(42)
    # 7 days of epoch-seconds timestamps starting 2023-11-14
    base = 1_700_000_000
    ts = np.concatenate([
        base + day * 86_400 + rng.integers(0, 86_400, n_per_day)
        for day in range(n_days)
    ])
    y_true = rng.integers(0, 2, len(ts))
    y_pred = rng.random(len(ts))

    out = compute_ml_perf_by_time(y_true, y_pred, ts, freq="D", metric="roc_auc", min_samples=100)

    # Pre-fix: all 7000 rows collapsed to 1970-01-01 -> 1 bin.
    # Post-fix: ~7 bins for the 7-day span.
    assert len(out) >= 5, (
        f"expected ~7 daily bins for 7-day span; got {len(out)} bin(s). "
        f"Suggests int64 epoch-seconds collapsed to ns interpretation again."
    )
    # All bins must fall in late-2023, not 1970.
    bin_years = sorted({pd.Timestamp(b).year for b in out.index})
    assert bin_years == [2023] or bin_years == [2023, 2024], (
        f"bins must be in 2023 (or 2023/2024 if span crosses year), got years {bin_years}"
    )


def test_datetime64_input_still_works():
    """Datetime64 input continues to round-trip correctly (no regression on the path
    that was already correct pre-fix)."""
    n = 5000
    ts = pd.date_range("2023-11-14", periods=n, freq="h").to_numpy()
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, n)
    y_pred = rng.random(n)
    out = compute_ml_perf_by_time(y_true, y_pred, ts, freq="D", metric="roc_auc", min_samples=10)
    # 5000 hours = ~208 days; expect ~208 bins.
    assert 150 < len(out) < 250, f"expected ~208 daily bins, got {len(out)}"


def test_int64_epoch_milliseconds_auto_detected():
    """ms-scale epoch input should also auto-detect (1.7e12 range)."""
    n_per_day = 500
    n_days = 5
    rng = np.random.default_rng(42)
    base = 1_700_000_000_000  # epoch ms
    ts = np.concatenate([
        base + day * 86_400_000 + rng.integers(0, 86_400_000, n_per_day)
        for day in range(n_days)
    ])
    y_true = rng.integers(0, 2, len(ts))
    y_pred = rng.random(len(ts))
    out = compute_ml_perf_by_time(y_true, y_pred, ts, freq="D", metric="roc_auc", min_samples=50)
    assert len(out) >= 3, f"expected ~5 daily bins; got {len(out)}"
