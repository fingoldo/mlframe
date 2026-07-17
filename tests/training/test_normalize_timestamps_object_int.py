"""Sensor for dummy_baselines._normalize_timestamps: object-dtype arrays
holding bare int epoch-seconds must NOT collapse to nanosecond interpretation.

When a pandas DataFrame has a mixed-dtype timestamp column (e.g. pd.Timestamp
instances inadvertently downcast to object alongside ints from a JSON parse),
``df['ts'].values`` returns an object ndarray. _normalize_timestamps's old
branch did ``pd.to_datetime(obj_arr, utc=True, errors='coerce')`` which
interpreted ints as nanoseconds, collapsing real seconds-since-epoch values to
1970-01-01 + a few microseconds.

Fix: sniff the first non-null element; if numeric int, route through
target_temporal_audit.coerce_timestamps_for_audit (auto-unit detect).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.training.baselines.dummy import _normalize_timestamps


def test_object_array_of_int_epoch_seconds_detected_correctly():
    """object dtype + int epoch-seconds: must read as 2023, not 1970."""
    ts_obj = np.array([1_700_000_000, 1_700_000_001, 1_700_000_002], dtype=object)
    out = _normalize_timestamps(ts_obj)
    assert out is not None
    out_pd = pd.to_datetime(out).year
    assert (out_pd == 2023).all() or (out_pd == 2023).any()


def test_object_array_of_pd_timestamps_still_parsed():
    """The pd.Timestamp branch must still work after the new int branch was added."""
    ts_obj = np.array(
        [
            pd.Timestamp("2023-11-14"),
            pd.Timestamp("2023-11-15"),
            pd.Timestamp("2023-11-16"),
        ],
        dtype=object,
    )
    out = _normalize_timestamps(ts_obj)
    assert out is not None
    out_pd = pd.to_datetime(out).year
    assert (out_pd == 2023).all()


def test_pure_int64_array_path_unchanged():
    """Non-object int64 input continues to fall through the "already numeric" branch
    (no audit coercion applied; downstream diff arithmetic uses raw ints)."""
    ts = np.array([1_700_000_000, 1_700_000_001], dtype=np.int64)
    out = _normalize_timestamps(ts)
    np.testing.assert_array_equal(out, ts)


def test_datetime64_input_unchanged():
    """datetime64[ns] input passes through the existing 'M' branch."""
    ts = pd.date_range("2023-11-14", periods=5, freq="D").to_numpy()
    out = _normalize_timestamps(ts)
    # Output is int64 ns since epoch
    assert out.dtype == np.int64
    # First value: 2023-11-14 in ns since epoch
    first_ns = pd.Timestamp("2023-11-14").value
    assert out[0] == first_ns
