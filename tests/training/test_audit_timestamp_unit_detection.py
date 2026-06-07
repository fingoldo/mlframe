"""Sensor tests for the audit timestamp-unit auto-detector
(``_coerce_timestamps_for_audit``).

Fixed bug: pandas ``pd.to_datetime(int64_arr)`` without ``unit=`` interprets
values as NANOSECONDS since the unix epoch by default. A typical "epoch seconds"
input (values around 1.7e9 for late-2023 data) then collapses every row into the
first ~2 seconds after epoch, producing a single-bin audit with no change-point
coverage. The auto-detector now picks the COARSEST unit that lands every
timestamp inside [1970-01-01, 2200-01-01].
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.targets.target_temporal_audit import (
    coerce_timestamps_for_audit as _coerce_timestamps_for_audit,
    _pick_granularity,
)


def test_int64_epoch_seconds_auto_detected_as_s():
    """1.7e9 must read as 2023-11-14 (s), not 1970-01-01.0000000017 (ns)."""
    n = 1000
    ts = (1_700_000_000 + np.arange(n, dtype=np.int64))  # epoch seconds, Nov 2023
    out = _coerce_timestamps_for_audit(ts)
    out_pd = pd.to_datetime(out)
    assert out_pd[0].year == 2023, f"expected 2023, got {out_pd[0]}"
    # Span should be ~1000 seconds = ~17 minutes, NOT ~1000 ns
    span_seconds = (out_pd[-1] - out_pd[0]).total_seconds()
    assert 900 < span_seconds < 1100, f"span={span_seconds}s, expected ~1000"


def test_int64_epoch_milliseconds_auto_detected_as_ms():
    """1.7e12 must read as 2023-11-14 (ms)."""
    n = 500
    ts = (1_700_000_000_000 + np.arange(n, dtype=np.int64) * 1000)  # epoch ms
    out = _coerce_timestamps_for_audit(ts)
    out_pd = pd.to_datetime(out)
    assert out_pd[0].year == 2023
    span_seconds = (out_pd[-1] - out_pd[0]).total_seconds()
    assert 400 < span_seconds < 600


def test_int64_epoch_microseconds_auto_detected_as_us():
    """1.7e15 must read as 2023-11-14 (us)."""
    n = 500
    ts = (1_700_000_000_000_000 + np.arange(n, dtype=np.int64) * 1_000_000)  # epoch us
    out = _coerce_timestamps_for_audit(ts)
    out_pd = pd.to_datetime(out)
    assert out_pd[0].year == 2023
    span_seconds = (out_pd[-1] - out_pd[0]).total_seconds()
    assert 400 < span_seconds < 600


def test_int64_epoch_nanoseconds_auto_detected_as_ns():
    """1.7e18 must read as 2023-11-14 (ns) -- already in ns, no shift needed."""
    n = 500
    ts = (1_700_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 1_000_000_000)  # epoch ns
    out = _coerce_timestamps_for_audit(ts)
    out_pd = pd.to_datetime(out)
    assert out_pd[0].year == 2023
    span_seconds = (out_pd[-1] - out_pd[0]).total_seconds()
    assert 400 < span_seconds < 600


def test_datetime64_passthrough_no_conversion():
    """datetime64 input must round-trip unchanged."""
    ts = pd.date_range("2023-11-14", periods=100, freq="h").to_numpy()
    out = _coerce_timestamps_for_audit(ts)
    np.testing.assert_array_equal(out, ts)


def test_float64_epoch_seconds_with_fractional_sub_seconds():
    """Float epoch-seconds (e.g. high-resolution tick data) should auto-detect as s."""
    n = 1000
    ts = (1_700_000_000.0 + np.arange(n) * 0.5).astype(np.float64)  # 0.5s spacing
    out = _coerce_timestamps_for_audit(ts)
    out_pd = pd.to_datetime(out)
    assert out_pd[0].year == 2023
    span_seconds = (out_pd[-1] - out_pd[0]).total_seconds()
    assert 480 < span_seconds < 520  # ~500s span


def test_explicit_unit_override_forces_interpretation():
    """When caller passes explicit_unit='ns', the auto-detector is bypassed."""
    ts = (1_700_000_000 + np.arange(100, dtype=np.int64))  # would auto-detect as 's'
    out_default = _coerce_timestamps_for_audit(ts)
    out_forced = _coerce_timestamps_for_audit(ts, explicit_unit="ns")
    out_default_pd = pd.to_datetime(out_default)
    out_forced_pd = pd.to_datetime(out_forced)
    assert out_default_pd[0].year == 2023
    assert out_forced_pd[0].year == 1970


def test_explicit_unit_invalid_raises():
    ts = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="audit timestamp unit"):
        _coerce_timestamps_for_audit(ts, explicit_unit="bogus")


def test_yyyymmdd_integers_get_seconds_interpretation():
    """YYYYMMDD-style ints (e.g. 20231114) DON'T match any plausible epoch unit
    but DO fit the s-range numerically (2e7 < 7.26e9), so detector picks 's'.
    Resulting date is 1970-08-22 which is wrong but the detector did the best
    it could; only an explicit unit can salvage this case."""
    ts = np.array([20231114, 20231115, 20231116], dtype=np.int64)
    out = _coerce_timestamps_for_audit(ts)
    out_pd = pd.to_datetime(out)
    assert out_pd[0].year == 1970  # coerced via 's' interpretation


def test_pick_granularity_int64_epoch_seconds_picks_meaningful_bin():
    """Regression: _pick_granularity used to call bare pd.to_datetime(pd.Series(ts))
    on int64 epoch-seconds, which collapsed the entire span to ~2 nanoseconds and
    returned 'month' as the fallback. With the lib-level coerce helper threaded in,
    a 25k-row span of ~7 hours now picks a sub-day granularity (minute/hour)."""
    n = 25_000
    ts = (1_700_000_000 + np.arange(n, dtype=np.int64))  # epoch seconds, ~7 hours span
    gran = _pick_granularity(ts)
    # 25k seconds = ~6.9 hours; sub-day granularity expected (minute, hour).
    # Old broken behaviour: span collapsed to ns scale -> "month" fallback.
    assert gran in ("minute", "hour"), f"expected sub-day granularity, got {gran!r}"


def test_pick_granularity_int64_epoch_seconds_long_span_picks_day_or_week():
    """A 1-year epoch-seconds span (n=365*24*3600 sec) should pick day/week, not 'month' fallback."""
    span_seconds = 365 * 86_400  # ~1 year
    ts = (1_700_000_000 + np.linspace(0, span_seconds, num=1000, dtype=np.int64))
    gran = _pick_granularity(ts)
    assert gran in ("day", "week", "month"), f"got {gran!r}"


def test_truly_out_of_range_negative_falls_back_with_warning(caplog):
    """Negative pre-1970 epoch values fall outside [1970, 2200] under every
    candidate unit; detector logs warning and falls back to ns (preserving
    pre-fix behaviour, which produces a clamped datetime64 result)."""
    # All-negative values: every unit yields a datetime < 1970, so the bounds check fails.
    ts = np.array([-1_000_000_000, -500_000_000, -100_000_000], dtype=np.int64)
    import logging
    with caplog.at_level(logging.WARNING):
        _ = _coerce_timestamps_for_audit(ts)
    assert any(
        "fall outside" in (rec.getMessage() if hasattr(rec, "getMessage") else str(rec.message))
        for rec in caplog.records
    ), f"expected 'fall outside' warning; got records: {[rec.message for rec in caplog.records]}"


def test_max_int64_out_of_range_falls_back_with_warning(caplog):
    """Values above 7.26e18 (the ns upper bound) fall outside every unit's range."""
    ts = np.array([8_000_000_000_000_000_000, 9_000_000_000_000_000_000], dtype=np.int64)
    import logging
    with caplog.at_level(logging.WARNING):
        _ = _coerce_timestamps_for_audit(ts)
    assert any(
        "fall outside" in (rec.getMessage() if hasattr(rec, "getMessage") else str(rec.message))
        for rec in caplog.records
    )
