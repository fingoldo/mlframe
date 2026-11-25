"""Cluster 41 regression: the tz-stripping warning fires only for tz-aware input (audit 2026-11-20, finding 41-F30).

``pd.to_datetime(..., utc=True)`` makes every result tz-aware, so the warning used to fire for naive timestamps too. Detection now happens on a
plain parse, so naive input is silent and tz-aware input still warns and comes back as datetime64[ns].
"""

import logging

import numpy as np
import pandas as pd

from mlframe.training.targets._target_temporal_audit_coerce import coerce_timestamps_for_audit

_TZ_MSG = "contained tz-aware Timestamps"


def test_naive_timestamps_do_not_warn(caplog):
    """Naive input produces datetime64[ns] with no tz warning (41-F30)."""
    arr = pd.to_datetime(["2020-01-01", "2020-01-02"])
    with caplog.at_level(logging.WARNING):
        out = coerce_timestamps_for_audit(arr)
    assert out.dtype == np.dtype("datetime64[ns]")
    assert not any(_TZ_MSG in r.getMessage() for r in caplog.records)


def test_tz_aware_timestamps_warn_and_strip(caplog):
    """tz-aware input warns, is converted to UTC and returned as naive datetime64[ns] (41-F30)."""
    arr = pd.to_datetime(["2020-01-01T02:00:00+02:00", "2020-01-02T00:00:00+00:00"])
    with caplog.at_level(logging.WARNING):
        out = coerce_timestamps_for_audit(arr)
    assert out.dtype == np.dtype("datetime64[ns]")
    assert out[0] == np.datetime64("2020-01-01T00:00:00")
    assert any(_TZ_MSG in r.getMessage() for r in caplog.records)
