"""Regression: the pandas (ImportError) fallback of the temporal-audit batch must coerce
timestamps via ``_coerce_timestamps_for_audit`` -- same as the polars path.

Pre-fix the fallback built the audit frame from RAW int64 timestamps. pandas reads bare int64
as nanoseconds-since-epoch, so epoch-SECONDS input collapsed every row into the first nanosecond
after epoch -> a one-bin audit with no change-point coverage.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from mlframe.training.core._phase_temporal_audit import _coerce_timestamps_for_audit
import mlframe.training.core._phase_temporal_audit as _ta_mod


def test_raw_int64_seconds_collapse_but_coerced_span_years():
    # 5 years of monthly epoch-second timestamps.
    secs = np.arange(0, 5 * 365 * 24 * 3600, 30 * 24 * 3600, dtype=np.int64)
    assert len(secs) > 50

    raw = pd.to_datetime(secs)  # pre-fix behaviour: int64 -> nanoseconds
    raw_span_days = (raw.max() - raw.min()).days
    assert raw_span_days == 0, "raw int64-as-ns collapses epoch-seconds to a sub-nanosecond span"

    coerced = pd.to_datetime(_coerce_timestamps_for_audit(secs))
    coerced_span_days = (coerced.max() - coerced.min()).days
    assert coerced_span_days > 365 * 4, "coerced epoch-seconds must span multiple years"


def test_pandas_fallback_branch_calls_coercion():
    """Pin that the ImportError fallback routes through _coerce_timestamps_for_audit."""
    src = inspect.getsource(_ta_mod.run_temporal_audit_batch)
    fallback = src.split("except ImportError:", 1)[1]
    assert "_coerce_timestamps_for_audit" in fallback
