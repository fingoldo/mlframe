"""Regression: the pandas (ImportError) fallback of the temporal-audit batch must coerce
timestamps via ``_coerce_timestamps_for_audit`` -- same as the polars path.

Pre-fix the fallback built the audit frame from RAW int64 timestamps. pandas reads bare int64
as nanoseconds-since-epoch, so epoch-SECONDS input collapsed every row into the first nanosecond
after epoch -> a one-bin audit with no change-point coverage.
"""
from __future__ import annotations

import sys
import types

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


def test_pandas_fallback_branch_calls_coercion(monkeypatch):
    """Drive the ImportError (no-polars) fallback and assert it routed the raw timestamps through
    _coerce_timestamps_for_audit -- the polars import is forced to fail and the coercion helper is
    replaced by a spy that records its input. Pre-fix the fallback built the audit frame from the
    raw int64 array, so the spy would never see them."""
    # Force `import polars` inside the function to raise ImportError.
    monkeypatch.setitem(sys.modules, "polars", None)

    seen: dict[str, np.ndarray] = {}

    def _spy(ts_arr, explicit_unit=None):
        seen["ts"] = np.asarray(ts_arr)
        return _coerce_timestamps_for_audit(ts_arr, explicit_unit=explicit_unit)

    monkeypatch.setattr(_ta_mod, "_coerce_timestamps_for_audit", _spy)

    secs = np.arange(0, 5 * 365 * 24 * 3600, 30 * 24 * 3600, dtype=np.int64)
    behavior_config = types.SimpleNamespace(target_temporal_audit_column="ts")
    fte = types.SimpleNamespace(ts_field="ts")
    target_by_type = {"regression": {"y": np.linspace(0.0, 1.0, len(secs))}}

    _ta_mod.run_temporal_audit_batch(
        behavior_config=behavior_config,
        features_and_targets_extractor=fte,
        timestamps=secs,
        target_by_type=target_by_type,
        verbose=False,
    )

    assert "ts" in seen, "pandas fallback must route timestamps through _coerce_timestamps_for_audit"
    np.testing.assert_array_equal(seen["ts"], secs)
