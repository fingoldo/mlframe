"""Sensor: target_temporal_audit carve preserves identity + facade under budget."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mlframe.training.targets import target_temporal_audit as parent
from mlframe.training.targets import _target_temporal_audit_aggregate as agg
from mlframe.training.targets import _target_temporal_audit_coerce as coerce


def test_w12b_target_temporal_audit_identity_preserved():
    """W12b target temporal audit identity preserved."""
    assert parent.coerce_timestamps_for_audit is coerce.coerce_timestamps_for_audit
    assert parent._pick_granularity is agg._pick_granularity
    assert parent._aggregate_by_time_polars is agg._aggregate_by_time_polars
    assert parent._aggregate_by_time_polars_multi is agg._aggregate_by_time_polars_multi
    assert parent._aggregate_by_time_pandas is agg._aggregate_by_time_pandas
    assert parent._format_bin_label is agg._format_bin_label
    assert parent.Granularity is agg.Granularity


def test_w12b_target_temporal_audit_facade_under_budget():
    """W12b target temporal audit facade under budget."""
    facade_loc = sum(1 for _ in Path(parent.__file__).open(encoding="utf-8"))
    assert facade_loc < 750, f"target_temporal_audit.py LOC={facade_loc} exceeds 750 budget"


def test_w12b_target_temporal_audit_smoke_runs():
    """W12b target temporal audit smoke runs."""
    ts = pd.date_range("2020-01-01", periods=300, freq="D")
    y = (np.arange(300) > 150).astype(int)
    df = pd.DataFrame({"ts": ts, "y": y})
    result = parent.audit_target_over_time(
        df,
        "ts",
        "y",
        target_type="binary_classification",
        granularity="month",
        method="zscore",
    )
    assert result.granularity == "month"
    assert len(result.bins) > 0
    assert isinstance(result, parent.TemporalAuditResult)


def test_w12b_target_temporal_audit_coerce_numeric_to_ns():
    """W12b target temporal audit coerce numeric to ns."""
    arr = np.array([1700000000.0, 1701000000.0, 1702000000.0])
    out = parent.coerce_timestamps_for_audit(arr)
    assert out.dtype == np.dtype("datetime64[ns]")
    assert out.shape == (3,)
