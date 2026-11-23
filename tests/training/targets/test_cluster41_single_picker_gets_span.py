"""Cluster 41 regression: the single-target granularity picker receives (min, max), not a per-row list (audit 2026-11-20, finding 41-F10)."""

import polars as pl

from mlframe.training.targets import target_temporal_audit as tta


def test_single_path_passes_span_tuple_to_picker(monkeypatch):
    """The auto-picker gets a 2-tuple of endpoints so a large polars frame is not turned into Python datetimes (41-F10)."""
    seen = {}

    def _spy(ts, *a, **k):
        """Record the argument shape and return a fixed granularity."""
        seen["arg"] = ts
        return "day"

    monkeypatch.setattr(tta, "_pick_granularity", _spy)
    df = pl.DataFrame({
        "ts": pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 3, 1), interval="1d", eager=True),
    })
    df = df.with_columns(y=pl.lit(1.0))
    tta.audit_target_over_time(df, timestamp_col="ts", target_col="y", target_type="regression", granularity="auto")
    assert isinstance(seen["arg"], tuple) and len(seen["arg"]) == 2
