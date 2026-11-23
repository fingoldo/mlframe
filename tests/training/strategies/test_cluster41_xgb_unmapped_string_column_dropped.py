"""Cluster 41 regression: a string column missing from the XGBoost category map is dropped, not re-encoded per frame (audit 2026-11-20, finding 41-F21)."""

import polars as pl

from mlframe.training.strategies.xgboost import XGBoostStrategy


def test_unmapped_string_column_is_dropped_not_given_a_frame_local_enum():
    """A column absent from the map has no train-time codes, so it is dropped rather than encoded per frame (41-F21)."""
    enum_a = pl.Enum(["x", "y"])
    df = pl.DataFrame({"mapped": ["x", "y"], "extra": ["p", "q"]})
    out = XGBoostStrategy().prepare_polars_dataframe(df, ["mapped", "extra"], category_map={"mapped": enum_a})
    assert "extra" not in out.columns
    assert out["mapped"].dtype == enum_a
