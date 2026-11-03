"""Dynamic windows are labelled by their END by default, and numeric time columns need an explicit unit."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("polars")

from mlframe.feature_engineering.polars_dynamic_window import polars_dynamic_window_aggregate


def _events():
    """One value per day for ten days."""
    return pd.DataFrame({"t": pd.date_range("2024-01-01", periods=10, freq="D"), "v": range(10)})


def test_default_label_is_the_window_end_so_a_join_back_sees_only_the_past():
    """3-day windows stepping by one day: the row labelled 2024-01-04 aggregates days 1-3 (values 0, 1, 2), none from the day it is labelled with or later."""
    out = polars_dynamic_window_aggregate(_events(), "t", ["v"], every="1d", period="3d", agg_funcs=("sum",))
    row = out.loc[out["t"] == pd.Timestamp("2024-01-04")]
    assert len(row) == 1 and float(row["v_sum"].iloc[0]) == 0 + 1 + 2


def test_left_label_keeps_the_previous_behaviour():
    """label='left' labels the same window with its start."""
    out = polars_dynamic_window_aggregate(_events(), "t", ["v"], every="1d", period="3d", agg_funcs=("sum",), label="left")
    row = out.loc[out["t"] == pd.Timestamp("2024-01-01")]
    assert float(row["v_sum"].iloc[0]) == 3.0


def test_numeric_time_without_a_unit_is_rejected():
    """An integer day index would be read as microseconds and collapse into one window."""
    df = pd.DataFrame({"t": range(30), "v": range(30)})
    with pytest.raises(ValueError, match="time_unit"):
        polars_dynamic_window_aggregate(df, "t", ["v"], every="7d")


def test_numeric_time_with_a_unit_gives_the_expected_windows():
    """Day offsets 0..29 with time_unit='d' and 10-day windows give three windows of ten rows."""
    df = pd.DataFrame({"t": range(30), "v": 1.0})
    out = polars_dynamic_window_aggregate(df, "t", ["v"], every="10d", agg_funcs=("count",), time_unit="d")
    assert out["v_count"].tolist() == [10, 10, 10]
