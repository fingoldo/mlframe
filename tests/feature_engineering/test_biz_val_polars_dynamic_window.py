"""biz_value test for ``feature_engineering.polars_dynamic_window_aggregate``.

The win: for large panel/time-series data, computing per-entity rolling-window aggregates via polars
``group_by_dynamic`` should be dramatically faster than the equivalent pandas ``groupby().resample()`` while
producing the same window aggregates (verified against a hand-computable small example, since polars and
pandas use slightly different window-boundary conventions and won't produce identical row counts).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_engineering.polars_dynamic_window import polars_dynamic_window_aggregate


def test_polars_dynamic_window_aggregate_matches_hand_computed_windows():
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-08", "2024-01-09"]),
            "x": [10.0, 20.0, 30.0, 100.0, 200.0],
        }
    )
    result = polars_dynamic_window_aggregate(df, "t", ["x"], every="7d", agg_funcs=["mean", "sum", "count"])

    assert len(result) == 2
    first, second = result.iloc[0], result.iloc[1]
    assert first["x_mean"] == 20.0  # rows Jan 1-3: (10+20+30)/3
    assert first["x_sum"] == 60.0
    assert first["x_count"] == 3
    assert second["x_mean"] == 150.0  # rows Jan 8-9: (100+200)/2
    assert second["x_sum"] == 300.0
    assert second["x_count"] == 2


def test_biz_val_polars_dynamic_window_aggregate_beats_pandas_resample_speed():
    rng = np.random.default_rng(0)
    n_entities = 2000
    n_days = 60
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    entity_ids = np.repeat(np.arange(n_entities), n_days)
    t = np.tile(dates, n_entities)
    x = rng.normal(0, 1, n_entities * n_days)
    df = pd.DataFrame({"entity": entity_ids, "t": t, "x": x})

    t0 = time.perf_counter()
    polars_dynamic_window_aggregate(df, "t", ["x"], every="7d", group_col="entity", agg_funcs=["mean"])
    t_polars = time.perf_counter() - t0

    t0 = time.perf_counter()
    df.set_index("t").groupby("entity")["x"].resample("7D").mean()
    t_pandas = time.perf_counter() - t0

    assert t_polars < t_pandas * 0.5, (
        f"polars group_by_dynamic should be substantially faster than pandas groupby().resample() at panel scale: "
        f"polars={t_polars:.4f}s pandas={t_pandas:.4f}s"
    )


def test_polars_dynamic_window_aggregate_per_group_independent_windows():
    df = pd.DataFrame(
        {
            "entity": ["a", "a", "b", "b"],
            "t": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "x": [1.0, 2.0, 100.0, 200.0],
        }
    )
    result = polars_dynamic_window_aggregate(df, "t", ["x"], every="7d", group_col="entity", agg_funcs=["mean"])
    by_entity = {row["entity"]: row["x_mean"] for _, row in result.iterrows()}
    assert by_entity["a"] == 1.5
    assert by_entity["b"] == 150.0
