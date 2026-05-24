"""Biz-value and unit tests for mlframe.training.composite_auto_detect.

Three public detectors:
  - ``detect_time_column_candidates``
  - ``sort_df_by_time_column``
  - ``detect_group_column_candidates``
  - ``detect_cat_columns``

Per the U3 audit finding and project memory ``project_mlframe_int_as_cat_detector``,
the group-detector thresholds (``min_unique=3, max_unique=500``) are calibrated for
``linear_residual_grouped``. The biz_value test below pins those calibration points
so a future tweak that breaks the contract is caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.composite_auto_detect import (
    detect_cat_columns,
    detect_group_column_candidates,
    detect_time_column_candidates,
    sort_df_by_time_column,
)


# ----------------------------------------------------------------------------
# detect_time_column_candidates
# ----------------------------------------------------------------------------


def test_time_detect_picks_datetime_column():
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=100, freq="D"),
        "noise": np.random.default_rng(0).normal(size=100),
    })
    out = detect_time_column_candidates(df)
    assert any(name == "ts" for name, _ in out), \
        f"datetime col must be detected; got {[n for n, _ in out]}"
    # Datetime should be highest-scored
    top_name, top_info = out[0]
    assert top_name == "ts"
    assert top_info["is_datetime"] is True
    assert top_info["score"] >= 100.0, "datetime score must equal the 100.0 floor"


def test_time_detect_picks_monotonic_numeric():
    df = pd.DataFrame({
        "row_id": np.arange(50, dtype=np.int64),  # strictly monotonic asc
        "noise": np.random.default_rng(1).normal(size=50),
    })
    out = detect_time_column_candidates(df)
    names = [n for n, _ in out]
    assert "row_id" in names, f"monotonic numeric col must be detected; got {names}"
    info = dict(out)["row_id"]
    assert info["is_monotonic"] is True
    assert info["monotonic_direction"] == "asc"


def test_time_detect_picks_monotonic_descending():
    df = pd.DataFrame({"row_id": np.arange(50, 0, -1, dtype=np.int64)})
    out = detect_time_column_candidates(df)
    info = dict(out)["row_id"]
    assert info["is_monotonic"] is True
    assert info["monotonic_direction"] == "desc"


def test_time_detect_rejects_non_monotonic_numeric():
    """Random noise must not be detected as a time column."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"noise": rng.normal(size=200)})
    out = detect_time_column_candidates(df)
    assert out == [], f"noise column must not be flagged as time; got {out!r}"


def test_time_detect_polars_parity():
    """Polars and pandas should produce the same set of candidates."""
    n = 30
    pd_df = pd.DataFrame({"row_id": np.arange(n, dtype=np.int64), "x": np.zeros(n)})
    pl_df = pl.DataFrame({"row_id": np.arange(n, dtype=np.int64), "x": np.zeros(n)})
    pd_names = {n for n, _ in detect_time_column_candidates(pd_df)}
    pl_names = {n for n, _ in detect_time_column_candidates(pl_df)}
    assert pd_names == pl_names, f"polars vs pandas mismatch: {pd_names!r} vs {pl_names!r}"


def test_time_detect_unsupported_raises():
    with pytest.raises(TypeError, match="unsupported df type"):
        detect_time_column_candidates([1, 2, 3])


def test_sort_df_by_time_pandas():
    df = pd.DataFrame({"ts": [3, 1, 2], "v": [30, 10, 20]})
    sorted_df = sort_df_by_time_column(df, "ts")
    assert sorted_df["ts"].tolist() == [1, 2, 3]
    assert sorted_df["v"].tolist() == [10, 20, 30]


def test_sort_df_by_time_polars():
    df = pl.DataFrame({"ts": [3, 1, 2], "v": [30, 10, 20]})
    sorted_df = sort_df_by_time_column(df, "ts")
    assert sorted_df["ts"].to_list() == [1, 2, 3]
    assert sorted_df["v"].to_list() == [10, 20, 30]


def test_sort_df_by_time_unsupported_raises():
    with pytest.raises(TypeError, match="unsupported df type"):
        sort_df_by_time_column({"ts": [1]}, "ts")


# ----------------------------------------------------------------------------
# detect_group_column_candidates — biz_value calibrated to memory thresholds
# ----------------------------------------------------------------------------


def test_group_detect_picks_uniform_group_id():
    """biz_value: a column with 100 unique IDs evenly distributed across N rows MUST be
    picked. This is the canonical ``linear_residual_grouped`` setup."""
    n_rows = 2000
    n_groups = 100
    rng = np.random.default_rng(42)
    # Each group gets exactly 20 rows (uniform, satisfies min_size_ratio=0.01 default)
    group_ids = np.tile(np.arange(n_groups, dtype=np.int64), n_rows // n_groups)
    rng.shuffle(group_ids)
    df = pd.DataFrame({
        "well_id": group_ids,
        "feature_x": rng.normal(size=n_rows),
    })
    out = detect_group_column_candidates(df)
    names = [n for n, _ in out]
    assert "well_id" in names, f"uniform 100-group col must be flagged; got {names}"
    info = dict(out)["well_id"]
    assert info["n_unique"] == 100
    assert info["min_group_size"] == 20  # exactly even split
    assert info["max_group_size"] == 20


def test_group_detect_rejects_too_few_groups():
    """min_unique=3 default: a 2-group column must be REJECTED."""
    n_rows = 500
    df = pd.DataFrame({"binary_flag": np.tile([0, 1], n_rows // 2).astype(np.int64)})
    out = detect_group_column_candidates(df)
    names = [n for n, _ in out]
    assert "binary_flag" not in names, f"binary col should be rejected (< min_unique=3); got {names}"


def test_group_detect_rejects_too_many_groups():
    """max_unique=500 default: a 600-group column must be REJECTED."""
    n_rows = 6000
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 600, size=n_rows, dtype=np.int64)
    df = pd.DataFrame({"too_many": ids})
    out = detect_group_column_candidates(df)
    names = [n for n, _ in out]
    assert "too_many" not in names, f"600-group col should be rejected (> max_unique=500); got {names}"


def test_group_detect_rejects_tiny_smallest_group():
    """min_size_ratio=0.01: smallest group must hold >= 1% of rows."""
    n_rows = 2000
    # 5 groups, but group 0 has only 5 rows (< 20 = 1% * 2000)
    ids = np.concatenate([
        np.full(5, 0, dtype=np.int64),
        np.full(500, 1, dtype=np.int64),
        np.full(500, 2, dtype=np.int64),
        np.full(495, 3, dtype=np.int64),
        np.full(500, 4, dtype=np.int64),
    ])
    df = pd.DataFrame({"unbalanced": ids})
    out = detect_group_column_candidates(df)
    names = [n for n, _ in out]
    assert "unbalanced" not in names, f"col with tiny minority group must be rejected; got {names}"


def test_group_detect_empty_df():
    df = pd.DataFrame({"col": pd.array([], dtype="Int64")})
    out = detect_group_column_candidates(df)
    assert out == [], f"empty df must return []; got {out!r}"


def test_group_detect_all_null_column():
    n = 100
    df = pd.DataFrame({"all_null": pd.array([None] * n, dtype="Int64")})
    out = detect_group_column_candidates(df)
    names = [n for n, _ in out]
    assert "all_null" not in names, "all-null col cannot be a group key"


def test_group_detect_single_value_column():
    n = 100
    df = pd.DataFrame({"const": np.zeros(n, dtype=np.int64)})
    out = detect_group_column_candidates(df)
    names = [n for n, _ in out]
    assert "const" not in names, "constant col cannot be a group key (n_unique=1 < min_unique=3)"


def test_group_detect_polars_pandas_parity_explicit_candidates():
    """With ``candidate_columns`` passed explicitly (bypassing the default-column-pick
    heuristics that differ between pandas and polars per memory ``int_as_cat_detector``),
    the detector must give the same answer on the same gid column."""
    rng = np.random.default_rng(7)
    group_ids = np.tile(np.arange(50, dtype=np.int64), 30)
    rng.shuffle(group_ids)
    pd_df = pd.DataFrame({"gid": group_ids, "x": rng.normal(size=len(group_ids))})
    pl_df = pl.DataFrame({"gid": group_ids, "x": rng.normal(size=len(group_ids))})
    pd_names = {n for n, _ in detect_group_column_candidates(pd_df, candidate_columns=["gid"])}
    pl_names = {n for n, _ in detect_group_column_candidates(pl_df, candidate_columns=["gid"])}
    assert "gid" in pd_names and "gid" in pl_names, \
        f"polars/pandas with explicit candidate list must both pick gid; pd={pd_names}, pl={pl_names}"


def test_group_detect_unsupported_raises():
    with pytest.raises(TypeError, match="unsupported df type"):
        detect_group_column_candidates({"gid": [1, 2, 3]})


# ----------------------------------------------------------------------------
# detect_cat_columns — symmetric sibling for FHC target_mean / WoE
# ----------------------------------------------------------------------------


def test_cat_detect_picks_balanced_categorical():
    """A categorical column with enough samples per category must be picked."""
    rng = np.random.default_rng(3)
    cats = rng.choice(["A", "B", "C", "D"], size=500)
    df = pd.DataFrame({"category": cats, "x": rng.normal(size=500)})
    out = detect_cat_columns(df)
    names = [n for n, _ in out]
    assert "category" in names, f"balanced 4-cat string col must be flagged; got {names}"


def test_cat_detect_rejects_rare_categories():
    """min_samples_per_cat=20 (default): a category with only 5 samples must reject the col."""
    n = 1000
    cats = np.array(["A"] * 495 + ["B"] * 495 + ["RARE"] * 10)
    np.random.default_rng(0).shuffle(cats)
    df = pd.DataFrame({"feat": cats})
    out = detect_cat_columns(df)
    names = [n for n, _ in out]
    assert "feat" not in names, f"col with <20-sample rare category must be rejected; got {names}"


def test_cat_detect_binary_indicator_passes():
    """min_unique=2 (default) — binary indicator IS a valid categorical."""
    n = 500
    df = pd.DataFrame({"flag": np.tile([0, 1], n // 2).astype(np.int64)})
    out = detect_cat_columns(df)
    names = [n for n, _ in out]
    assert "flag" in names, f"binary indicator should be picked; got {names}"


def test_cat_detect_empty_df():
    df = pd.DataFrame({"col": pd.array([], dtype="string")})
    out = detect_cat_columns(df)
    assert out == [], f"empty df must return []; got {out!r}"


def test_cat_detect_unsupported_raises():
    with pytest.raises(TypeError, match="unsupported df type"):
        detect_cat_columns(42)
