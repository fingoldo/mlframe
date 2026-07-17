"""Biz-value and unit tests for mlframe.training.composite.discovery.auto_detect.

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

from mlframe.training.composite.discovery.auto_detect import (
    detect_cat_columns,
    detect_group_column_candidates,
    detect_time_column_candidates,
    sort_df_by_time_column,
)


# ----------------------------------------------------------------------------
# detect_time_column_candidates
# ----------------------------------------------------------------------------


def test_time_detect_picks_datetime_column():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=100, freq="D"),
            "noise": np.random.default_rng(0).normal(size=100),
        }
    )
    out = detect_time_column_candidates(df)
    assert any(name == "ts" for name, _ in out), f"datetime col must be detected; got {[n for n, _ in out]}"
    # Datetime should be highest-scored
    top_name, top_info = out[0]
    assert top_name == "ts"
    assert top_info["is_datetime"] is True
    assert top_info["score"] >= 100.0, "datetime score must equal the 100.0 floor"


def test_time_detect_picks_monotonic_numeric():
    df = pd.DataFrame(
        {
            "row_id": np.arange(50, dtype=np.int64),  # strictly monotonic asc
            "noise": np.random.default_rng(1).normal(size=50),
        }
    )
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
    df = pd.DataFrame(
        {
            "well_id": group_ids,
            "feature_x": rng.normal(size=n_rows),
        }
    )
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
    # 5 groups, but group 0 has only 5 rows (< 20 = 1% * 2000)
    ids = np.concatenate(
        [
            np.full(5, 0, dtype=np.int64),
            np.full(500, 1, dtype=np.int64),
            np.full(500, 2, dtype=np.int64),
            np.full(495, 3, dtype=np.int64),
            np.full(500, 4, dtype=np.int64),
        ]
    )
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
    assert "gid" in pd_names and "gid" in pl_names, f"polars/pandas with explicit candidate list must both pick gid; pd={pd_names}, pl={pl_names}"


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


# ----------------------------------------------------------------------------
# A24 (2026-06-11): info_bonus must be UNIMODAL, not monotone-increasing.
# The pre-fix bonus n_unique/(log1p(n_unique)+1) rose without bound, so an
# ID-like high-cardinality int column outranked a clean moderate-cardinality
# categorical whenever coverage was comparable -- "ID-like int columns rank
# first", the exact defect A24 reports.
# ----------------------------------------------------------------------------


def _build_equal_coverage_cat_columns(n_total: int = 200_000):
    """Build two int columns with IDENTICAL top-10 coverage but different
    cardinality: a clean 25-level categorical and an ID-like 400-level column.

    Top-10 levels carry 60% of rows in BOTH columns, so ``coverage_top10`` is
    equal and the detector's ranking is decided ENTIRELY by ``info_bonus``.
    Every level holds >= the default 20 samples, so both clear the gates.
    """

    def realise(n_tail_levels: int, hot_frac: float = 0.60):
        hot_rows = round(n_total * hot_frac)
        tail_rows = n_total - hot_rows
        counts = [hot_rows // 10] * 10 + [tail_rows // n_tail_levels] * n_tail_levels
        return np.repeat(np.arange(len(counts)), counts)

    clean = realise(15)  # 10 hot + 15 tail = 25 levels
    idlike = realise(390)  # 10 hot + 390 tail = 400 levels
    m = min(len(clean), len(idlike))
    return pd.DataFrame(
        {
            "clean_cat": clean[:m].astype(np.int64),
            "id_like": idlike[:m].astype(np.int64),
        }
    )


def test_cat_detect_info_bonus_is_unimodal():
    """The cardinality bonus must peak in the 10-100 'sweet spot' and decay
    beyond it. The pre-fix monotone shape fails ``bonus(40) > bonus(500)``."""
    from mlframe.training.composite.discovery.auto_detect import _cat_info_bonus

    peak = 40.0
    # Rising up to the peak (this half the old monotone shape also satisfied).
    assert _cat_info_bonus(5, peak) > _cat_info_bonus(2, peak)
    assert _cat_info_bonus(40, peak) > _cat_info_bonus(5, peak)
    # Decaying past the peak -- the half the OLD monotone bonus violated.
    assert _cat_info_bonus(40, peak) > _cat_info_bonus(500, peak), "info_bonus must decay for ID-like high cardinality (A24)"
    assert _cat_info_bonus(40, peak) > _cat_info_bonus(1000, peak)
    # Unique global maximum at the configured peak.
    grid = [_cat_info_bonus(n, peak) for n in (2, 5, 10, 40, 100, 500, 1000)]
    assert max(grid) == grid[3], "global max must sit at the sweet-spot peak"


def test_cat_detect_clean_categorical_outranks_id_like_at_equal_coverage():
    """biz_value (A24): with top-10 coverage held EQUAL, a clean 25-category
    column MUST outrank an ID-like 400-level column. The pre-fix monotone
    bonus inverted this (id_like score 34.3 vs clean 3.5)."""
    df = _build_equal_coverage_cat_columns()
    out = dict(detect_cat_columns(df, max_unique=1000))
    assert "clean_cat" in out and "id_like" in out
    # Coverage is engineered identical -> isolates the bonus shape.
    assert abs(out["clean_cat"]["coverage_top10"] - out["id_like"]["coverage_top10"]) < 1e-9
    assert out["clean_cat"]["score"] > out["id_like"]["score"], (
        "clean moderate-cardinality cat must outrank ID-like high-cardinality "
        f"at equal coverage; got clean={out['clean_cat']['score']:.4f} "
        f"id_like={out['id_like']['score']:.4f}"
    )


def test_cat_detect_regression_pre_fix_monotone_bonus_inverts_ranking():
    """Regression pin (A24): the OLD monotone bonus n/(log1p(n)+1) -- applied
    to the SAME detected (n_unique, coverage_top10) -- ranks the ID-like column
    FIRST. This test FAILS on the pre-fix logic and documents why the fix is
    needed; the new unimodal bonus reverses the verdict."""
    df = _build_equal_coverage_cat_columns()
    out = dict(detect_cat_columns(df, max_unique=1000))
    clean, idlike = out["clean_cat"], out["id_like"]

    def _old_monotone_bonus(n_unique: int) -> float:
        return float(n_unique) / float(np.log1p(n_unique) + 1.0)

    old_clean = clean["coverage_top10"] * _old_monotone_bonus(clean["n_unique"])
    old_idlike = idlike["coverage_top10"] * _old_monotone_bonus(idlike["n_unique"])
    # The OLD shape would have ranked the ID-like column first (the bug)...
    assert old_idlike > old_clean, "pre-fix bonus must inflate the ID-like column"
    # ...while the SHIPPED detector ranks the clean column first (the fix).
    assert clean["score"] > idlike["score"]


def test_cat_detect_sweet_spot_peak_is_tunable():
    """The peak is configurable: shifting it moves which cardinality scores
    highest, so a dataset with very granular categoricals can be retuned."""
    from mlframe.training.composite.discovery.auto_detect import _cat_info_bonus

    # With a high peak, a 200-level column beats a 20-level column...
    assert _cat_info_bonus(200, 200.0) > _cat_info_bonus(20, 200.0)
    # ...but with the default low peak the 20-level column wins.
    assert _cat_info_bonus(20, 40.0) > _cat_info_bonus(200, 40.0)


# ----------------------------------------------------------------------------
# Regression: an object column holding one ndarray per row (embeddings) must
# be SKIPPED, not crash the whole scan (fuzz c0050/c0058/c0114).
# ----------------------------------------------------------------------------


def _df_with_embedding_column(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "cat_0": rng.integers(0, 5, size=n),
            "num_0": rng.normal(size=n),
            "emb_0": [rng.normal(size=8) for _ in range(n)],
        }
    )


def test_group_detect_skips_unhashable_embedding_column_instead_of_raising():
    """``detect_group_column_candidates`` scans every non-numeric column as a group-key
    candidate; an embedding column (object dtype, one ndarray per row) is not numeric so it
    used to reach ``np.unique`` and raise ``TypeError: unhashable type: 'numpy.ndarray'``,
    aborting the whole scan instead of just excluding that one column."""
    df = _df_with_embedding_column()
    out = detect_group_column_candidates(df)
    assert all(name != "emb_0" for name, _ in out)


def test_cat_detect_skips_unhashable_embedding_column_instead_of_raising():
    """Same failure mode as above for ``detect_cat_columns`` (its symmetric sibling)."""
    df = _df_with_embedding_column()
    out = detect_cat_columns(df, max_unique=1000)
    assert all(name != "emb_0" for name, _ in out)
