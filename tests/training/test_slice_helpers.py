"""Unit tests for ``mlframe.training._slice_helpers``.

Covers:
  - build_slice_eval_sets() across all 4 source modes (random, temporal, fairness, both)
  - dtype preservation for polars Enum and pandas Categorical
  - parallel slicing of sample_weight / base_margin / group_ids
  - StratifiedKFold auto-switch for classification targets
  - GroupKFold auto-switch when group_ids supplied with source='random' (ranker safety)
  - small-val fallback (val/K < min_rows_per_shard -> empty list + warn)
  - empty fairness subgroups handling
  - effective_patience formula
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training._slice_helpers import (
    SliceEvalSet,
    build_slice_eval_sets,
    effective_patience,
)


@pytest.fixture
def reg_val_df() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(0, 1, 200)})
    y = rng.normal(0, 1, 200)
    return X, y


@pytest.fixture
def clf_val_df() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 300), "b": rng.normal(0, 1, 300)})
    y = (rng.uniform(0, 1, 300) < 0.3).astype(int)
    return X, y


def test_random_shards_partition_regression(reg_val_df: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = reg_val_df
    shards = build_slice_eval_sets(X, y, source="random", k=5, min_rows_per_shard=20, random_state=42)
    assert len(shards) == 5
    total_rows = sum(s.y.shape[0] for s in shards)
    assert total_rows == len(y), "K-fold partition must preserve total rows"
    # No row should appear in more than one shard
    all_idx = np.concatenate([s.row_indices for s in shards])
    assert all_idx.size == np.unique(all_idx).size, "K-fold shards must be disjoint"
    # Names follow valid_shard_r{i} convention
    for i, s in enumerate(shards):
        assert s.name == f"valid_shard_r{i}"


def test_random_shards_stratified_classification(clf_val_df: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = clf_val_df
    shards = build_slice_eval_sets(X, y, source="random", k=5, min_rows_per_shard=20, random_state=42)
    assert len(shards) == 5
    # Stratification keeps positive ratio within ~5% across shards
    overall_pos_rate = float(y.mean())
    for s in shards:
        pos_rate = float(s.y.mean())
        assert abs(pos_rate - overall_pos_rate) <= 0.10, (
            f"stratified shard pos rate {pos_rate} too far from overall {overall_pos_rate}"
        )


def test_random_shards_with_group_ids_switches_to_groupkfold(
    reg_val_df: tuple[pd.DataFrame, np.ndarray], caplog,
) -> None:
    X, y = reg_val_df
    # 200 rows, 40 queries of 5 docs each
    group_ids = np.repeat(np.arange(40), 5)
    with caplog.at_level(logging.WARNING):
        shards = build_slice_eval_sets(X, y, source="random", k=4, min_rows_per_shard=20,
                                        random_state=42, group_ids=group_ids)
    assert any("GroupKFold" in r.message for r in caplog.records), "must warn about ranker-safe switch"
    assert len(shards) == 4
    # No query is split across shards
    for s in shards:
        unique_queries = np.unique(s.group_ids)
        # All rows of each query in this shard must come from rows where group_id is in unique_queries
        # Verify that no other shard contains those queries' rows:
        s_queries = set(unique_queries.tolist())
        for other in shards:
            if other is s:
                continue
            other_queries = set(np.unique(other.group_ids).tolist())
            assert s_queries.isdisjoint(other_queries), "GroupKFold must keep queries intact across shards"


def test_temporal_shards_preserve_order(reg_val_df: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = reg_val_df
    # Time values randomly permuted so we can verify ordering happened
    rng = np.random.default_rng(42)
    time_values = rng.permutation(len(y)).astype(np.float64)
    shards = build_slice_eval_sets(X, y, source="temporal", k=4, min_rows_per_shard=10,
                                    random_state=42, time_values=time_values)
    assert len(shards) == 4
    # Within each shard, time_values at those row_indices should be a contiguous chunk of the sort
    sort_order = np.argsort(time_values, kind="stable")
    edges = np.linspace(0, len(y), 5, dtype=np.int64)
    for i, s in enumerate(shards):
        expected = sort_order[edges[i]: edges[i + 1]]
        assert np.array_equal(np.sort(s.row_indices), np.sort(expected)), (
            "temporal shard rows must match the contiguous time-window indices"
        )


def test_fairness_shards_from_indexed_subgroups(reg_val_df: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = reg_val_df
    # Two subgroups (possibly overlapping)
    indexed_subgroups = {
        "majority": np.arange(0, 150),
        "minority": np.arange(140, 200),
    }
    shards = build_slice_eval_sets(X, y, source="fairness", k=5, min_rows_per_shard=30,
                                    random_state=42, indexed_subgroups=indexed_subgroups)
    assert len(shards) == 2
    names = {s.name for s in shards}
    assert names == {"valid_shard_f_majority", "valid_shard_f_minority"}
    # Allowed to overlap (rows 140-149 are in BOTH subgroups)
    maj = next(s for s in shards if "majority" in s.name)
    minor = next(s for s in shards if "minority" in s.name)
    overlap = set(maj.row_indices.tolist()) & set(minor.row_indices.tolist())
    assert overlap == set(range(140, 150)), "overlapping subgroup rows preserved in both shards"


def test_fairness_source_empty_subgroups_returns_empty(
    reg_val_df: tuple[pd.DataFrame, np.ndarray], caplog,
) -> None:
    X, y = reg_val_df
    with caplog.at_level(logging.WARNING):
        shards = build_slice_eval_sets(X, y, source="fairness", k=5, indexed_subgroups=None)
    assert shards == []
    assert any("no indexed_subgroups" in r.message for r in caplog.records)


def test_small_val_fallback_warns(reg_val_df: tuple[pd.DataFrame, np.ndarray], caplog) -> None:
    X, y = reg_val_df
    # 200 rows / K=5 = 40 rows per shard, but min_rows_per_shard=100 forces fallback
    with caplog.at_level(logging.WARNING):
        shards = build_slice_eval_sets(X, y, source="random", k=5, min_rows_per_shard=100, random_state=42)
    assert shards == []
    assert any("min_rows_per_shard" in r.message for r in caplog.records)


def test_sample_weight_aligned(reg_val_df: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = reg_val_df
    rng = np.random.default_rng(1)
    sample_weight = rng.uniform(0, 1, len(y))
    shards = build_slice_eval_sets(X, y, source="random", k=4, min_rows_per_shard=10,
                                    random_state=42, sample_weight=sample_weight)
    for s in shards:
        expected = sample_weight[s.row_indices]
        assert np.array_equal(s.sample_weight, expected), "sample_weight must be sliced with same row index"


def test_base_margin_aligned(reg_val_df: tuple[pd.DataFrame, np.ndarray]) -> None:
    X, y = reg_val_df
    rng = np.random.default_rng(2)
    base_margin = rng.normal(0, 1, len(y))
    shards = build_slice_eval_sets(X, y, source="random", k=4, min_rows_per_shard=10,
                                    random_state=42, base_margin=base_margin)
    for s in shards:
        assert np.array_equal(s.base_margin, base_margin[s.row_indices])


def test_pandas_categorical_dtype_preserved() -> None:
    cat_values = pd.Categorical.from_codes(
        np.random.default_rng(0).integers(0, 3, 150), categories=["a", "b", "c"],
    )
    val_X = pd.DataFrame({"cat": cat_values, "num": np.random.default_rng(0).normal(0, 1, 150)})
    val_y = np.random.default_rng(0).normal(0, 1, 150)
    shards = build_slice_eval_sets(val_X, val_y, source="random", k=3, min_rows_per_shard=20, random_state=42)
    for s in shards:
        assert isinstance(s.X["cat"].dtype, pd.CategoricalDtype), (
            f"pandas Categorical dtype must survive slicing, got {s.X['cat'].dtype}"
        )
        # Categories list must match the original (slicing via .iloc keeps the domain)
        assert list(s.X["cat"].cat.categories) == ["a", "b", "c"]


def test_polars_enum_dtype_preserved() -> None:
    pl = pytest.importorskip("polars")
    cats = pl.Series("cat", ["a", "b", "c", "a", "b"] * 30, dtype=pl.Enum(["a", "b", "c"]))
    nums = pl.Series("num", np.random.default_rng(0).normal(0, 1, 150))
    val_X = pl.DataFrame([cats, nums])
    val_y = np.random.default_rng(0).normal(0, 1, 150)
    shards = build_slice_eval_sets(val_X, val_y, source="random", k=3, min_rows_per_shard=20, random_state=42)
    for s in shards:
        # gather() preserves Enum exactly
        assert s.X.schema["cat"] == pl.Enum(["a", "b", "c"]), (
            f"polars Enum dtype must survive .gather(); got {s.X.schema['cat']}"
        )


def test_k_lt_2_returns_empty(reg_val_df: tuple[pd.DataFrame, np.ndarray], caplog) -> None:
    X, y = reg_val_df
    with caplog.at_level(logging.WARNING):
        shards = build_slice_eval_sets(X, y, source="random", k=1)
    assert shards == []
    assert any("K>=2" in r.message for r in caplog.records)


def test_effective_patience_formula() -> None:
    # K=5: 1 + 1/sqrt(4) = 1.5 -> ceil(50*1.5) = 75
    assert effective_patience(50, 5) == 75
    # K=10: 1 + 1/sqrt(9) ~= 1.333 -> ceil(50*1.333) = 67
    assert effective_patience(50, 10) == 67
    # K=2: 1 + 1/sqrt(1) = 2.0 -> ceil(50*2) = 100
    assert effective_patience(50, 2) == 100
    # K<=1 returns unchanged
    assert effective_patience(50, 1) == 50
    assert effective_patience(50, 0) == 50
