"""Tests for ``_setup_eval_set`` extension with ``extra_eval_sets`` + aligned arrays.

The function builds the booster-specific kwarg dict; we verify the structure of that dict
across model categories without actually running a fit (that lives in the manual E2E suite).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training._data_helpers import _setup_eval_set, _groupids_to_sizes
from mlframe.training.slicing._slice_helpers import SliceEvalSet


@pytest.fixture
def val_data() -> tuple[pd.DataFrame, np.ndarray, list[SliceEvalSet]]:
    """4 shards of 20 rows each, sharing the same column schema."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(0, 1, 100), "b": rng.normal(0, 1, 100)})
    y = rng.normal(0, 1, 100)
    shards = [
        SliceEvalSet(
            name=f"valid_shard_r{i}",
            X=X.iloc[i * 20 : (i + 1) * 20].reset_index(drop=True),
            y=y[i * 20 : (i + 1) * 20],
            sample_weight=np.full(20, 0.5 + 0.1 * i),
            base_margin=np.full(20, float(i)),
            group_ids=None,
            row_indices=np.arange(i * 20, (i + 1) * 20),
        )
        for i in range(4)
    ]
    return X, y, shards


def test_default_no_shards_bit_identical(val_data) -> None:
    """Default no shards bit identical."""
    X, y, _ = val_data
    fit_params: dict = {}
    _setup_eval_set("XGBClassifier", fit_params, X, y, model_category="xgb")
    # Without extra_eval_sets the XGB path produces a single-element list
    assert fit_params["eval_set"] == [(X, y)]
    assert "sample_weight_eval_set" not in fit_params
    assert "base_margin_eval_set" not in fit_params


def test_xgb_with_shards_appends_parallel_arrays(val_data) -> None:
    """Xgb with shards appends parallel arrays."""
    X, y, shards = val_data
    fit_params: dict = {}
    _setup_eval_set(
        "XGBClassifier",
        fit_params,
        X,
        y,
        model_category="xgb",
        extra_eval_sets=shards,
        sample_weight_val=np.full(100, 1.0),
        base_margin_val=np.zeros(100),
    )
    eval_set = fit_params["eval_set"]
    assert len(eval_set) == 5, "1 full val + 4 shards"
    assert eval_set[0][0] is X
    assert all(eval_set[i + 1][0] is shards[i].X for i in range(4))
    # sample_weight_eval_set / base_margin_eval_set are parallel-aligned
    sw = fit_params["sample_weight_eval_set"]
    bm = fit_params["base_margin_eval_set"]
    assert len(sw) == 5 and len(bm) == 5
    assert np.array_equal(sw[0], np.full(100, 1.0))
    for i in range(4):
        assert np.array_equal(sw[i + 1], shards[i].sample_weight)
        assert np.array_equal(bm[i + 1], shards[i].base_margin)


def test_lgb_with_shards_uses_list_format(val_data) -> None:
    """Lgb with shards uses list format."""
    X, y, shards = val_data
    fit_params: dict = {}
    _setup_eval_set("LGBMClassifier", fit_params, X, y, model_category="lgb", extra_eval_sets=shards)
    # LGB legacy path was tuple-only; shards force a list-of-tuples (LGB accepts both).
    assert isinstance(fit_params["eval_set"], list)
    assert len(fit_params["eval_set"]) == 5


def test_cb_with_shards_propagates_sample_weight(val_data) -> None:
    """Cb with shards propagates sample weight."""
    X, y, shards = val_data
    fit_params: dict = {}
    _setup_eval_set("CatBoostClassifier", fit_params, X, y, model_category="cb", extra_eval_sets=shards, sample_weight_val=np.full(100, 0.9))
    assert len(fit_params["eval_set"]) == 5
    sw = fit_params["sample_weight_eval_set"]
    assert len(sw) == 5
    assert np.array_equal(sw[0], np.full(100, 0.9))


def test_xgb_qid_propagation(val_data) -> None:
    """Xgb qid propagation."""
    X, y, _ = val_data
    # Build qid-aware shards (10 queries of 10 docs each, shard by query).
    qid = np.repeat(np.arange(10), 10)
    shards = [
        SliceEvalSet(
            name=f"valid_shard_r{i}",
            X=X.iloc[i * 50 : (i + 1) * 50].reset_index(drop=True),
            y=y[i * 50 : (i + 1) * 50],
            group_ids=qid[i * 50 : (i + 1) * 50],
            row_indices=np.arange(i * 50, (i + 1) * 50),
        )
        for i in range(2)
    ]
    fit_params: dict = {}
    _setup_eval_set("XGBRanker", fit_params, X, y, model_category="xgb", extra_eval_sets=shards, group_ids_val=qid)
    assert "eval_qid" in fit_params
    assert len(fit_params["eval_qid"]) == 3
    assert np.array_equal(fit_params["eval_qid"][0], qid)


def test_lgb_eval_group_converts_qid_to_sizes(val_data) -> None:
    """Lgb eval group converts qid to sizes."""
    X, y, _ = val_data
    qid = np.repeat(np.arange(10), 10)
    shards = [
        SliceEvalSet(
            name=f"valid_shard_r{i}",
            X=X.iloc[i * 50 : (i + 1) * 50].reset_index(drop=True),
            y=y[i * 50 : (i + 1) * 50],
            group_ids=qid[i * 50 : (i + 1) * 50],
            row_indices=np.arange(i * 50, (i + 1) * 50),
        )
        for i in range(2)
    ]
    fit_params: dict = {}
    _setup_eval_set("LGBMRanker", fit_params, X, y, model_category="lgb", extra_eval_sets=shards, group_ids_val=qid)
    # eval_group is per-query SIZE list, not ids
    eg = fit_params["eval_group"]
    assert len(eg) == 3
    # Full val: 10 queries x 10 docs -> [10]*10
    assert np.array_equal(eg[0], np.full(10, 10, dtype=np.int64))


def test_groupids_to_sizes_basic() -> None:
    """Groupids to sizes basic."""
    qid = np.array([1, 1, 1, 2, 2, 3, 3, 3, 3], dtype=np.int64)
    sizes = _groupids_to_sizes(qid)
    assert np.array_equal(sizes, np.array([3, 2, 4], dtype=np.int64))


def test_groupids_to_sizes_none() -> None:
    """Groupids to sizes none."""
    assert _groupids_to_sizes(None) is None


def test_multioutput_skip_preserved(val_data) -> None:
    """Multioutput skip preserved."""
    X, y, shards = val_data
    fit_params: dict = {}
    _setup_eval_set("MultiOutputClassifier", fit_params, X, y, model_category="cb", extra_eval_sets=shards)
    # MultiOutput path returns early without setting eval_set
    assert "eval_set" not in fit_params
