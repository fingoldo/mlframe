"""Unit tests for `training.ranking` helpers.

Pre-existing `test_ranking_strategies.py` / `test_ranking_ensemble.py` /
`test_ranking_metrics.py` / `test_ranking_splitting.py` exercise suite-level
paths. This file adds focused per-function tests for `qid_to_group_sizes`
and the three `prepare_*_inputs` backends to lock parity contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.ranking import (
    prepare_cb_inputs,
    prepare_lgb_inputs,
    prepare_xgb_inputs,
    qid_to_group_sizes,
)


# ----- qid_to_group_sizes --------------------------------------------------


def test_qid_to_group_sizes_basic_contiguous():
    # docstring example.
    """Qid to group sizes basic contiguous."""
    g = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    sizes = qid_to_group_sizes(g)
    np.testing.assert_array_equal(sizes, [3, 2, 4])


def test_qid_to_group_sizes_empty():
    """Qid to group sizes empty."""
    sizes = qid_to_group_sizes(np.array([], dtype=int))
    assert sizes.shape == (0,)
    assert sizes.dtype == np.intp


def test_qid_to_group_sizes_single_query():
    """Qid to group sizes single query."""
    sizes = qid_to_group_sizes(np.array([7, 7, 7, 7]))
    np.testing.assert_array_equal(sizes, [4])


def test_qid_to_group_sizes_singleton_queries():
    """Qid to group sizes singleton queries."""
    sizes = qid_to_group_sizes(np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(sizes, [1, 1, 1, 1])


def test_qid_to_group_sizes_total_equals_n_rows():
    """Qid to group sizes total equals n rows."""
    rng = np.random.default_rng(0)
    n_queries = 50
    sizes_truth = rng.integers(1, 8, size=n_queries)
    g = np.repeat(np.arange(n_queries), sizes_truth)
    sizes = qid_to_group_sizes(g)
    assert sizes.sum() == len(g)
    np.testing.assert_array_equal(sizes, sizes_truth)


def test_qid_to_group_sizes_parity_with_unique_return_counts():
    # The numpy reference: np.unique(..., return_counts=True) matches when
    # group_ids are contiguous AND sorted. (Otherwise diff path is correct
    # per the docstring; unique sums duplicates across the array.)
    """Qid to group sizes parity with unique return counts."""
    g = np.array([0, 0, 1, 1, 1, 2, 2])
    sizes = qid_to_group_sizes(g)
    _, expected = np.unique(g, return_counts=True)
    np.testing.assert_array_equal(sizes, expected)


# ----- prepare_*_inputs backends ------------------------------------------


def _make_synthetic(n_rows: int = 30, n_features: int = 3, n_queries: int = 6, seed: int = 0):
    """Make synthetic."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = rng.integers(0, 5, size=n_rows)
    # Contiguous group_ids: evenly split rows into n_queries.
    group_ids = np.repeat(np.arange(n_queries), n_rows // n_queries)
    # Pad to n_rows if not divisible.
    if group_ids.size < n_rows:
        group_ids = np.concatenate([group_ids, np.full(n_rows - group_ids.size, n_queries - 1)])
    return X, y, group_ids


def test_prepare_cb_inputs_sorted_input_returns_in_place():
    """Prepare cb inputs sorted input returns in place."""
    X, y, g = _make_synthetic(seed=1)
    _X_out, y_out, g_out, sort_idx = prepare_cb_inputs(X, y, g)
    # Sorted -> sort_idx is the identity permutation.
    np.testing.assert_array_equal(sort_idx, np.arange(len(y)))
    np.testing.assert_array_equal(g_out, g)
    np.testing.assert_array_equal(y_out, y)


def test_prepare_cb_inputs_unsorted_input_gets_sorted():
    """Prepare cb inputs unsorted input gets sorted."""
    X, y, g = _make_synthetic(seed=2)
    perm = np.random.default_rng(99).permutation(len(y))
    X_perm = X.iloc[perm].reset_index(drop=True)
    y_perm = y[perm]
    g_perm = g[perm]
    _X_out, y_out, g_out, sort_idx = prepare_cb_inputs(X_perm, y_perm, g_perm)
    # Output group_ids must be non-decreasing (sorted).
    assert np.all(np.diff(g_out) >= 0)
    # y / X re-aligned: undo via argsort(sort_idx) recovers permuted input.
    inv = np.argsort(sort_idx)
    np.testing.assert_array_equal(y_out[inv], y_perm)


def test_prepare_xgb_inputs_pass_through():
    """Prepare xgb inputs pass through."""
    X, y, g = _make_synthetic(seed=3)
    X_out, y_out, g_out = prepare_xgb_inputs(X, y, g)
    # XGB requires NO sort; outputs equal inputs (qid is the per-row group array).
    assert X_out is X  # exact identity (no copy)
    np.testing.assert_array_equal(y_out, y)
    np.testing.assert_array_equal(g_out, g)


def test_prepare_lgb_inputs_sorted_uses_identity_sort_idx():
    """Prepare lgb inputs sorted uses identity sort idx."""
    X, y, g = _make_synthetic(seed=4)
    _X_out, _y_out, group_sizes, sort_idx = prepare_lgb_inputs(X, y, g)
    np.testing.assert_array_equal(sort_idx, np.arange(len(y)))
    # group_sizes must sum to row count.
    assert group_sizes.sum() == len(y)


def test_prepare_lgb_inputs_unsorted_returns_sorted_group_sizes():
    """Prepare lgb inputs unsorted returns sorted group sizes."""
    X, y, g = _make_synthetic(seed=5)
    perm = np.random.default_rng(11).permutation(len(y))
    X_perm = X.iloc[perm].reset_index(drop=True)
    y_perm = y[perm]
    g_perm = g[perm]
    _X_out, _y_out, group_sizes, _sort_idx = prepare_lgb_inputs(X_perm, y_perm, g_perm)
    # After sort, group_sizes sum stays N.
    assert group_sizes.sum() == len(y)


def test_prepare_inputs_parity_row_counts_across_backends():
    # CB / XGB / LGB MUST agree on row counts on the same input. A silent
    # divergence here was historically the most common ranking-suite bug.
    """Prepare inputs parity row counts across backends."""
    X, y, g = _make_synthetic(seed=6)
    _cb_X, cb_y, cb_g, _ = prepare_cb_inputs(X, y, g)
    _xgb_X, xgb_y, xgb_g = prepare_xgb_inputs(X, y, g)
    _lgb_X, lgb_y, lgb_sizes, _ = prepare_lgb_inputs(X, y, g)
    assert len(cb_y) == len(xgb_y) == len(lgb_y) == len(y)
    # CB and LGB sort by group: their group-derived sizes must match.
    cb_sizes = qid_to_group_sizes(cb_g)
    np.testing.assert_array_equal(cb_sizes, lgb_sizes)
    # XGB keeps per-row qid; sum of unique counts must equal N.
    assert xgb_g.shape[0] == len(y)


def test_prepare_cb_inputs_rejects_length_mismatch():
    """Prepare cb inputs rejects length mismatch."""
    X, y, g = _make_synthetic(seed=7)
    with pytest.raises(ValueError, match="length mismatch"):
        prepare_cb_inputs(X, y[:-1], g)


def test_prepare_xgb_inputs_rejects_empty():
    """Prepare xgb inputs rejects empty."""
    X = pd.DataFrame(columns=["f0"])
    with pytest.raises(ValueError, match="empty"):
        prepare_xgb_inputs(X, np.array([]), np.array([]))


def test_prepare_lgb_inputs_rejects_length_mismatch():
    """Prepare lgb inputs rejects length mismatch."""
    X, y, g = _make_synthetic(seed=8)
    with pytest.raises(ValueError, match="length mismatch"):
        prepare_lgb_inputs(X, y, g[:-1])
