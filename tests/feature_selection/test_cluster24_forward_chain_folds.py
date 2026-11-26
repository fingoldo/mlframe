"""Cluster 24 regression: the stability vote's temporal folds never train on a later row (audit 2026-11-20, finding 24-F45).

The partition used to put every other block in training, so a row's held-out value came from models fit on the future. Forward chaining trains
each fold on the blocks before its test block only. The SHAP explainer keeps its partition for now (it needs a value for every row).
"""

import numpy as np

from mlframe.feature_selection.cv_policy import CVPolicy, forward_chain_folds
from mlframe.feature_selection.filters._fe_stability_vote import _vote_folds


def _temporal(n: int) -> CVPolicy:
    """A temporal policy whose timestamps are the row positions, so chronological order equals row order."""
    return CVPolicy("temporal", "test", timestamps=np.arange(n, dtype=np.float64))


def test_no_fold_trains_on_a_later_row():
    """Every training row of fold i is earlier in time than every test row of fold i (24-F45)."""
    n, k = 120, 4
    folds = forward_chain_folds(_temporal(n), n, k)
    assert folds is not None and len(folds) == k - 1
    for tr, te in folds:
        assert tr.size > 0
        assert tr.max() < te.min()


def test_first_block_is_never_a_test_set_and_folds_are_disjoint():
    """The first block has no earlier data, so no fold tests on it; the test sets are disjoint and cover blocks 1..k-1 (24-F45)."""
    n, k = 120, 4
    folds = forward_chain_folds(_temporal(n), n, k)
    tests = [te for _, te in folds]
    covered = np.concatenate(tests)
    assert covered.min() >= n // k
    assert len(np.unique(covered)) == covered.size


def test_stability_vote_folds_use_forward_chaining_under_a_temporal_policy():
    """_vote_folds returns the forward-chained test sets for a temporal policy, not a partition that covers the first block (24-F45)."""
    n, k = 120, 4
    rng = np.random.default_rng(0)
    vote = _vote_folds(_temporal(n), n, k, rng)
    expected = [te for _, te in forward_chain_folds(_temporal(n), n, k)]
    assert len(vote) == len(expected) == k - 1
    for got, want in zip(vote, expected):
        np.testing.assert_array_equal(got, want)


def test_too_few_rows_or_non_temporal_policy_returns_none():
    """Too few rows for the blocks, or a non-temporal policy, leaves the folds to the caller (24-F45)."""
    assert forward_chain_folds(_temporal(5), 5, 4) is None
    assert forward_chain_folds(CVPolicy("iid", "test"), 120, 4) is None
    assert forward_chain_folds(None, 120, 4) is None
