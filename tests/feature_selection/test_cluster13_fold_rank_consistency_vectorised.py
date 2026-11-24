"""Cluster 13 regression: fold_rank_consistency builds the per-fold top mask without a Python loop over columns (audit 2026-11-20, finding 13-F21).

The result must match a naive per-column reference exactly; the loop was replaced by a boolean mask over the top indices.
"""

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxied_fit_residual import fold_rank_consistency


def _reference(per_fold, top_k):
    """Naive per-column membership reference for the consistency mask."""
    fold_pool = min(per_fold.shape[1], max(top_k * 3, 1))
    consistent = np.ones(per_fold.shape[1], dtype=bool)
    for fold_row in per_fold:
        fold_top = set(np.argsort(-fold_row)[:fold_pool].tolist())
        consistent &= np.array([j in fold_top for j in range(per_fold.shape[1])], dtype=bool)
    return consistent, fold_pool


def test_matches_naive_reference():
    """The vectorised mask equals the per-column reference on random fold importances (13-F21)."""
    rng = np.random.default_rng(0)
    per_fold = rng.random((4, 30))
    got, pool_got = fold_rank_consistency(per_fold, top_k=3)
    want, pool_want = _reference(per_fold, top_k=3)
    np.testing.assert_array_equal(got, want)
    assert pool_got == pool_want


def test_column_outside_top_pool_in_one_fold_is_dropped():
    """A column that falls out of the pool in any fold is not consistent (13-F21)."""
    per_fold = np.array([[9.0, 8.0, 1.0, 0.5], [9.0, 0.1, 8.0, 0.5]])
    consistent, pool = fold_rank_consistency(per_fold, top_k=1)
    assert pool == 3
    assert consistent.tolist() == [True, False, True, False]
