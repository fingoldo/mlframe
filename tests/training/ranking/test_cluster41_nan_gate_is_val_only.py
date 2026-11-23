"""Cluster 41 regression: ensemble membership ignores test-set NaNs (audit 2026-11-20, finding 41-F24)."""

import numpy as np

from mlframe.training.ranking._ranker_suite_train_helpers import _sanitize_test_scores, _train_mlframe_rank_lockstep_rank_fusion_math


def test_member_with_nan_only_in_test_is_kept():
    """A member whose test scores contain NaN but whose val scores are clean stays in the ensemble (41-F24)."""
    val = [np.array([0.1, 0.5, 0.9]), np.array([0.2, 0.4, 0.8])]
    test = [np.array([0.1, 0.5, 0.9]), np.array([np.nan, 0.4, 0.8])]
    models = {"a": {"val_metrics": {"ndcg@10": 0.9}}, "b": {"val_metrics": {"ndcg@10": 0.8}}}
    order, gated_test, _gated_val = _train_mlframe_rank_lockstep_rank_fusion_math(True, val, ["a", "b"], test, models)
    assert order == ["a", "b"]
    assert np.all(np.isfinite(gated_test[1]))


def test_nan_test_score_is_floored_not_dropped():
    """The NaN test row is replaced by the member's lowest finite score (41-F24)."""
    out = _sanitize_test_scores(np.array([np.nan, 0.4, 0.8]))
    assert out.tolist() == [0.4, 0.4, 0.8]


def test_clean_scores_pass_through_unchanged():
    """Control: all-finite test scores are returned as they are (41-F24)."""
    out = _sanitize_test_scores(np.array([0.1, 0.2]))
    assert out.tolist() == [0.1, 0.2]
