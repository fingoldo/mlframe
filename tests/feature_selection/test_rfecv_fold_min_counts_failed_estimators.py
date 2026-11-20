"""A failed estimator makes the fold score NaN; it is not dropped from the worst-case minimum (audit 2026-11-20, finding 15-F1)."""
import numpy as np

from mlframe.feature_selection.wrappers.rfecv._fit_fold import _commit_fold_scores


def test_nan_estimator_score_makes_the_fold_nan():
    """With one estimator failed the fold must not be decided by the other estimator alone."""
    scores, fis = [], {}
    assert np.isnan(_commit_fold_scores([0.9, float("nan")], [("e0", {}), ("e1", {})], scores, fis))


def test_worst_case_minimum_when_all_estimators_score():
    """All estimators scored: the fold takes the worst of them."""
    scores, fis = [], {}
    assert _commit_fold_scores([0.9, 0.8], [("e0", {}), ("e1", {})], scores, fis) == 0.8
