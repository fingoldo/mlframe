"""Regression: hetero_vote._cv_skill must LOG when CV scoring fails, not silently return 0.0.

Pre-fix the broad ``except Exception: return 0.0`` swallowed any inner failure (a real
shape/dtype/wiring bug, or a degenerate single-class fold) with no trace, silently mis-weighting
the panel vote. The legitimate fallback (zero above-chance skill) is kept, but now logged."""

import logging

import numpy as np
import pytest

from mlframe.feature_selection.hetero_vote import _cv_skill


def test_cv_skill_logs_warning_on_inner_failure(caplog):
    from sklearn.linear_model import LogisticRegression

    # 3 rows + 3 folds + single-class target -> StratifiedKFold/roc_auc raises inside cross_val_score.
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 0, 0])
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.hetero_vote"):
        r = _cv_skill(LogisticRegression(), X, y, classification=True, folds=3, random_state=0)
    assert r == 0.0  # legitimate fallback preserved
    msgs = [rec.getMessage() for rec in caplog.records]
    assert any("_cv_skill" in m and "CV scoring failed" in m for m in msgs), f"inner CV failure must be logged, not silently swallowed; got {msgs!r}"


def test_cv_skill_happy_path_no_warning(caplog):
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    with caplog.at_level(logging.WARNING, logger="mlframe.feature_selection.hetero_vote"):
        r = _cv_skill(DecisionTreeClassifier(random_state=0), X, y, classification=True, folds=3, random_state=0)
    assert r >= 0.0
    assert not [rec for rec in caplog.records if "CV scoring failed" in rec.getMessage()]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov", "-p", "no:cacheprovider"])
