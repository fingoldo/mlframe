"""Per-fold scores land at their fold's position even when worker threads finish out of order (cv_results_ split columns)."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.wrappers.rfecv._fit_fold import _commit_fold_scores
from mlframe.feature_selection.wrappers.rfecv._fold_scores import FoldOrderedScores


def test_scores_committed_out_of_order_are_stored_in_fold_order():
    """Folds 2, 0, 1 finishing in that order are read back as folds 0, 1, 2."""
    scores = FoldOrderedScores()
    fi: dict = {}
    for nfold, value in [(2, 0.30), (0, 0.10), (1, 0.20)]:
        _commit_fold_scores([value], [], scores, fi, nfold)
    assert list(scores) == [0.10, 0.20, 0.30]


def test_nan_fold_keeps_its_slot():
    """A skipped fold stores NaN at its own position."""
    scores = FoldOrderedScores()
    _commit_fold_scores([0.5], [], scores, {}, 1)
    scores.add(0, float("nan"))
    assert np.isnan(scores[0]) and scores[1] == 0.5
