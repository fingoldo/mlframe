"""AUC with a NaN score is NaN, not a finite number computed with the NaN ranked as the best score."""
import numpy as np

from mlframe.metrics import fast_roc_auc


def test_nan_score_makes_the_auc_nan():
    """y = [0, 1, 1], score = [nan, .2, .9]: NaN sorted first in the descending order and the AUC read 0.0."""
    assert np.isnan(fast_roc_auc(np.array([0, 1, 1]), np.array([np.nan, 0.2, 0.9])))


def test_finite_scores_are_unchanged():
    """No NaN, no change."""
    assert fast_roc_auc(np.array([0, 1, 1, 0]), np.array([0.1, 0.8, 0.9, 0.3])) == 1.0
