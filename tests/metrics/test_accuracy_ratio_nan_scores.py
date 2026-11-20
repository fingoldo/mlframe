"""accuracy_ratio is NaN, not a finite value, when a score is NaN (audit 2026-11-20, finding 17-F5)."""
import numpy as np

from mlframe.metrics.classification._classification_calibration import accuracy_ratio


def test_nan_score_gives_nan():
    """The default early-stopping scorer must not read a missing score as a ranking."""
    y = np.array([0, 1, 0, 1, 1])
    s = np.array([0.1, 0.9, np.nan, 0.8, 0.4])
    assert np.isnan(accuracy_ratio(y, s))


def test_finite_scores_still_give_the_ratio():
    """Finite scores keep the ratio: a perfect ranking has AR 1."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert abs(accuracy_ratio(y, s) - 1.0) < 1e-9
