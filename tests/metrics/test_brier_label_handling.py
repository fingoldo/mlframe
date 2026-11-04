"""Brier score maps two labels outside {0, 1} with the larger as positive, like sklearn."""
import numpy as np
from sklearn.metrics import brier_score_loss

from mlframe.metrics import fast_brier_score_loss


def test_labels_one_and_two_match_sklearn():
    """Label 2 was squared as y = 2 and gave a plausible-looking number."""
    y = np.array([1, 2, 1, 2, 2, 1])
    p = np.array([0.2, 0.9, 0.3, 0.8, 0.7, 0.1])
    assert abs(fast_brier_score_loss(y, p) - brier_score_loss(y, p)) < 1e-12


def test_zero_one_labels_are_unchanged():
    """The usual case is untouched."""
    y = np.array([0, 1, 1, 0])
    p = np.array([0.1, 0.8, 0.6, 0.3])
    assert abs(fast_brier_score_loss(y, p) - brier_score_loss(y, p)) < 1e-12
