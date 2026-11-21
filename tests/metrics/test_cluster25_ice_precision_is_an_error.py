"""Audit 2026-11-20 finding 25-F47: the ICE "precision" method reports an error, so higher is worse like every other method.

Before the fix it returned precision itself, which moves the opposite way to the other methods' errors. The extremes pin the
direction: perfect precision gives 0, and all-wrong positives give 1.
"""

import numpy as np

from mlframe.metrics._ice_metric import compute_probabilistic_multiclass_error


def test_perfect_precision_is_zero_error():
    """Every positive prediction is correct: the error is 0."""
    y = np.array([1, 1, 0, 0])
    assert compute_probabilistic_multiclass_error(y, np.array([0.9, 0.8, 0.1, 0.2]), method="precision") == 0.0


def test_all_wrong_positives_is_full_error():
    """Every positive prediction is wrong: the error is 1, the worst value."""
    y = np.array([1, 1, 0, 0])
    assert compute_probabilistic_multiclass_error(y, np.array([0.1, 0.2, 0.9, 0.8]), method="precision") == 1.0
