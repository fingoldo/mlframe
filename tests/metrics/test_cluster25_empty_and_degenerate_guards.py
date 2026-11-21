"""Audit 2026-11-20 findings 25-F42 and 25-F43: empty and zero-label inputs return NaN instead of dividing by zero.

fast_rmse on no rows and hamming_loss on a zero-label matrix both divided by an empty count. The rmse_loss sibling returns NaN
for no rows, so the fast path now does the same.
"""

import numpy as np

from mlframe.metrics._multilabel_metrics import hamming_loss
from mlframe.metrics.scoring import fast_rmse


def test_fast_rmse_on_no_rows_is_nan():
    """No rows gives NaN, matching rmse_loss, not a ZeroDivisionError (25-F42)."""
    assert np.isnan(fast_rmse(np.array([], dtype=np.float64), np.array([], dtype=np.float64)))


def test_hamming_loss_with_zero_labels_is_nan():
    """A (5, 0) label matrix has no label to score: NaN, not a ZeroDivisionError (25-F43)."""
    assert np.isnan(hamming_loss(np.zeros((5, 0), dtype=np.uint8), np.zeros((5, 0), dtype=np.uint8)))


def test_hamming_loss_still_scores_a_normal_matrix():
    """The guard does not change a real score: one mismatch in a 2 x 2 matrix is 0.25."""
    a = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    b = np.array([[0, 1], [1, 1]], dtype=np.uint8)
    assert abs(hamming_loss(a, b) - 0.25) < 1e-12
