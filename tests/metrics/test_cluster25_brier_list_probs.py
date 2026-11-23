"""Cluster 25 regression: a list y_prob works in the Brier score, as a y_true list already did (audit 2026-11-20, finding 25-F65)."""

import numpy as np
import pytest

from mlframe.metrics._core_brier import fast_brier_score_loss


def test_list_probabilities_match_array_probabilities():
    """A plain-list y_prob gives the same Brier score as the equivalent array (25-F65)."""
    y = [0, 1, 1, 0]
    p_list = [0.1, 0.8, 0.6, 0.3]
    assert fast_brier_score_loss(y, p_list) == pytest.approx(fast_brier_score_loss(np.array(y), np.array(p_list)))
