"""Cluster 25 regression: duplicate quantile alphas are refused rather than silently dropped from the per-alpha result (audit 2026-11-20, finding 25-F70)."""

import numpy as np
import pytest

from mlframe.metrics.quantile import pinball_loss_per_alpha


def test_duplicate_alphas_are_refused():
    """Two columns at the same alpha would collapse to one key; the call raises instead (25-F70)."""
    y = np.array([1.0, 2.0, 3.0])
    preds = np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
    with pytest.raises(ValueError, match="distinct"):
        pinball_loss_per_alpha(y, preds, [0.5, 0.5])


def test_distinct_alphas_return_one_key_each():
    """Control: distinct alphas give one loss per alpha (25-F70)."""
    y = np.array([1.0, 2.0, 3.0])
    preds = np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]])
    out = pinball_loss_per_alpha(y, preds, [0.1, 0.9])
    assert set(out) == {0.1, 0.9}
