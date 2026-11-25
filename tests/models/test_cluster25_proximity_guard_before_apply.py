"""Cluster 25 regression: the row-count guard fires before the forest is applied (audit 2026-11-20, finding 25-F63)."""

import numpy as np
import pytest

from mlframe.models.rf_proximity import rf_proximity_matrix


class _Forest:
    """Forest stand-in that counts how often it is applied."""

    def __init__(self):
        """Start with no applies recorded."""
        self.calls = 0

    def apply(self, X):
        """Return per-row leaf ids and record the call."""
        self.calls += 1
        return np.zeros((len(X), 2), dtype=np.int64)


def test_guard_refuses_before_apply():
    """With X larger than max_n the error is raised and the forest is never applied (25-F63)."""
    forest = _Forest()
    with pytest.raises(ValueError, match="exceeds max_n"):
        rf_proximity_matrix(forest, np.zeros((10, 2)), max_n=5)
    assert forest.calls == 0
