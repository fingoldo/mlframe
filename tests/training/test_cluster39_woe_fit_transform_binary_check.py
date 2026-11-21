"""Audit 2026-11-20 finding 39-F17: fit_transform applies the same binary {0, 1} target check for WoE as fit does."""

import numpy as np
import pytest

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def test_woe_fit_transform_refuses_a_non_binary_target():
    """A three-level target with method='woe' is refused by fit_transform, not silently encoded."""
    enc = LeakageSafeEncoder(method="woe")
    col = np.array(["a", "b", "a", "b", "c", "c"], dtype=object)
    y = np.array([0, 1, 2, 1, 0, 2])
    with pytest.raises(ValueError, match="binary"):
        enc.fit_transform(col, y)


def test_woe_fit_transform_accepts_a_binary_target():
    """A {0, 1} target goes through fit_transform and returns one encoding per row."""
    enc = LeakageSafeEncoder(method="woe")
    col = np.array(["a", "b", "a", "b", "a", "b", "a", "b"] * 3, dtype=object)
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 3)
    out = enc.fit_transform(col, y)
    assert np.asarray(out).shape[0] == len(y)
