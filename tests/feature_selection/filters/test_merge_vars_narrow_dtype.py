"""merge_vars refuses a working dtype too narrow for the joint class ids instead of wrapping silently (tracker 2026-11-02, INT7-1)."""

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory._class_encoding import merge_vars


def test_merge_vars_int8_overflow_raises_not_wraps():
    """Two 16-bin columns give 256 joint cells, which an int8 class array cannot hold."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 16, size=(500, 2)).astype(np.int32)
    nbins = np.array([16, 16], dtype=np.int64)
    with pytest.raises(ValueError):
        merge_vars(data, [0, 1], None, nbins, np.int8)


def test_merge_vars_int32_unchanged():
    """The default dtype still returns frequencies summing to one."""
    data = np.array([[0, 1], [1, 0], [1, 1], [0, 1]], dtype=np.int32)
    _, freqs, ncl = merge_vars(data, [0, 1], None, np.array([2, 2], dtype=np.int64), np.int32)
    assert ncl == 3
    assert abs(freqs.sum() - 1.0) < 1e-12
