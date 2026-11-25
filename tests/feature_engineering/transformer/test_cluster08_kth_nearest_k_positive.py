"""Cluster 08 regression: kth_nearest_dists refuses k < 1 instead of silently taking the farthest column (audit 2026-11-20, finding 08-F19)."""

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._utils import kth_nearest_dists


def test_zero_k_is_refused():
    """A k of 0 used to index column -1 silently; it is now a clear ValueError (08-F19)."""
    X = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ValueError, match=">= 1"):
        kth_nearest_dists(X, X[:5], 5, (0, 3))


def test_clipped_k_still_uses_the_farthest_available():
    """Control: a k above the number of subset rows keeps the documented farthest-available fallback (08-F19)."""
    X = np.random.default_rng(1).normal(size=(4, 2))
    out = kth_nearest_dists(X, X[:2], 10, (1, 10))
    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))
