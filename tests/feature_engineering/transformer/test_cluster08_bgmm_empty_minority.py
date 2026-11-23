"""Cluster 08 regression: an empty minority set raises a clear error naming the caller (audit 2026-11-20, finding 08-F18)."""

import numpy as np
import pytest

from mlframe.feature_engineering.transformer._bgmm_sample import fit_bgmm_and_sample


def test_empty_minority_names_the_caller():
    """With zero minority rows the error names the cause and the caller, not numpy's integers(0, 0) (08-F18)."""
    with pytest.raises(ValueError, match="no minority rows"):
        fit_bgmm_and_sample(np.empty((0, 3)), 5, 2, 0, caller="test_caller")


def test_small_minority_still_bootstraps():
    """Control: a few minority rows still produce the requested number of samples (08-F18)."""
    X = np.arange(6, dtype=float).reshape(3, 2)
    out = fit_bgmm_and_sample(X, 7, 5, 0, caller="test_caller")
    assert out.shape == (7, 2)
