"""TRAINING_COMPOSITE_CORE_A-7 (2026-08-05 audit): ``n_features`` must treat a 1-D array as a single feature
column (returning 1), not 0 -- matching the sklearn convention of ``X.reshape(-1, 1)`` for a bare 1-D
feature vector.
"""

from __future__ import annotations

import numpy as np

from mlframe.training.composite._composite_array_shared import n_features


def test_1d_ndarray_returns_one_feature():
    """A 1-D ndarray must be treated as a single feature column."""
    assert n_features(np.arange(10)) == 1


def test_2d_ndarray_unchanged():
    """A 2-D ndarray's feature count is unaffected by the 1-D fix."""
    assert n_features(np.zeros((10, 3))) == 3


def test_0d_scalar_returns_zero():
    """A 0-D scalar array has no feature axis and must return 0."""
    assert n_features(np.array(5.0)) == 0
