"""Cluster 26 regression: the batch univariate AUC refuses multi-class targets rather than scoring one label one-vs-rest (audit 2026-11-20, finding 26-F14)."""

import numpy as np
import pytest

from mlframe.preprocessing.align_feature_direction import batch_univariate_auc


def test_multiclass_target_is_refused():
    """A three-class y raises instead of silently treating label 1 as the positive class (26-F14)."""
    X = np.random.default_rng(0).normal(size=(30, 2))
    y = np.repeat([0, 1, 2], 10)
    with pytest.raises(ValueError, match="binary-only"):
        batch_univariate_auc(X, y)


def test_binary_target_still_scores():
    """Control: a binary y returns one AUC per column (26-F14)."""
    X = np.random.default_rng(1).normal(size=(40, 3))
    y = np.repeat([0, 1], 20)
    out = batch_univariate_auc(X, y)
    assert out.shape == (3,)
