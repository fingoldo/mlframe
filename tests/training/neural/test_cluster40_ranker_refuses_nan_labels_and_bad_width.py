"""Audit 2026-11-20 finding 40-F22: MLPRanker refuses NaN relevance labels and a feature-count mismatch at predict."""

import numpy as np
import pytest

pytest.importorskip("lightning")
from mlframe.training.neural.ranker import MLPRanker


def _data(n=40, k=3, seed=0):
    """Small synthetic ranking frame: 4 rows per query, label driven by the first column."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, k))
    y = (X[:, 0] > 0).astype(float)
    group = np.repeat(np.arange(n // 4), 4)
    return X, y, group


def test_nan_relevance_labels_are_refused():
    """NaN in y would silently drop pairs from training; fit refuses instead (40-F22)."""
    X, y, group = _data()
    y[3] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        MLPRanker(n_epochs=1, random_state=0).fit(X, y, group_ids=group)


def test_predict_refuses_a_different_feature_count():
    """predict with a different number of columns than fit raises instead of scoring garbage (40-F22)."""
    X, y, group = _data()
    m = MLPRanker(n_epochs=1, random_state=0).fit(X, y, group_ids=group)
    with pytest.raises(ValueError, match="features"):
        m.predict(X[:, :2])
