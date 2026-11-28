"""Cluster 08 regression: the regression aux MLP emits only its prediction, not an all-zero residual column (audit 2026-11-20, finding 08-F7)."""

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer.aux_mlp import compute_aux_mlp_features


def test_regression_outputs_one_column_in_both_modes():
    """Mode A and Mode B return only ``<prefix>_pred`` for a regression target."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4)).astype(np.float32)
    y = X[:, 0] + 0.1 * rng.normal(size=120)
    oof = compute_aux_mlp_features(X, y, None, KFold(3), seed=0, task="regression", max_iter=30)
    query = compute_aux_mlp_features(X, y, X[:10], None, seed=0, task="regression", max_iter=30)
    assert oof.columns == ["mlp_pred"] and query.columns == ["mlp_pred"]
