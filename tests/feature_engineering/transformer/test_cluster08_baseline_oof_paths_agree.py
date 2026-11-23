"""Cluster 08 regression: the surprise baseline and the shared OOF baseline produce identical out-of-fold predictions (audit 2026-11-20, finding 08-F21).

baseline_surprise keeps its own fit/predict body by design (it also predicts on a query set). This test pins the two
copies together, so a change to the LightGBM settings in one must be made in the other too.
"""

import numpy as np

from mlframe.feature_engineering.transformer import _baseline_oof as bo
from mlframe.feature_engineering.transformer import baseline_surprise as bs


def test_binary_oof_predictions_match_the_shared_helper():
    """Binary task: the surprise baseline's train predictions equal the shared OOF helper's (08-F21)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = (X[:, 0] > 0).astype(float)
    p_surprise, _ = bs._fit_baseline_predict(X, y, X[:5], task="binary", seed=4)
    p_shared = bo.fit_baseline_predict_oof(X, y, "binary", 4)
    np.testing.assert_array_equal(p_surprise, p_shared)


def test_regression_oof_predictions_match_the_shared_helper():
    """Regression task: the surprise baseline's train predictions equal the shared OOF helper's (08-F21)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=60)
    p_surprise, _ = bs._fit_baseline_predict(X, y, X[:5], task="regression", seed=9)
    p_shared = bo.fit_baseline_predict_oof(X, y, "regression", 9)
    np.testing.assert_array_equal(p_surprise, p_shared)
