"""Cluster 09 regression: the binary branch predicts only on the query rows (audit 2026-11-20, finding 09-F32).

The binary output uses only the query predictions, so the three baselines must not run on the training fold. The regression branch still needs
its training predictions for the out-of-fold residuals.
"""

import numpy as np
import pytest

from mlframe.feature_engineering.transformer import pairwise_kl_divergence as pkl


def test_binary_branch_predicts_only_query_rows(monkeypatch):
    """LightGBM predict_proba is never called with the training rows in the binary branch (09-F32)."""
    lgb = pytest.importorskip("lightgbm")
    seen = []
    original = lgb.LGBMClassifier.predict_proba

    def _spy(self, X, *args, **kwargs):
        """Record the row count of every predict_proba call, then delegate."""
        seen.append(len(X))
        return original(self, X, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMClassifier, "predict_proba", _spy)
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(120, 3))
    y_t = (Xt[:, 0] > 0).astype(np.int64)
    Xq = rng.normal(size=(15, 3))
    out = pkl._fit_3baselines_with_sigma(Xt, y_t, Xq, "binary", seed=0)
    assert seen and all(n == 15 for n in seen)
    assert all(len(p) == 15 for p in out[0])
