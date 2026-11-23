"""Cluster 09 regression: the binary baseline branch makes no train-set predictions (audit 2026-11-20, finding 09-F32).

The binary output uses only the query predictions, so the train predict calls were wasted. A predict_proba spy counts calls on the
train frame; the query frame still gets its predictions.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlframe.feature_engineering.transformer import pairwise_kl_divergence as pk


def test_binary_branch_does_not_predict_on_the_train_frame(monkeypatch):
    """predict_proba is called on the query frame only, never on the train frame (09-F32)."""
    rng = np.random.default_rng(0)
    Xt = rng.normal(size=(80, 3))
    Xq = rng.normal(size=(15, 3))
    y_t = (Xt[:, 0] > 0).astype(np.int32)
    seen = []
    real = LogisticRegression.predict_proba

    def _spy(self, X, *a, **k):
        """Record the row count of each predict_proba call, then delegate."""
        seen.append(len(X))
        return real(self, X, *a, **k)

    monkeypatch.setattr(LogisticRegression, "predict_proba", _spy)
    (p1, p2, p3), _sigmas = pk._fit_3baselines_with_sigma(Xt, y_t, Xq, "binary", seed=0)
    assert len(p1) == len(p2) == len(p3) == Xq.shape[0]
    assert 80 not in seen
    assert 15 in seen
