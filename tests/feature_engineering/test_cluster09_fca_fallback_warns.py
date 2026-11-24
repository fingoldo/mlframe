"""Cluster 09 regression: an FCA lattice failure is logged at WARNING (audit 2026-11-20, finding 09-F16).

The fallback sets every concept feature to zero, which downstream reads as "no concept matched". A failure is now a WARNING, not INFO.
"""

import logging
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import fca_closed_concepts as fca


def test_lattice_failure_logs_warning(caplog):
    """A failing concept lattice gives zero features and a WARNING (09-F16)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    y = X[:, 0] + rng.normal(size=40) * 0.1
    with patch("concepts.Context", side_effect=RuntimeError("boom")), caplog.at_level(logging.WARNING):
        out = fca.compute_fca_closed_concepts_features(X, y, None, KFold(3, shuffle=True, random_state=0), seed=0, task="regression", top_k=4)
    assert np.all(out.to_numpy()[:, :4] == 0.0)
    assert any(r.levelno == logging.WARNING and "lattice construction failed" in r.getMessage() for r in caplog.records)
