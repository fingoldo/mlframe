"""Cluster 09 regression: a non-converged NCA fit is logged at WARNING (audit 2026-11-20, finding 09-F18).

The warnings around ``nca.fit`` were all suppressed, so an unconverged projection looked like a valid one. A ConvergenceWarning is now logged.
"""

import logging
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning

from mlframe.feature_engineering.transformer import nca_projection as nca


def test_convergence_warning_is_logged(monkeypatch, caplog):
    """A fit that emits ConvergenceWarning produces a WARNING log naming max_iter (09-F18)."""
    class _Stub:
        """Stand-in NCA whose fit emits a ConvergenceWarning."""

        def __init__(self, **kwargs):
            """Accept and ignore the NCA constructor arguments."""

        def fit(self, X, y):
            """Warn that the fit did not converge."""
            warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
            return self

    import sklearn.neighbors

    monkeypatch.setattr(sklearn.neighbors, "NeighborhoodComponentsAnalysis", _Stub)
    X = np.random.default_rng(0).normal(size=(30, 3))
    y = (X[:, 0] > 0).astype(int)
    with caplog.at_level(logging.WARNING):
        nca._fit_nca(X, y, n_components=2, max_iter=50, seed=0)
    assert any(r.levelno == logging.WARNING and "did not converge within max_iter=50" in r.getMessage() for r in caplog.records)
