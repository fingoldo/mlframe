"""Cluster 24 regression: a linear member wrapped in a Pipeline votes with |coef_| of its final step (audit 2026-11-20, finding 24-F43).

The module docstring promises |coef_| for linear members, but a Pipeline has no coef_ of its own, so the member fell back to permutation
importance on a subsample. The importance must now equal the absolute coefficients of the fitted final estimator.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.hetero_vote import _importance


def test_pipeline_linear_member_uses_absolute_coefficients():
    """The pipeline's importance equals |coef_| of its fitted LogisticRegression, with no permutation step (24-F43)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] - 0.5 * X[:, 2] + 0.2 * rng.normal(size=300) > 0).astype(int)
    est = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    imp = _importance(est, X, y)

    fitted = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    expected = np.abs(fitted.steps[-1][1].coef_).max(axis=0)
    np.testing.assert_allclose(imp, expected, rtol=1e-9)
