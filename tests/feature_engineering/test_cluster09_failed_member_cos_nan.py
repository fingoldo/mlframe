"""Cluster 09 regression: a failed gradient member gives NaN cosines, not fake orthogonality (audit 2026-11-20, finding 09-F22).

A missing third member used to get a zero gradient, so its cosines read as 0 (orthogonal) and looked like real disagreement. Its cosines are now
NaN, and the mean and minimum use only the cosines that exist.
"""

import numpy as np
from unittest.mock import patch

from mlframe.feature_engineering.transformer import gradient_direction_agreement as gda


def test_failed_third_member_gives_nan_cosines():
    """With the linear member failing, cos23/cos13 are NaN and mean_cos is finite (09-F22)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + rng.normal(size=80) * 0.2 > 0).astype(int)
    Xq = rng.normal(size=(10, 3))
    with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=RuntimeError("no converge")):
        out = gda.compute_gradient_direction_agreement_features(X, y, Xq, None, seed=0, task="binary")
    arr = out.to_numpy()
    assert np.isnan(arr[:, 1]).all()
    assert np.isnan(arr[:, 2]).all()
    assert np.isfinite(arr[:, 3]).all()
