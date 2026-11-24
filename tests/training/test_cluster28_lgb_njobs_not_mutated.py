"""Cluster 28 regression: fitting the LightGBM shim leaves the caller's n_jobs unchanged, so get_params() and clone() match the config (audit 2026-11-20, finding 28-F58).

The resolved CPU count is used while building the fit params only; n_jobs is restored afterwards. Checked on a small regressor fit: a None
n_jobs stays None.
"""

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.training import lgb_shim as shim


def test_fit_does_not_overwrite_caller_n_jobs():
    """n_jobs is still None after fit, so get_params reports the caller's config (28-F58)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X[:, 0] * 2 + rng.normal(size=60) * 0.1
    model = shim.LGBMRegressorWithDatasetReuse(n_estimators=5, n_jobs=None, verbose=-1)
    model.fit(X, y)
    assert model.get_params()["n_jobs"] is None
