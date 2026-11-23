"""Cluster 29 regression: a refit without iteration-metric capture does not keep the previous fit's metrics (audit 2026-11-20, finding 29-F37)."""

import numpy as np
import pytest

pytest.importorskip("xgboost")
from mlframe.training import xgb_shim as shim


def _data(seed):
    """Small binary dataset for a quick fit."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_refit_without_capture_clears_previous_metrics():
    """A second fit with capture off must not report the first fit's iteration_metrics_ (29-F37)."""
    cls = shim.XGBClassifierWithDMatrixReuse
    X, y = _data(0)
    est = cls(n_estimators=5)
    est.fit(X, y, eval_set=[(X, y)], capture_iteration_metrics=True)
    assert hasattr(est, "iteration_metrics_")
    est.fit(X, y, eval_set=[(X, y)], capture_iteration_metrics=False)
    assert not hasattr(est, "iteration_metrics_")
