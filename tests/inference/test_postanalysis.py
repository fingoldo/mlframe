"""Smoke test for mlframe.inference.postanalysis (W5-4).

Note: the audit asked for compute_residuals coverage, but the module currently only exposes
analyze_xgboost_model. We test what actually exists; compute_residuals can be added later
once the function lands.
"""

from __future__ import annotations

import pytest


@pytest.mark.fast
def test_analyze_xgboost_model_smoke():
    """analyze_xgboost_model returns expected summary dict on a tiny XGBoost regressor."""
    xgb = pytest.importorskip("xgboost")
    np = pytest.importorskip("numpy")
    from mlframe.inference.postanalysis import analyze_xgboost_model

    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=50)
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, tree_method="hist", verbosity=0)
    model.fit(X, y)

    summary = analyze_xgboost_model(model)
    assert isinstance(summary, dict)
    assert summary["total_trees"] == 3
    assert summary["total_leaves"] >= 3  # at least one leaf per tree
    assert summary["max_tree_leaves"] >= 1
    assert summary["model_desc_size_gb"] >= 0.0
