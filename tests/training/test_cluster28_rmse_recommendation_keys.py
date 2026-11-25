"""Cluster 28 regression: every RMSE recommendation carries the same keys as the other branches (audit 2026-11-20, finding 28-F66)."""

import numpy as np

from mlframe.training.loss_recommendation import recommend_boosting_regression_loss


def test_rmse_branch_has_extra_params_keys():
    """The plain RMSE path exposes lgb/xgb/cb extra-params keys, so callers can index them (28-F66)."""
    y = np.random.default_rng(0).normal(size=500)
    rec = recommend_boosting_regression_loss(y)
    assert rec["cb"] == "RMSE"
    for key in ("lgb_extra_params", "xgb_extra_params", "cb_extra_params"):
        assert key in rec


def test_small_sample_rmse_branch_has_extra_params_keys():
    """The insufficient-sample RMSE fallback carries the same keys (28-F66)."""
    rec = recommend_boosting_regression_loss(np.array([1.0, 2.0, 3.0]))
    assert "xgb_extra_params" in rec and "cb_extra_params" in rec
