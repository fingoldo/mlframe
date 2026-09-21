"""The FI chart names what its numbers are instead of a bare "Importance"."""

from __future__ import annotations

import pytest

from mlframe.training._feature_importances import describe_importance_kind


def test_labels_per_backend():
    cb = pytest.importorskip("catboost")
    lgb = pytest.importorskip("lightgbm")
    assert "PredictionValuesChange" in describe_importance_kind(cb.CatBoostRegressor())
    assert "LightGBM split" in describe_importance_kind(lgb.LGBMRegressor())
    assert "permutation" in describe_importance_kind(object(), importances_std=[0.1])
