"""Tests for mlframe.training.train_eval.select_target.

select_target is a heavy orchestrator — we test only its lightweight pre-dispatch
logic: model_name decoration (binary imbalance percentage / regression mean) and
error handling for unsupported target types. configure_training_params is
monkeypatched to avoid a full sklearn/XGBoost/LightGBM config materialization.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

import mlframe.training.train_eval as te
from mlframe.training.train_eval import select_target
from mlframe.training.configs import TargetTypes


@pytest.fixture
def mock_configure(monkeypatch):
    """Stub configure_training_params so select_target returns without heavy work.

    Returns the list of captured (args, kwargs) to let tests assert what was passed.
    """
    captured = []

    def _stub(*args, **kwargs):
        captured.append((args, kwargs))
        # Mirror real 7-tuple shape
        return (
            dict(model_name=kwargs.get("model_name", "")),  # common_params
            {},  # models_params
            object(),  # cb_rfecv
            object(),  # lgb_rfecv
            object(),  # xgb_rfecv
            object(),  # cpu_configs
            object(),  # gpu_configs
        )

    monkeypatch.setattr(te, "configure_training_params", _stub)
    return captured


def test_select_target_regression_mean_suffix(mock_configure):
    df = pd.DataFrame({"x": np.arange(10)})
    target = pd.Series(np.arange(10, dtype=float))  # mean = 4.5
    common_params, *_ = select_target(
        model_name="base",
        target=target,
        target_type=TargetTypes.REGRESSION,
        df=df,
    )
    # Model name gets augmented with MT=<mean>
    passed_name = mock_configure[0][1]["model_name"]
    assert "MT=" in passed_name
    assert "4.5000" in passed_name


def test_select_target_binary_pct_pandas(mock_configure):
    df = pd.DataFrame({"x": np.arange(100)})
    # 30% positive
    y = pd.Series([1] * 30 + [0] * 70)
    select_target(
        model_name="m",
        target=y,
        target_type=TargetTypes.BINARY_CLASSIFICATION,
        df=df,
    )
    name = mock_configure[0][1]["model_name"]
    assert "BT=30%" in name


def test_select_target_binary_pct_numpy(mock_configure):
    df = pd.DataFrame({"x": np.arange(50)})
    y = np.array([1] * 10 + [0] * 40)
    select_target(
        model_name="m",
        target=y,
        target_type=TargetTypes.BINARY_CLASSIFICATION,
        df=df,
    )
    assert "BT=20%" in mock_configure[0][1]["model_name"]


def test_select_target_binary_pct_polars(mock_configure):
    df = pd.DataFrame({"x": np.arange(20)})
    y = pl.Series("t", [1] * 5 + [0] * 15)
    select_target(
        model_name="m",
        target=y,
        target_type=TargetTypes.BINARY_CLASSIFICATION,
        df=df,
    )
    assert "BT=25%" in mock_configure[0][1]["model_name"]


def test_select_target_binary_no_positives(mock_configure):
    df = pd.DataFrame({"x": np.arange(10)})
    y = pd.Series([0] * 10)
    select_target(
        model_name="m",
        target=y,
        target_type=TargetTypes.BINARY_CLASSIFICATION,
        df=df,
    )
    # When no positives, perc defaults to 0
    assert "BT=0%" in mock_configure[0][1]["model_name"]


def test_select_target_rejects_unsupported_type():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(TypeError, match="np.ndarray, pd.Series, or pl.Series"):
        select_target(
            model_name="m",
            target=[1, 0, 1],  # plain list not accepted for classification
            target_type=TargetTypes.BINARY_CLASSIFICATION,
            df=df,
        )
