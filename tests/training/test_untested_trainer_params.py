"""Tests for model-param configuration helpers in mlframe.training.trainer.

Targets:
- _configure_xgboost_params
- _configure_lightgbm_params
- configure_training_params (smoke — only assert it returns a 7-tuple
  with expected shape; no actual training)

NOTE: the audit asked for _configure_catboost_params, but no such helper exists
at module level — CatBoost parameterization is done inline in
configure_training_params. Skipped with a note rather than faked.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from mlframe.training.trainer import (
    _configure_xgboost_params,
    _configure_lightgbm_params,
    configure_training_params,
)
from mlframe.training.helpers import get_training_configs


@pytest.fixture(scope="module")
def real_configs():
    cpu = get_training_configs(iterations=10, has_gpu=False)
    # has_gpu=None lets the helper decide; for tests we want deterministic CPU
    gpu = get_training_configs(iterations=10, has_gpu=False)
    return cpu, gpu


def _identity(x):
    return x


# ----- _configure_xgboost_params -----

def test_xgb_regression_cpu(real_configs):
    cpu, gpu = real_configs
    out = _configure_xgboost_params(
        configs=gpu, cpu_configs=cpu,
        use_regression=True,
        prefer_cpu_for_xgboost=True,
        prefer_calibrated_classifiers=False,
        use_flaml_zeroshot=False,
        xgboost_verbose=False,
        metamodel_func=_identity,
    )
    assert "model" in out
    assert "fit_params" in out
    assert out["fit_params"] == {"verbose": False}
    # Regressor type
    from xgboost import XGBRegressor
    assert isinstance(out["model"], XGBRegressor)


def test_xgb_classification_calibrated(real_configs):
    cpu, gpu = real_configs
    out = _configure_xgboost_params(
        configs=gpu, cpu_configs=cpu,
        use_regression=False,
        prefer_cpu_for_xgboost=True,
        prefer_calibrated_classifiers=True,
        use_flaml_zeroshot=False,
        xgboost_verbose=0,
        metamodel_func=_identity,
    )
    from xgboost import XGBClassifier
    assert isinstance(out["model"], XGBClassifier)


def test_xgb_classification_non_calibrated(real_configs):
    cpu, gpu = real_configs
    out = _configure_xgboost_params(
        configs=gpu, cpu_configs=cpu,
        use_regression=False,
        prefer_cpu_for_xgboost=False,  # try GPU config path
        prefer_calibrated_classifiers=False,
        use_flaml_zeroshot=False,
        xgboost_verbose=False,
        metamodel_func=_identity,
    )
    from xgboost import XGBClassifier
    assert isinstance(out["model"], XGBClassifier)


# ----- _configure_lightgbm_params -----

def test_lgb_regression(real_configs):
    cpu, gpu = real_configs
    out = _configure_lightgbm_params(
        configs=gpu, cpu_configs=cpu,
        use_regression=True,
        prefer_cpu_for_lightgbm=True,
        prefer_calibrated_classifiers=False,
        use_flaml_zeroshot=False,
        metamodel_func=_identity,
    )
    assert "model" in out
    assert out["fit_params"] == {}
    from lightgbm import LGBMRegressor
    assert isinstance(out["model"], LGBMRegressor)


def test_lgb_classification_calibrated_adds_eval_metric(real_configs):
    cpu, gpu = real_configs
    out = _configure_lightgbm_params(
        configs=gpu, cpu_configs=cpu,
        use_regression=False,
        prefer_cpu_for_lightgbm=True,
        prefer_calibrated_classifiers=True,
        use_flaml_zeroshot=False,
        metamodel_func=_identity,
    )
    assert "eval_metric" in out["fit_params"]
    from lightgbm import LGBMClassifier
    assert isinstance(out["model"], LGBMClassifier)


def test_lgb_classification_non_calibrated(real_configs):
    cpu, gpu = real_configs
    out = _configure_lightgbm_params(
        configs=gpu, cpu_configs=cpu,
        use_regression=False,
        prefer_cpu_for_lightgbm=True,
        prefer_calibrated_classifiers=False,
        use_flaml_zeroshot=False,
        metamodel_func=_identity,
    )
    assert out["fit_params"] == {}


# ----- configure_training_params smoke -----

def test_configure_training_params_returns_7tuple_regression():
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        "f0": rng.standard_normal(n),
        "f1": rng.standard_normal(n),
    })
    target = pd.Series(rng.standard_normal(n))
    out = configure_training_params(
        df=df,
        train_df=df.iloc[:30],
        val_df=df.iloc[30:40],
        test_df=df.iloc[40:],
        target=target,
        train_target=target.iloc[:30],
        val_target=target.iloc[30:40],
        test_target=target.iloc[40:],
        train_idx=np.arange(30),
        val_idx=np.arange(30, 40),
        test_idx=np.arange(40, n),
        use_regression=True,
        prefer_gpu_configs=False,
        mlframe_models=["lgb"],  # only build one to keep test fast
        verbose=False,
        config_params={"iterations": 10},
    )
    assert isinstance(out, tuple)
    assert len(out) == 7
    common_params, models_params, cb_rfecv, lgb_rfecv, xgb_rfecv, cpu_configs, gpu_configs = out
    assert isinstance(common_params, dict)
    assert isinstance(models_params, dict)
    # Model building filtered: only lgb requested
    assert "lgb" in models_params


def test_configure_training_params_binary_classification():
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({"f": rng.standard_normal(n)})
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    out = configure_training_params(
        df=df, train_df=df.iloc[:40], val_df=df.iloc[40:50], test_df=df.iloc[50:],
        target=y, train_target=y.iloc[:40], val_target=y.iloc[40:50], test_target=y.iloc[50:],
        train_idx=np.arange(40), val_idx=np.arange(40, 50), test_idx=np.arange(50, n),
        use_regression=False,
        prefer_gpu_configs=False,
        mlframe_models=["xgb"],
        verbose=False,
        config_params={"iterations": 10},
    )
    assert len(out) == 7
    models_params = out[1]
    assert "xgb" in models_params
