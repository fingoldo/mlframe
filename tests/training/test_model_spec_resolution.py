"""Tests for estimator-object / tuple model-spec resolution."""

from __future__ import annotations

import logging

import pytest
from sklearn.dummy import DummyClassifier

from mlframe.training.strategies import (
    CatBoostStrategy,
    HGBStrategy,
    LinearModelStrategy,
    TreeModelStrategy,
    XGBoostStrategy,
    _resolve_model_spec,
    get_strategy,
)


def test_resolve_string_alias_unchanged():
    key, est, strat = _resolve_model_spec("cb")
    assert key == "cb"
    assert est is None
    assert isinstance(strat, CatBoostStrategy)


def test_resolve_string_linear():
    key, est, strat = _resolve_model_spec("ridge")
    assert key == "ridge"
    assert est is None
    assert isinstance(strat, LinearModelStrategy)


def test_resolve_lgbm_instance_key_is_class_name():
    lightgbm = pytest.importorskip("lightgbm")
    est = lightgbm.LGBMClassifier()
    key, resolved_est, strat = _resolve_model_spec(est)
    assert key == "LGBMClassifier"
    assert resolved_est is est
    # LightGBM dispatches to the generic TreeModelStrategy.
    assert isinstance(strat, TreeModelStrategy)


def test_resolve_tuple_uses_explicit_name():
    lightgbm = pytest.importorskip("lightgbm")
    est = lightgbm.LGBMClassifier()
    key, resolved_est, strat = _resolve_model_spec(("my_model", est))
    assert key == "my_model"
    assert resolved_est is est
    assert isinstance(strat, TreeModelStrategy)


def test_resolve_catboost_instance():
    catboost = pytest.importorskip("catboost")
    est = catboost.CatBoostClassifier(verbose=0)
    key, resolved_est, strat = _resolve_model_spec(est)
    assert key == "CatBoostClassifier"
    assert isinstance(strat, CatBoostStrategy)


def test_resolve_xgboost_instance():
    xgboost = pytest.importorskip("xgboost")
    est = xgboost.XGBClassifier()
    key, _, strat = _resolve_model_spec(est)
    assert key == "XGBClassifier"
    assert isinstance(strat, XGBoostStrategy)


def test_resolve_hgb_instance():
    from sklearn.ensemble import HistGradientBoostingRegressor
    est = HistGradientBoostingRegressor()
    key, _, strat = _resolve_model_spec(est)
    assert key == "HistGradientBoostingRegressor"
    assert isinstance(strat, HGBStrategy)


def test_resolve_unknown_estimator_falls_back_linear_with_warning(caplog):
    est = DummyClassifier()
    with caplog.at_level(logging.WARNING, logger="mlframe.training.strategies"):
        key, resolved_est, strat = _resolve_model_spec(est)
    assert key == "DummyClassifier"
    assert resolved_est is est
    assert isinstance(strat, LinearModelStrategy)
    messages = [r.getMessage() for r in caplog.records]
    assert any("No registered strategy" in m and "DummyClassifier" in m for m in messages)


def test_duplicate_keys_get_suffixed():
    lightgbm = pytest.importorskip("lightgbm")
    used = set()
    k1, _, _ = _resolve_model_spec(lightgbm.LGBMClassifier(), used_keys=used)
    k2, _, _ = _resolve_model_spec(lightgbm.LGBMClassifier(), used_keys=used)
    k3, _, _ = _resolve_model_spec(lightgbm.LGBMClassifier(), used_keys=used)
    assert k1 == "LGBMClassifier"
    assert k2 == "LGBMClassifier_2"
    assert k3 == "LGBMClassifier_3"


def test_tuple_name_is_slugified():
    from sklearn.linear_model import Ridge
    used = set()
    k1, _, _ = _resolve_model_spec(("my model v1!", Ridge()), used_keys=used)
    # spaces and exclamation collapse to underscores.
    assert k1 == "my_model_v1"
    # Collision handling on slugified duplicates.
    k2, _, _ = _resolve_model_spec(("my model v1!", Ridge()), used_keys=used)
    assert k2 == "my_model_v1_2"


def test_tuple_with_non_string_name_raises():
    from sklearn.linear_model import Ridge
    with pytest.raises(TypeError):
        _resolve_model_spec((123, Ridge()))


def test_get_strategy_accepts_estimator_instance():
    lightgbm = pytest.importorskip("lightgbm")
    strat = get_strategy(lightgbm.LGBMRegressor())
    assert isinstance(strat, TreeModelStrategy)


def test_get_strategy_accepts_tuple():
    from sklearn.linear_model import Ridge
    strat = get_strategy(("custom", Ridge()))
    assert isinstance(strat, LinearModelStrategy)


def test_mixed_list_resolution_preserves_distinct_keys():
    """Heterogeneous list: string + instance + tuple + dup-class."""
    lightgbm = pytest.importorskip("lightgbm")
    entries = [
        "cb",
        lightgbm.LGBMClassifier(),
        ("my_lgb_v1", lightgbm.LGBMClassifier()),
        lightgbm.LGBMClassifier(),  # duplicate class → _2 suffix
    ]
    used = set()
    keys = [_resolve_model_spec(e, used_keys=used)[0] for e in entries]
    assert keys == ["cb", "LGBMClassifier", "my_lgb_v1", "LGBMClassifier_2"]
