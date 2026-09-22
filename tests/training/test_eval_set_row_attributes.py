"""The single val carries its weights / query groups into every booster under the kwarg that booster reads."""

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.training._data_helpers import _setup_eval_set

lightgbm = pytest.importorskip("lightgbm")
xgboost = pytest.importorskip("xgboost")
catboost = pytest.importorskip("catboost")

_X = pd.DataFrame({"a": np.arange(8, dtype=float), "b": np.arange(8, dtype=float)[::-1]})
_Y = pd.Series([0, 1] * 4)
_W = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
_G = np.array([0, 0, 0, 1, 1, 2, 2, 2])


def _setup(category, model, **kw):
    fp = {}
    _setup_eval_set(type(model).__name__, fp, _X, _Y, model_obj=model, model_category=category, **kw)
    return fp


def test_xgboost_gets_val_weights_but_no_qid_on_a_classifier():
    fp = _setup("xgb", xgboost.XGBClassifier(), sample_weight_val=_W, group_ids_val=_G)
    assert fp["sample_weight_eval_set"][0] is _W
    assert "eval_qid" not in fp, "XGBClassifier.fit has no eval_qid; passing it would crash the fit"


def test_xgboost_ranker_gets_eval_qid():
    fp = _setup("xgb", xgboost.XGBRanker(), group_ids_val=_G)
    assert list(fp["eval_qid"][0]) == list(_G)


def test_lightgbm_reads_eval_sample_weight_not_the_xgboost_name():
    fp = _setup("lgb", lightgbm.LGBMClassifier(), sample_weight_val=_W, group_ids_val=_G)
    assert fp["eval_sample_weight"][0] is _W
    assert "sample_weight_eval_set" not in fp and "eval_group" not in fp


def test_lightgbm_ranker_gets_group_sizes():
    fp = _setup("lgb", lightgbm.LGBMRanker(), group_ids_val=_G)
    assert list(fp["eval_group"][0]) == [3, 2, 3]


def test_catboost_val_weights_land_on_the_eval_pool():
    from mlframe.training.cb._cb_eval_weights import apply_cb_eval_sample_weights

    fp = _setup("cb", catboost.CatBoostClassifier(), sample_weight_val=_W)
    assert "sample_weight_eval_set" not in fp, "CatBoost.fit has no such kwarg"
    apply_cb_eval_sample_weights(fp)
    (pool,) = fp["eval_set"]
    assert isinstance(pool, catboost.Pool)
    assert np.allclose(pool.get_weight(), _W)
    model = catboost.CatBoostClassifier(iterations=5, verbose=False).fit(_X, _Y, **fp)
    assert model.tree_count_ >= 1


def test_a_reused_catboost_pool_is_reset_when_the_next_fit_is_unweighted():
    from mlframe.training.cb._cb_eval_weights import CB_EVAL_WEIGHTS_KEY, apply_cb_eval_sample_weights

    pool = catboost.Pool(_X, _Y)
    apply_cb_eval_sample_weights({"eval_set": [pool], CB_EVAL_WEIGHTS_KEY: [_W]})
    assert np.allclose(pool.get_weight(), _W)
    apply_cb_eval_sample_weights({"eval_set": [pool]})
    assert np.allclose(pool.get_weight(), 1.0), "the cached val Pool kept the previous fit's weights"


def test_uniform_weights_leave_the_catboost_eval_set_untouched():
    from mlframe.training.cb._cb_eval_weights import CB_EVAL_WEIGHTS_KEY, apply_cb_eval_sample_weights

    fp = {"eval_set": [(_X, _Y)], CB_EVAL_WEIGHTS_KEY: [np.ones(8)]}
    apply_cb_eval_sample_weights(fp)
    assert fp["eval_set"] == [(_X, _Y)] and CB_EVAL_WEIGHTS_KEY not in fp


def test_an_unrecognised_model_name_warns_that_it_trains_without_early_stopping(caplog):
    fp = {}
    with caplog.at_level(logging.WARNING, logger="mlframe.training._data_helpers"):
        _setup_eval_set("MyBoostWrapper", fp, _X, _Y)
    assert fp == {}
    assert any("WITHOUT an eval set" in r.getMessage() for r in caplog.records)
