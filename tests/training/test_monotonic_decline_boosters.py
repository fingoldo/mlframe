"""Monotonic strict-decline overfitting stop for the LightGBM + XGBoost shims.

Unit: the LGB / XGB callbacks fire on a synthetic monotone-worsening eval series.
Integration: a real lgb / xgb fit on an overfit-prone target stops with FEWER trees than the
no-monotonic baseline, at the same-or-better holdout RMSE.
"""
from __future__ import annotations

import numpy as np
import pytest


# --------------------------------------------------------------------------- unit


def test_lgb_callback_fires_on_monotone_worsening():
    lgb = pytest.importorskip("lightgbm")
    from mlframe.training.callbacks.monotonic_decline import LGBMonotonicDeclineStop

    cb = LGBMonotonicDeclineStop(patience=3, monitor_dataset="valid_0", monitor_metric="l2", mode="min")

    def _env(it, value):
        return type("E", (), {"iteration": it, "evaluation_result_list": [("valid_0", "l2", value, False)]})()

    # improving then 3 strict rises -> EarlyStopException
    cb(_env(0, 0.5))
    cb(_env(1, 0.3))   # best
    cb(_env(2, 0.32))
    cb(_env(3, 0.34))
    with pytest.raises(lgb.callback.EarlyStopException):
        cb(_env(4, 0.36))


def test_xgb_callback_fires_on_monotone_worsening():
    pytest.importorskip("xgboost")
    from mlframe.training.callbacks.monotonic_decline import _make_xgb_monotonic_callback

    cb = _make_xgb_monotonic_callback(patience=3, monitor_dataset="validation_0",
                                      monitor_metric="rmse", mode="min")
    assert cb is not None

    def _log(value):
        return {"validation_0": {"rmse": [value]}}

    assert cb.after_iteration(None, 0, _log(0.5)) is False
    assert cb.after_iteration(None, 1, _log(0.3)) is False  # best
    assert cb.after_iteration(None, 2, _log(0.32)) is False
    assert cb.after_iteration(None, 3, _log(0.34)) is False
    assert cb.after_iteration(None, 4, _log(0.36)) is True  # 3rd strict rise -> stop


def test_xgb_callback_disabled_returns_none():
    pytest.importorskip("xgboost")
    from mlframe.training.callbacks.monotonic_decline import _make_xgb_monotonic_callback

    assert _make_xgb_monotonic_callback(patience=None) is None


# --------------------------------------------------------------------------- integration


def _overfit_data(seed=0, n=600, d=20):
    """Mostly-noise target: extra trees overfit train, val turns up -> monotone-worsening tail."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (0.5 * X[:, 0] + rng.randn(n) * 1.4).astype(np.float64)
    return X[:-150], y[:-150], X[-150:], y[-150:]


def _rmse(yt, yp):
    return float(np.sqrt(np.mean((np.asarray(yp).reshape(-1) - yt) ** 2)))


def _n_trees_lgb(booster):
    return booster.num_trees()


def test_lgb_shim_monotonic_stops_early():
    pytest.importorskip("lightgbm")
    from mlframe.training.lgb_shim import LGBMRegressorWithDatasetReuse

    Xtr, ytr, Xv, yv = _overfit_data()

    def _fit(mono):
        m = LGBMRegressorWithDatasetReuse(
            n_estimators=400, learning_rate=0.1, num_leaves=63, min_child_samples=5,
            verbose=-1, random_state=0,
        )
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric="l2",
              monotonic_decline_patience=mono)
        return m

    m_mono = _fit(3)
    m_none = _fit(None)
    n_mono = _n_trees_lgb(m_mono.booster_)
    n_none = _n_trees_lgb(m_none.booster_)
    assert n_mono < n_none, f"lgb monotonic stop grew {n_mono} trees, baseline {n_none}"
    assert _rmse(yv, m_mono.predict(Xv)) <= _rmse(yv, m_none.predict(Xv)) + 0.05


def test_xgb_shim_monotonic_stops_early():
    pytest.importorskip("xgboost")
    from mlframe.training.xgb_shim import XGBRegressorWithDMatrixReuse

    Xtr, ytr, Xv, yv = _overfit_data()

    def _fit(mono):
        m = XGBRegressorWithDMatrixReuse(
            n_estimators=400, learning_rate=0.1, max_depth=8, min_child_weight=1,
            eval_metric="rmse", random_state=0,
        )
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], monotonic_decline_patience=mono)
        return m

    m_mono = _fit(3)
    m_none = _fit(None)
    n_mono = m_mono.get_booster().num_boosted_rounds()
    n_none = m_none.get_booster().num_boosted_rounds()
    assert n_mono < n_none, f"xgb monotonic stop grew {n_mono} rounds, baseline {n_none}"
    assert _rmse(yv, m_mono.predict(Xv)) <= _rmse(yv, m_none.predict(Xv)) + 0.05
