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
    """Lgb callback fires on monotone worsening."""
    lgb = pytest.importorskip("lightgbm")
    from mlframe.training.callbacks.monotonic_decline import LGBMonotonicDeclineStop

    cb = LGBMonotonicDeclineStop(patience=3, monitor_dataset="valid_0", monitor_metric="l2", mode="min")

    def _env(it, value):
        """Env."""
        return type("E", (), {"iteration": it, "evaluation_result_list": [("valid_0", "l2", value, False)]})()

    # improving then 3 strict rises -> EarlyStopException
    cb(_env(0, 0.5))
    cb(_env(1, 0.3))  # best
    cb(_env(2, 0.32))
    cb(_env(3, 0.34))
    with pytest.raises(lgb.callback.EarlyStopException):
        cb(_env(4, 0.36))


def test_xgb_callback_fires_on_monotone_worsening():
    """Xgb callback fires on monotone worsening."""
    pytest.importorskip("xgboost")
    from mlframe.training.callbacks.monotonic_decline import _make_xgb_monotonic_callback

    cb = _make_xgb_monotonic_callback(patience=3, monitor_dataset="validation_0", monitor_metric="rmse", mode="min")
    assert cb is not None

    def _log(value):
        """Log."""
        return {"validation_0": {"rmse": [value]}}

    assert cb.after_iteration(None, 0, _log(0.5)) is False
    assert cb.after_iteration(None, 1, _log(0.3)) is False  # best
    assert cb.after_iteration(None, 2, _log(0.32)) is False
    assert cb.after_iteration(None, 3, _log(0.34)) is False
    assert cb.after_iteration(None, 4, _log(0.36)) is True  # 3rd strict rise -> stop


def test_xgb_callback_disabled_returns_none():
    """Xgb callback disabled returns none."""
    pytest.importorskip("xgboost")
    from mlframe.training.callbacks.monotonic_decline import _make_xgb_monotonic_callback

    assert _make_xgb_monotonic_callback(patience=None) is None


def test_cb_callback_fires_on_monotone_worsening():
    """Cb callback fires on monotone worsening."""
    from mlframe.training.callbacks.monotonic_decline import CBMonotonicDeclineStop

    cb = CBMonotonicDeclineStop(patience=3, monitor_dataset="validation", monitor_metric="RMSE", mode="min")

    def _info(value):
        """Info."""
        return type("I", (), {"metrics": {"validation": {"RMSE": [value]}}})()

    # CatBoost convention: after_iteration returns True to continue, False to stop.
    assert cb.after_iteration(_info(0.5)) is True
    assert cb.after_iteration(_info(0.3)) is True  # best
    assert cb.after_iteration(_info(0.32)) is True
    assert cb.after_iteration(_info(0.34)) is True
    assert cb.after_iteration(_info(0.36)) is False  # 3rd strict rise -> stop


def test_cb_callback_disabled_always_continues():
    """Cb callback disabled always continues."""
    from mlframe.training.callbacks.monotonic_decline import CBMonotonicDeclineStop

    cb = CBMonotonicDeclineStop(patience=None, monitor_dataset="validation", monitor_metric="RMSE", mode="min")

    def _info(value):
        """Info."""
        return type("I", (), {"metrics": {"validation": {"RMSE": [value]}}})()

    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        assert cb.after_iteration(_info(v)) is True


# --------------------------------------------------------------------------- integration


def _overfit_data(seed=0, n=600, d=20):
    """Mostly-noise target: extra trees overfit train, val turns up -> monotone-worsening tail."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float64)
    y = (0.5 * X[:, 0] + rng.randn(n) * 1.4).astype(np.float64)
    return X[:-150], y[:-150], X[-150:], y[-150:]


def _rmse(yt, yp):
    """Rmse."""
    return float(np.sqrt(np.mean((np.asarray(yp).reshape(-1) - yt) ** 2)))


def _n_trees_lgb(booster):
    """N trees lgb."""
    return booster.num_trees()


def test_lgb_shim_monotonic_stops_early():
    """Lgb shim monotonic stops early."""
    pytest.importorskip("lightgbm")
    from mlframe.training.lgb_shim import LGBMRegressorWithDatasetReuse

    Xtr, ytr, Xv, yv = _overfit_data()

    def _fit(mono):
        """Fit."""
        m = LGBMRegressorWithDatasetReuse(
            n_estimators=400,
            learning_rate=0.1,
            num_leaves=63,
            min_child_samples=5,
            verbose=-1,
            random_state=0,
        )
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], eval_metric="l2", monotonic_decline_patience=mono)
        return m

    m_mono = _fit(3)
    m_none = _fit(None)
    n_mono = _n_trees_lgb(m_mono.booster_)
    n_none = _n_trees_lgb(m_none.booster_)
    assert n_mono < n_none, f"lgb monotonic stop grew {n_mono} trees, baseline {n_none}"
    assert _rmse(yv, m_mono.predict(Xv)) <= _rmse(yv, m_none.predict(Xv)) + 0.05


def test_xgb_shim_monotonic_stops_early():
    """Xgb shim monotonic stops early."""
    pytest.importorskip("xgboost")
    from mlframe.training.xgb_shim import XGBRegressorWithDMatrixReuse

    Xtr, ytr, Xv, yv = _overfit_data()

    def _fit(mono):
        """Fit."""
        m = XGBRegressorWithDMatrixReuse(
            n_estimators=400,
            learning_rate=0.1,
            max_depth=8,
            min_child_weight=1,
            eval_metric="rmse",
            random_state=0,
        )
        m.fit(Xtr, ytr, eval_set=[(Xv, yv)], monotonic_decline_patience=mono)
        return m

    m_mono = _fit(3)
    m_none = _fit(None)
    n_mono = m_mono.get_booster().num_boosted_rounds()
    n_none = m_none.get_booster().num_boosted_rounds()
    assert n_mono < n_none, f"xgb monotonic stop grew {n_mono} rounds, baseline {n_none}"
    assert _rmse(yv, m_mono.predict(Xv)) <= _rmse(yv, m_none.predict(Xv)) + 0.05


# --------------------------------------------------------------------------- D1: XGB best-iteration rollback


def test_xgb_callback_stamps_best_iteration_on_stop():
    """On stop the callback must stamp ``best_iteration`` on the booster so predict truncates to the
    pre-decline best, NOT score with the full overfit-tail booster."""
    pytest.importorskip("xgboost")
    from mlframe.training.callbacks.monotonic_decline import _make_xgb_monotonic_callback

    cb = _make_xgb_monotonic_callback(patience=3, monitor_dataset="validation_0", monitor_metric="rmse", mode="min")

    class _FakeBooster:
        """Groups tests covering fake booster."""
        def __init__(self):
            self.attrs = {}

        def set_attr(self, **kw):
            """Set attr."""
            self.attrs.update({k: v for k, v in kw.items()})

    model = _FakeBooster()

    def _log(value):
        """Log."""
        return {"validation_0": {"rmse": [value]}}

    cb.after_iteration(model, 0, _log(0.5))
    cb.after_iteration(model, 1, _log(0.3))  # best @1
    cb.after_iteration(model, 2, _log(0.32))
    cb.after_iteration(model, 3, _log(0.34))
    assert cb.after_iteration(model, 4, _log(0.36)) is True
    assert model.attrs.get("best_iteration") == "1", model.attrs


def test_biz_xgb_monotonic_predict_uses_best_iteration_not_full_booster():
    """Biz_value: XGB monotonic-stop holdout RMSE must equal the BEST-iteration prediction, not the
    full overfit booster. Regression sensor for D1 -- pre-fix the booster keeps the post-best tail and
    predict scores all rounds, inflating holdout error."""
    pytest.importorskip("xgboost")
    from mlframe.training.xgb_shim import XGBRegressorWithDMatrixReuse

    Xtr, ytr, Xv, yv = _overfit_data(seed=1)
    m = XGBRegressorWithDMatrixReuse(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=8,
        min_child_weight=1,
        eval_metric="rmse",
        random_state=0,
    )
    m.fit(Xtr, ytr, eval_set=[(Xv, yv)], monotonic_decline_patience=3)

    booster = m.get_booster()
    best_iter = booster.best_iteration  # raises if not stamped -> proves D1 wired
    n_rounds = booster.num_boosted_rounds()
    assert best_iter + 1 < n_rounds, "best_iter should be strictly before the last round (overfit tail kept)"

    rmse_default = _rmse(yv, m.predict(Xv))
    rmse_full = _rmse(yv, m.predict(Xv, iteration_range=(0, n_rounds)))
    rmse_best = _rmse(yv, m.predict(Xv, iteration_range=(0, best_iter + 1)))
    # Default predict must match best-iteration prediction (the fix), and be no worse than the full booster.
    assert abs(rmse_default - rmse_best) < 1e-9, (rmse_default, rmse_best)
    assert rmse_default <= rmse_full + 1e-9, (rmse_default, rmse_full)


# --------------------------------------------------------------------------- D5: unknown-metric SKIP


def test_resolve_mode_unknown_metric_returns_skip_sentinel():
    """Resolve mode unknown metric returns skip sentinel."""
    from mlframe.training.callbacks.monotonic_decline import _resolve_mode, _UNKNOWN_DIRECTION

    assert _resolve_mode("my_totally_custom_metric", None) == _UNKNOWN_DIRECTION
    # Explicit mode= always wins, even on an unknown name.
    assert _resolve_mode("my_totally_custom_metric", "max") == "max"
    # Known names still resolve.
    assert _resolve_mode("rmse", None) == "min"
    assert _resolve_mode("auc", None) == "max"


def test_lgb_callback_unknown_max_metric_does_not_stop_improving_curve():
    """An unknown higher-is-better metric must NOT be guessed as 'min' (which would stop a clearly
    IMPROVING max-metric curve). With the SKIP sentinel the detector disables itself: no stop."""
    pytest.importorskip("lightgbm")
    from mlframe.training.callbacks.monotonic_decline import LGBMonotonicDeclineStop

    cb = LGBMonotonicDeclineStop(patience=3, monitor_dataset="valid_0", monitor_metric="my_custom_skill_score", mode=None)

    def _env(it, value):
        """Env."""
        return type("E", (), {"iteration": it, "evaluation_result_list": [("valid_0", "my_custom_skill_score", value, True)]})()

    # Strictly improving (higher-is-better) curve. Under the old 'min' guess every step looks like a
    # decline -> would stop at iter 3. With SKIP the detector never fires.
    for it, v in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]):
        cb(_env(it, v))  # must not raise EarlyStopException


def test_cb_callback_unknown_max_metric_does_not_stop_improving_curve():
    """Cb callback unknown max metric does not stop improving curve."""
    from mlframe.training.callbacks.monotonic_decline import CBMonotonicDeclineStop

    cb = CBMonotonicDeclineStop(patience=3, monitor_dataset="validation", monitor_metric="my_custom_skill_score", mode=None)

    def _info(value):
        """Info."""
        return type("I", (), {"metrics": {"validation": {"my_custom_skill_score": [value]}}})()

    for v in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        assert cb.after_iteration(_info(v)) is True  # never stops


# --------------------------------------------------------------------------- D4: no false-stop on noisy-but-improving val


def test_monotonic_n3_does_not_false_stop_on_realistic_noisy_improving_curve():
    """Pinned baseline for a future noise-gate. A genuinely-improving max-metric val curve whose noise
    produces unlucky down-ticks -- but NOT 3-in-a-row, since each dip bounces back the next step -- must
    NOT be stopped by fixed-N=3 before the true best. This is the realistic noise shape (a down-tick is
    almost always followed by a bounce, which resets the streak). If this FAILS, N=3 is too aggressive on
    realistic noise: report loudly, do not mask.

    NOTE (D4 verdict): on the PATHOLOGICAL shape of 3 STRICTLY-consecutive declines, N=3 *does* false-stop
    before a later recovery -- see ``test_monotonic_n3_false_stops_on_three_consecutive_declines`` which
    pins that current (accepted-for-now) behaviour. The adaptive noise-gate that would fix it is deferred."""
    from mlframe.estimators.early_stopping_monotonic import MonotonicDeclineStopper

    s = MonotonicDeclineStopper(3, mode="max")
    # Improving trend with single-step dips, each followed by a bounce-up (resets the streak): never 3 strict declines in a row.
    curve = [0.50, 0.58, 0.55, 0.62, 0.60, 0.66, 0.64, 0.70, 0.68, 0.74]
    for i, v in enumerate(curve):
        assert not s.update(v), f"N=3 false-stopped at idx {i} on a noisy-but-improving curve (no 3 consecutive declines)"
    assert s.best == 0.74


def test_monotonic_n3_false_stops_on_three_consecutive_declines():
    """D4 honest finding (NOT masked): 3 STRICTLY-consecutive declines DO trip fixed-N=3 even when the
    curve later recovers past the best. Pins the current aggressive behaviour so a future noise-gate
    change has a baseline to move; the recovery at the tail is never reached because the stop fires first."""
    from mlframe.estimators.early_stopping_monotonic import MonotonicDeclineStopper

    s = MonotonicDeclineStopper(3, mode="max")
    curve = [0.50, 0.60, 0.70, 0.69, 0.68, 0.67, 0.72, 0.75]
    #                  best^   <-- 3 strict consecutive declines --> ^never reached (stop fired)
    stopped_at = next((i for i, v in enumerate(curve) if s.update(v)), None)
    assert stopped_at == 5, f"N=3 fires on the 3rd consecutive strict decline (idx 5); got {stopped_at}"


# --------------------------------------------------------------------------- D2: single monotonic detector by default


def test_universal_callback_has_no_worsening_detector():
    """The OLD budget-scaled worsening detector was REMOVED from UniversalCallback (benchmarked no-op +
    worst test accuracy). The fixed-N MonotonicDeclineStopper (wired into the shims) is now the SOLE
    monotonic stop, so UniversalCallback no longer carries any worsening machinery."""
    from mlframe.training.callbacks._callbacks import UniversalCallback

    cb = UniversalCallback(patience=10, monitor_dataset="valid_0", monitor_metric="rmse", mode="min", verbose=0)
    assert not hasattr(cb, "worsening_enabled")
    assert not hasattr(cb, "_update_worsening_streak")
    assert not hasattr(cb, "_worsening_threshold")
