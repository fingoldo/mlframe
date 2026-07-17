"""Unit + biz_value tests for A2 fit-wiring: the recalibrated-regressor wrapper, the 2-fold-on-calib
gain gate, and the finalize hook ``_recalibrate_regression_on_calib_slice``.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
from types import SimpleNamespace

import numpy as np

from mlframe.training._regression_calibration import (
    RecalibratedRegressor,
    cv2_recalibration_gain,
    fit_point_recalibrator,
)
from mlframe.training.core._phase_finalize import _recalibrate_regression_on_calib_slice


class _FakeBase:
    """Groups tests covering fake base."""
    feature_names_in_ = np.array(["a", "b"])

    def predict(self, X):
        """Predict."""
        return np.asarray(X, dtype=np.float64).reshape(-1)


def test_wrapper_applies_g_and_delegates_attrs_and_pickles():
    """Wrapper applies g and delegates attrs and pickles."""
    rng = np.random.default_rng(0)
    yp = rng.uniform(-2, 2, 1000)
    yt = 2.0 * yp + rng.standard_normal(1000)
    g = fit_point_recalibrator(yp, yt, method="isotonic")
    w = RecalibratedRegressor(_FakeBase(), g)
    x = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(w.predict(x), g.transform(x))  # predict = g(base.predict)
    assert list(w.feature_names_in_) == ["a", "b"]  # delegated to base
    w2 = pickle.loads(pickle.dumps(w))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert np.allclose(w.predict(x), w2.predict(x))


def test_cv2_gain_positive_on_shrunk_zero_on_calibrated():
    """Cv2 gain positive on shrunk zero on calibrated."""
    rng = np.random.default_rng(1)
    n = 4000
    y = rng.standard_t(df=3, size=n)
    shrunk = 0.5 * y + 0.1 * rng.standard_normal(n)
    assert cv2_recalibration_gain(shrunk, y, "isotonic") >= 0.05
    calibrated = y + 0.3 * rng.standard_normal(n)
    assert abs(cv2_recalibration_gain(calibrated, y, "isotonic")) <= 0.03


def _reg_ctx(rng, shrink, **over):
    """Reg ctx."""
    n = 4000
    y_cal = rng.standard_t(df=3, size=n)
    calib_preds = shrink * y_cal + (1 - shrink) * float(np.mean(y_cal)) + 0.1 * rng.standard_normal(n)
    y_test = rng.standard_t(df=3, size=n)
    test_preds = shrink * y_test + (1 - shrink) * float(np.mean(y_test)) + 0.1 * rng.standard_normal(n)
    e = SimpleNamespace(
        model=_FakeBase(),
        test_probs=None,
        calib_preds=calib_preds,
        calib_target=y_cal,
        test_preds=test_preds,
        test_target=y_test,
        val_preds=None,
        train_preds=None,
        oof_preds=None,
        model_name="m",
    )
    base = dict(models={"REGRESSION": {"y": [e]}}, metadata={}, verbose=0, regression_calibration_config=None, configs=None)
    base.update(over)
    return SimpleNamespace(**base), e


def test_hook_applies_recalibration_on_shrunk_when_enabled(monkeypatch):
    """Hook applies recalibration on shrunk when enabled."""
    monkeypatch.setenv("MLFRAME_REGRESSION_RECALIBRATION", "isotonic")
    rng = np.random.default_rng(2)
    ctx, e = _reg_ctx(rng, shrink=0.5)
    raw_test = np.array(e.test_preds, copy=True)
    _recalibrate_regression_on_calib_slice(ctx)
    assert isinstance(e.model, RecalibratedRegressor), "shrunk model should be wrapped"
    assert not np.allclose(e.test_preds, raw_test), "test_preds should be re-stamped to recalibrated"
    assert "regression_recalibration" in ctx.metadata
    rec = ctx.metadata["regression_recalibration"]["REGRESSION/y/m"]
    assert rec["method"] == "isotonic" and rec["cv2_gain"] > 0


def test_hook_skips_when_gate_rejects_calibrated(monkeypatch):
    """Hook skips when gate rejects calibrated."""
    monkeypatch.setenv("MLFRAME_REGRESSION_RECALIBRATION", "isotonic")
    rng = np.random.default_rng(3)
    ctx, e = _reg_ctx(rng, shrink=1.0)  # already calibrated -> gate should reject
    _recalibrate_regression_on_calib_slice(ctx)
    assert not isinstance(e.model, RecalibratedRegressor)
    assert "regression_recalibration" not in ctx.metadata


def test_hook_noop_when_disabled(monkeypatch):
    """Hook noop when disabled."""
    monkeypatch.delenv("MLFRAME_REGRESSION_RECALIBRATION", raising=False)
    rng = np.random.default_rng(4)
    ctx, e = _reg_ctx(rng, shrink=0.5)  # would benefit, but feature is OFF by default
    _recalibrate_regression_on_calib_slice(ctx)
    assert not isinstance(e.model, RecalibratedRegressor)
    assert "regression_recalibration" not in ctx.metadata
