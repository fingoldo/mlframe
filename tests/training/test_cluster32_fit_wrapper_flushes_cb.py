"""Cluster 32 regression: the fit wrapper flushes the CatBoost iteration callback after a successful fit (audit 2026-11-20, finding 32-F9).

With stride > 1 the final round is skipped during fit. The wrapper calls ``flush()`` once fit returns, so the final round is captured. On a
failed fit the wrapper does not flush, because a half-trained model should not be scored.
"""

import numpy as np
import pytest

import mlframe.training.cb.shared as cb_shared
from mlframe.training._training_loop import _train_model_with_fallback
from mlframe.training.callbacks.iteration_metrics import CBIterationMetricsCallback


class _Info:
    """CatBoost ``info`` object: 1-based round and the model."""

    def __init__(self, iteration, model):
        self.iteration = iteration
        self.model = model


class _Model:
    """Regression stub whose predict returns a round-dependent vector."""

    def predict(self, pool, ntree_end):
        """Return a vector that depends on the round."""
        return np.full(4, float(ntree_end))


def _fit_that_runs_rounds(rounds: int, fail: bool = False):
    """Build a stand-in guarded fit that feeds ``rounds`` iterations to the callbacks in fit_params, or raises."""

    def _fake(fn, model, model_obj, model_type_name, train_df, train_target, fit_params, verbose):
        """Stand-in for fit_with_cb_gpu_guard: streams rounds through the callback, then returns or raises."""
        cb = next(c for c in fit_params["callbacks"] if isinstance(c, CBIterationMetricsCallback))
        for it in range(1, rounds + 1):
            cb.after_iteration(_Info(it, _Model()))
        if fail:
            raise RuntimeError("fit failed")
        return model, None

    return _fake


def _cb(stride: int) -> CBIterationMetricsCallback:
    """Regression callback with the given stride over four validation rows."""
    return CBIterationMetricsCallback(val_pool=None, y_val=np.array([1.0, 2.0, 3.0, 4.0]), target_type="regression", stride=stride)


def test_successful_fit_flushes_the_final_round(monkeypatch):
    """After a successful fit the final round (index 11 of 12) is captured by the wrapper's flush (32-F9)."""
    monkeypatch.setattr(cb_shared, "fit_with_cb_gpu_guard", _fit_that_runs_rounds(12))
    cb = _cb(stride=5)
    _train_model_with_fallback("m", "m", "CatBoostRegressor", None, None, {"callbacks": [cb]})
    assert 11 in cb.iteration_metrics_


def test_failed_fit_does_not_flush(monkeypatch):
    """A failed fit propagates and leaves the final round unscored, so a half-trained model is not recorded (32-F9)."""
    monkeypatch.setattr(cb_shared, "fit_with_cb_gpu_guard", _fit_that_runs_rounds(12, fail=True))
    cb = _cb(stride=5)
    with pytest.raises(RuntimeError, match="fit failed"):
        _train_model_with_fallback("m", "m", "CatBoostRegressor", None, None, {"callbacks": [cb]})
    assert 11 not in cb.iteration_metrics_
