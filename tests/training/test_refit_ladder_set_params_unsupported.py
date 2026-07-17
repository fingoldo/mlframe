"""Regression test for the 2026-05-27 CatBoostError in refit-ladder.

Bug: ``_maybe_refit_on_degenerate_best_iter`` called ``model_obj.set_params``
followed by ``.fit`` after the model was already fit. CatBoost's fitted-model
guard raises ``CatBoostError: You can't change params of fitted model``,
and the ``except (ValueError, TypeError)`` clause did NOT catch it -- the
exception propagated up through ``process_model``, killing training on the
third composite target (TVT-linresYj-TVT_prev) at 01:41:21 in
D:\\Temp\\TVT_regression.log.

Fix: catch broader Exception in the set_params branch and fall back to a
fresh-instance rebuild whose __dict__ is atomically copied back into the
original model_obj. Caller's reference stays valid.
"""

from __future__ import annotations

import logging

import numpy as np


class _FakeFittedBoosterRejectingSetParams:
    """Mimics CatBoost's fitted-model surface: ``set_params`` raises after fit.

    Tracks how many times each path fires so the test can verify the
    rebuild fallback ran.
    """

    def __init__(self, **kwargs):
        # Stash params for get_params -- mimics the booster API.
        self._stored = dict(kwargs)
        self._is_fitted = False
        # Counters for the test.
        self.set_params_attempts = 0
        self.fit_calls = 0
        self.best_iteration_ = 0

    def get_params(self, deep: bool = True):
        return dict(self._stored)

    def set_params(self, **params):
        self.set_params_attempts += 1
        if self._is_fitted:
            raise RuntimeError("You can't change params of fitted model. (sensor)")
        self._stored.update(params)
        return self

    def fit(self, X, y, **kwargs):
        self.fit_calls += 1
        self._is_fitted = True
        # Pretend best_iter improves only on RMSE refit.
        if self._stored.get("loss_function") == "RMSE":
            self.best_iteration_ = 133
        else:
            self.best_iteration_ = 1
        return self

    def predict(self, X):
        return np.zeros(len(X) if hasattr(X, "__len__") else 0)


def test_refit_ladder_handles_set_params_rejection() -> None:
    """When set_params raises after fit (CatBoost-style), the refit ladder
    must fall back to a fresh-instance rebuild so the post-refit model is
    actually trained with RMSE loss, not silently dropped.
    """
    from mlframe.training._training_loop_refit import (
        _maybe_refit_on_degenerate_best_iter,
    )

    # Pretend the user configured iterations=1500 with Huber loss. Initial
    # fit converged at best_iter=1 -> ladder should fire.
    model_obj = _FakeFittedBoosterRejectingSetParams(
        loss_function="Huber:delta=1.345",
        eval_metric="Huber:delta=1.345",
        iterations=1500,
    )
    # Simulate a prior fit completing at best_iter=1.
    model_obj.fit(np.zeros((10, 3)), np.zeros(10))
    assert model_obj._is_fitted
    assert model_obj.best_iteration_ == 1

    train_df = np.zeros((100, 3))
    train_target = np.linspace(0, 1, 100)
    log = logging.getLogger("test_refit_ladder")
    new_best_iter = _maybe_refit_on_degenerate_best_iter(
        model_obj=model_obj,
        # Must start with "CatBoost" to hit the per-backend fallback table.
        model_type_name="CatBoostRegressor",
        best_iter=1,
        train_df=train_df,
        train_target=train_target,
        fit_params={},
        logger_=log,
    )
    # The refit must have produced a new best_iter (the fallback path
    # rebuilds and fits, so best_iteration_ should now be 133 per our fake).
    assert new_best_iter == 133, (
        f"refit ladder did not run / new_best_iter={new_best_iter}; set_params_attempts={model_obj.set_params_attempts}, fit_calls={model_obj.fit_calls}"
    )
    # The original model_obj reference must reflect the RMSE-refit state
    # (atomic __dict__ swap from the rebuild fallback).
    assert model_obj._stored.get("loss_function") == "RMSE", f"model_obj state not swapped to RMSE: {model_obj._stored}"
    assert model_obj.best_iteration_ == 133, f"model_obj.best_iteration_ not refreshed: {model_obj.best_iteration_}"
