"""Regression tests for the training-log audit 2026-09-20, ``TRN-01`` / ``TRN-02``.

Early stopping is aligned to the chosen objective, so a robust loss (Huber / MAE / quantile) makes the ES surface
bounded-influence too. Bounded influence is precisely indifference to large residuals -- the population RMSE and R^2
are made of -- so the eval metric can improve monotonically while the reported error diverges. A production fit ran
to ``es_best_iter: 999`` of 1000 that way and reached val R^2 = -4.61 against a per-group-mean baseline's +0.029.

``_maybe_refit_on_degenerate_best_iter`` covered only the opposite side (ES firing far too early).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training._training_loop_refit import (
    _eval_set_xy,
    _maybe_refit_on_saturated_best_iter,
)

logger = logging.getLogger(__name__)
MAX_ITER = 200


class _FakeBooster:
    """Minimal booster stand-in: records the loss it holds and what a refit changed it to.

    Predictions are scripted rather than learned so the test pins the POLICY (when does a refit fire, what does it
    set) without depending on a real CatBoost/LGB fit.
    """

    def __init__(self, *, loss_function="Huber:delta=1.0", eval_metric="Huber:delta=1.0", iterations=MAX_ITER, bad=True):
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.iterations = iterations
        self.bad = bad
        self.n_fits = 0

    def get_params(self, deep=True):
        return {
            "loss_function": self.loss_function,
            "eval_metric": self.eval_metric,
            "iterations": self.iterations,
        }

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    def fit(self, X, y, **kw):
        self.n_fits += 1
        # A refit onto the RMSE family is what makes the model good; anything else keeps it pathological.
        self.bad = "rmse" not in str(self.loss_function).lower()
        return self

    def predict(self, X):
        n = len(X)
        rng = np.random.default_rng(0)
        if self.bad:
            # The production shape: predictions with ~2.3x the target's spread, so R^2 goes deeply negative.
            return rng.normal(50.0, 400.0, n)
        return _Y_TRUE[:n] + rng.normal(0.0, 1.0, n)


_X_VAL = np.arange(2000, dtype=np.float64).reshape(-1, 1)
_Y_TRUE = np.sin(np.arange(2000) / 50.0) * 150.0 + 300.0


def _fit_params():
    return {"eval_set": [(_X_VAL, _Y_TRUE)]}


def _call(model, best_iter=MAX_ITER - 1, fit_params=None):
    return _maybe_refit_on_saturated_best_iter(
        model_obj=model,
        model_type_name="CatBoostRegressor",
        best_iter=best_iter,
        train_df=_X_VAL,
        train_target=_Y_TRUE,
        fit_params=_fit_params() if fit_params is None else fit_params,
        logger_=logger,
    )


def test_saturated_robust_loss_with_negative_r2_is_refit_on_rmse():
    model = _FakeBooster()
    new_best = _call(model)
    assert new_best is not None or model.n_fits == 1
    assert model.loss_function == "RMSE"
    assert model.eval_metric == "RMSE"
    assert model.n_fits == 1


def test_saturated_robust_loss_with_healthy_r2_is_left_alone():
    """Hitting the cap is not by itself a defect -- a model that is still winning should keep training, not be
    swapped onto another loss."""
    model = _FakeBooster(bad=False)
    assert _call(model) is None
    assert model.loss_function.startswith("Huber")
    assert model.n_fits == 0


def test_non_saturated_best_iter_is_left_alone():
    """Early stopping fired, so it did its job regardless of how bad the model is -- that case belongs to the
    degenerate-best-iter policy, not this one."""
    model = _FakeBooster()
    assert _call(model, best_iter=29) is None
    assert model.n_fits == 0


def test_rmse_loss_at_the_cap_is_left_alone():
    """A non-robust loss that saturates is ordinary under-training; refitting it onto itself would be a no-op that
    doubles the fit cost."""
    model = _FakeBooster(loss_function="RMSE", eval_metric="RMSE")
    assert _call(model) is None
    assert model.n_fits == 0


def test_a_deliberately_tiny_budget_is_respected():
    """A caller who asked for 20 iterations chose that budget; their loss must not be silently swapped."""
    model = _FakeBooster(iterations=20)
    assert _call(model, best_iter=19) is None
    assert model.n_fits == 0


def test_unscoreable_eval_set_warns_and_keeps_the_fit(caplog):
    """A CatBoost ``Pool`` exposes no (X, y), so the check cannot score it. It must say so rather than guess."""
    model = _FakeBooster()
    with caplog.at_level(logging.WARNING):
        assert _call(model, fit_params={"eval_set": object()}) is None
    assert model.n_fits == 0
    assert "not in a form this check can score" in caplog.text


def test_non_booster_models_are_ignored():
    model = _FakeBooster()
    assert (
        _maybe_refit_on_saturated_best_iter(
            model_obj=model, model_type_name="LightningMLPRegressor", best_iter=MAX_ITER - 1,
            train_df=_X_VAL, train_target=_Y_TRUE, fit_params=_fit_params(), logger_=logger,
        )
        is None
    )
    assert model.n_fits == 0


@pytest.mark.parametrize(
    "fit_params,expected",
    [
        ({"eval_set": (_X_VAL, _Y_TRUE)}, True),
        ({"eval_set": [(_X_VAL, _Y_TRUE)]}, True),
        ({"X_val": _X_VAL, "y_val": _Y_TRUE}, True),
        ({"eval_set": None}, False),
        ({}, False),
        ({"eval_set": object()}, False),
    ],
)
def test_eval_set_xy_understands_every_backend_convention(fit_params, expected):
    got = _eval_set_xy(fit_params)
    assert (got is not None) is expected
    if expected:
        assert len(got) == 2
        assert len(got[1]) == len(_Y_TRUE)
