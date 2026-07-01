"""Locks the 2026-05-26 MLP / recurrent collapse-detection + refit.

The booster ``best_iter < threshold`` detector doesn't catch Lightning
MLP / recurrent collapse modes -- those backends don't expose a
``best_iteration`` attribute on their regression heads. Instead, the
detector observes ``predict(X_train).std() / y_train.std() < 0.1``:
any model that emits a near-constant prediction (saturated tanh
output, dead ReLU, etc) triggers a refit with the bounded output
activation removed.

These tests pin the policy on a minimal Lightning-shaped stand-in.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pytest


class _StubLightning:
    """Minimal stand-in for ``PytorchLightningRegressor`` -- has the
    attributes the collapse helper inspects: ``network_params``,
    ``trainer_params``, plus a ``predict`` that returns whatever the
    test sets as ``next_preds``. Re-fit toggles the prediction
    callback so the test can simulate "linear-output refit fixed it"
    vs "still collapsed even after refit"."""

    def __init__(
        self,
        *,
        initial_pred_value: float,
        post_refit_pred_value: float,
        n_rows: int,
        output_activation: str = "tanh_train_range",
        max_epochs: int = 30,
    ):
        self.network_params = {
            "output_activation": output_activation,
            "output_activation_scale": 5.0,
            "output_activation_center": 11500.0,
        }
        self.trainer_params = {"max_epochs": max_epochs}
        self.network = object()  # truthy stand-in for a trained network
        self._initial_pred_value = initial_pred_value
        self._post_refit_pred_value = post_refit_pred_value
        self._n_rows = n_rows
        self._fit_count = 1  # first fit done before helper called
        self.fit_history: list[dict] = []

    def fit(self, X, y, **kwargs) -> "_StubLightning":
        self._fit_count += 1
        self.fit_history.append({"y_id": id(y)})
        return self

    def predict(self, X) -> np.ndarray:
        n = len(X) if hasattr(X, "__len__") else self._n_rows
        val = self._initial_pred_value if self._fit_count <= 1 else self._post_refit_pred_value
        return np.full(n, float(val), dtype=np.float64)


def _call_helper(stub, model_type_name: str, train_df, train_target):
    from mlframe.training._training_loop import _maybe_refit_on_collapsed_predictions
    # The helper inspects ``model_obj.network_params`` directly; pass
    # the stub as both ``model`` (predict surface) and ``model_obj``
    # (network_params surface) which mirrors the Lightning path
    # (the wrapped model and the regressor are the same object
    # after the trainer strips the Pipeline).
    return _maybe_refit_on_collapsed_predictions(
        model=stub,
        model_obj=stub,
        model_type_name=model_type_name,
        train_df=train_df,
        train_target=train_target,
        fit_params={},
        logger_=logging.getLogger("test"),
    )


class TestCollapseDetection:
    def test_collapsed_pred_triggers_linear_refit(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(500) * 10.0 + 11500.0
        stub = _StubLightning(
            initial_pred_value=11500.0,  # constant -> std=0
            post_refit_pred_value=11500.0 + rng.standard_normal(),  # post-refit not relevant
            n_rows=500,
        )
        result = _call_helper(stub, "PytorchLightningRegressor", train_df=list(range(500)), train_target=y)
        # Stub's post_refit_pred is constant -> ratio2 stays 0 < threshold,
        # so the helper walks the full ladder (3 rungs) and returns False
        # at the end. Architectural fixes ran (BN enabled, then 1-layer,
        # then dropout); none recovered a healthy fit.
        assert result is False
        # output_activation MUST NOT be stripped -- 2026-05-26 ladder
        # contract preserves tanh_train_range so the prediction bound
        # survives even when architectural fixes fail.
        assert stub.network_params["output_activation"] == "tanh_train_range"
        # All 3 ladder rungs attempted; each calls fit() once.
        assert len(stub.fit_history) == 3
        # Final state restored to original snapshot (last attempted rung
        # not persisted when the ladder exhausts).
        assert stub.network_params.get("use_batchnorm", False) is False

    def test_healthy_predictions_no_refit(self):
        """pred_std / y_std >= 0.1 -> no collapse, helper is a no-op."""
        rng = np.random.default_rng(1)
        y = rng.standard_normal(500) * 10.0 + 11500.0
        # Make predictions that vary like y (no collapse).
        n = 500
        class _HealthyStub(_StubLightning):
            def predict(self, X):
                return rng.standard_normal(n) * 9.0 + 11500.0
        stub = _HealthyStub(
            initial_pred_value=0, post_refit_pred_value=0, n_rows=n,
        )
        result = _call_helper(stub, "PytorchLightningRegressor", train_df=list(range(n)), train_target=y)
        assert result is False
        assert stub.network_params["output_activation"] == "tanh_train_range"
        assert not stub.fit_history

    def test_already_linear_still_tries_ladder(self):
        """2026-05-26 ladder contract: even when output_activation is
        already 'linear', the helper still tries the architectural
        ladder (BN, 1-layer, dropout). Earlier policy short-circuited
        and gave up; new policy gives the model architectural fixes
        regardless of which output activation was configured. The
        architecture rungs ran but the constant-pred stub returns to
        baseline so result is False (ladder exhausted)."""
        y = np.random.default_rng(2).standard_normal(500) * 10.0 + 11500.0
        stub = _StubLightning(
            initial_pred_value=11500.0,
            post_refit_pred_value=11500.0,
            n_rows=500,
            output_activation="linear",
        )
        result = _call_helper(stub, "PytorchLightningRegressor", train_df=list(range(500)), train_target=y)
        assert result is False
        # Architectural ladder DID run (3 rungs).
        assert len(stub.fit_history) == 3
        # output_activation untouched (was 'linear', stays 'linear').
        assert stub.network_params["output_activation"] == "linear"

    def test_tiny_max_epochs_no_refit(self):
        """User explicitly chose ``max_epochs=2``: a collapsed-looking
        fit is the budget, not a failure. Helper must skip retry."""
        y = np.random.default_rng(3).standard_normal(500) * 10.0 + 11500.0
        stub = _StubLightning(
            initial_pred_value=11500.0,
            post_refit_pred_value=11500.0,
            n_rows=500,
            max_epochs=2,  # tiny budget
        )
        result = _call_helper(stub, "PytorchLightningRegressor", train_df=list(range(500)), train_target=y)
        assert result is False
        assert not stub.fit_history

    def test_no_network_params_no_refit(self):
        """Non-Lightning model (no ``network_params`` attribute) is
        skipped -- the helper has no knob to tweak."""
        class _Plain:
            def predict(self, X):
                return np.zeros(500)
        result = _call_helper(_Plain(), "Ridge",
                              train_df=list(range(500)),
                              train_target=np.arange(500, dtype=np.float64))
        assert result is False

    def test_constant_y_train_no_refit(self):
        """y_train.std() == 0 means the ratio is undefined; helper
        skips (the target itself is degenerate)."""
        y = np.full(500, 11500.0)
        stub = _StubLightning(
            initial_pred_value=11500.0,
            post_refit_pred_value=11500.0,
            n_rows=500,
        )
        result = _call_helper(stub, "PytorchLightningRegressor", train_df=list(range(500)), train_target=y)
        assert result is False
        assert not stub.fit_history

    def test_refit_failure_swallowed(self):
        """When EVERY refit rung raises, helper exhausts the ladder
        and returns False; the model survives so the chart shows the
        truth (R^2 likely < 0). With the 2026-05-26 ladder, fit() can
        be called up to 3 times -- each ladder rung tries to refit;
        all 3 should raise here, and the helper should still gracefully
        return False without propagating the exception."""
        y = np.random.default_rng(4).standard_normal(500) * 10.0 + 11500.0
        class _RaisingStub(_StubLightning):
            def fit(self, X, y, **kwargs):
                raise RuntimeError("simulated rebuild error")
        stub = _RaisingStub(
            initial_pred_value=11500.0,
            post_refit_pred_value=11500.0,
            n_rows=500,
        )
        result = _call_helper(stub, "PytorchLightningRegressor", train_df=list(range(500)), train_target=y)
        assert result is False
