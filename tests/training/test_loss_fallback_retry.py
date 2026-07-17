"""Locks the 2026-05-26 loss-fallback retry on degenerate ES.

Production failure (CB on TVT-addres-TVT_prev composite, excess_kurt
+42.67 / skew -4.96): ``recommend_boosting_regression_loss`` swapped
CatBoost to ``Huber:delta=1.345``, but on the extreme-kurt residual
the Huber gradient ``delta * sign(r)`` collapses to ~0 when most
rows have ``r ~ 0``; the overfit-detector saw no improvement and ES
fired at iter=0. Model returned a constant prediction (MAE=14.10,
R^2=-5.05, ``@iter=0`` stamp in chart title).

Fix: ``_maybe_refit_on_degenerate_best_iter`` detects ``best_iter <
_MIN_BEST_ITER_HEALTHY`` on a CB / LGB / XGB model that currently
runs a non-default loss (Huber / MAE / L1 / quantile / pseudohuber)
and refits with the RMSE-family default. RMSE is less robust to
outliers but the training surface always has a usable gradient -- a
non-trivial fit beats a constant-prediction collapse.
"""

from __future__ import annotations

import logging



# ---------------------------------------------------------------------------
# Stand-in booster: minimal CB / LGB / XGB interface needed by the helper.
# ---------------------------------------------------------------------------


class _StubBooster:
    """Records set_params + fit calls; emits configurable best_iter."""

    def __init__(
        self,
        *,
        loss_param_key: str,
        initial_loss: str,
        best_iter_first: int,
        best_iter_after_refit: int,
        booster_type_name: str = "CatBoostRegressor",
    ):
        self._params = {loss_param_key: initial_loss}
        self._loss_param_key = loss_param_key
        self._best_iter_first = best_iter_first
        self._best_iter_after_refit = best_iter_after_refit
        # Production flow: ``_train_model_with_fallback`` runs the first
        # fit BEFORE invoking the helper, so the stub simulates a model
        # that already saw one fit. The helper's refit becomes the
        # second fit (fit_count == 2) and the stub then exposes the
        # ``best_iter_after_refit`` value.
        self._fit_count = 1
        self.set_params_history: list[dict] = []
        self.fit_history: list[dict] = []
        # ``model_type_name`` is the helper's prefix-matcher input;
        # the stub exposes the same name via ``__class__.__name__`` so
        # downstream callers (not exercised here) wouldn't break.
        self.__class__.__name__ = booster_type_name

    def get_params(self, deep: bool = True) -> dict:
        """Get params."""
        return dict(self._params)

    def set_params(self, **kwargs) -> "_StubBooster":
        """Set params."""
        self._params.update(kwargs)
        self.set_params_history.append(kwargs)
        return self

    def fit(self, X, y, **fit_params) -> "_StubBooster":
        """Fit."""
        self._fit_count += 1
        self.fit_history.append({"X_id": id(X), "y_id": id(y), "fit_params": dict(fit_params)})
        return self

    # ``get_model_best_iter`` calls these in order on CB / LGB / XGB.
    def get_best_iteration(self) -> int:
        """Get best iteration."""
        return self._best_iter_first if self._fit_count <= 1 else self._best_iter_after_refit

    @property
    def best_iteration(self) -> int:
        """Best iteration."""
        return self.get_best_iteration()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefitOnDegenerateBestIter:
    """Groups tests covering refit on degenerate best iter."""
    def _call_helper(self, stub, *, model_type_name: str, best_iter: int):
        """Call helper."""
        from mlframe.training._training_loop import _maybe_refit_on_degenerate_best_iter

        return _maybe_refit_on_degenerate_best_iter(
            model_obj=stub,
            model_type_name=model_type_name,
            best_iter=best_iter,
            train_df=[[0]],
            train_target=[0],
            fit_params={},
            logger_=logging.getLogger("test"),
        )

    def test_catboost_huber_degenerate_refits_with_rmse(self):
        """Catboost huber degenerate refits with rmse."""
        stub = _StubBooster(
            loss_param_key="loss_function",
            initial_loss="Huber:delta=1.345",
            best_iter_first=0,
            best_iter_after_refit=120,
        )
        new_best = self._call_helper(stub, model_type_name="CatBoostRegressor", best_iter=0)
        assert new_best == 120
        # Loss was reset to RMSE.
        assert stub.set_params_history
        last = stub.set_params_history[-1]
        assert last.get("loss_function") == "RMSE"
        assert last.get("eval_metric") == "RMSE"
        # Refit actually ran.
        assert len(stub.fit_history) == 1

    def test_lgb_huber_degenerate_refits_with_regression(self):
        """Lgb huber degenerate refits with regression."""
        stub = _StubBooster(
            loss_param_key="objective",
            initial_loss="huber",
            best_iter_first=2,
            best_iter_after_refit=85,
            booster_type_name="LGBMRegressor",
        )
        new_best = self._call_helper(stub, model_type_name="LGBMRegressor", best_iter=2)
        assert new_best == 85
        last = stub.set_params_history[-1]
        assert last.get("objective") == "regression"
        assert last.get("metric") == "l2"

    def test_xgb_pseudohuber_degenerate_refits_with_squarederror(self):
        """Xgb pseudohuber degenerate refits with squarederror."""
        stub = _StubBooster(
            loss_param_key="objective",
            initial_loss="reg:pseudohubererror",
            best_iter_first=1,
            best_iter_after_refit=60,
            booster_type_name="XGBRegressor",
        )
        new_best = self._call_helper(stub, model_type_name="XGBRegressor", best_iter=1)
        assert new_best == 60
        last = stub.set_params_history[-1]
        assert last.get("objective") == "reg:squarederror"
        assert last.get("eval_metric") == "rmse"

    def test_healthy_best_iter_no_refit(self):
        """best_iter >= 3 means the booster trained successfully; no
        refit needed even when loss is Huber."""
        stub = _StubBooster(
            loss_param_key="loss_function",
            initial_loss="Huber:delta=1.345",
            best_iter_first=147,
            best_iter_after_refit=999,
        )
        new_best = self._call_helper(stub, model_type_name="CatBoostRegressor", best_iter=147)
        assert new_best is None
        assert not stub.set_params_history
        assert not stub.fit_history

    def test_degenerate_with_default_loss_no_refit(self):
        """A booster on RMSE that ES'd at iter=2 is a legitimate
        convergence (or a different problem). Don't refit -- swapping
        away from RMSE would only make it worse."""
        stub = _StubBooster(
            loss_param_key="loss_function",
            initial_loss="RMSE",
            best_iter_first=2,
            best_iter_after_refit=999,
        )
        new_best = self._call_helper(stub, model_type_name="CatBoostRegressor", best_iter=2)
        assert new_best is None
        assert not stub.fit_history

    def test_non_boosting_model_no_refit(self):
        """Ridge / Lasso / MLP etc don't have best_iter semantics; the
        helper must early-return without touching them."""
        stub = _StubBooster(
            loss_param_key="loss_function",
            initial_loss="Huber:delta=1.345",
            best_iter_first=0,
            best_iter_after_refit=120,
        )
        new_best = self._call_helper(stub, model_type_name="Ridge", best_iter=0)
        assert new_best is None
        assert not stub.fit_history

    def test_tiny_iterations_budget_no_refit(self):
        """User explicitly chose ``iterations=2``: best_iter=1 is half
        the budget, not a degenerate ES. Adaptive threshold must NOT
        trigger a retry even though best_iter=1 < absolute floor 3."""
        stub = _StubBooster(
            loss_param_key="loss_function",
            initial_loss="Huber:delta=1.345",
            best_iter_first=1,
            best_iter_after_refit=120,
        )
        # Expose ``iterations=2`` so the adaptive rule computes
        # max(1, 2*0.05) = 1; min(3, 1) = 1; best_iter=1 not < 1.
        stub._params["iterations"] = 2
        new_best = self._call_helper(stub, model_type_name="CatBoostRegressor", best_iter=1)
        assert new_best is None
        assert not stub.fit_history, "must not refit when user explicitly chose a tiny iteration budget"

    def test_large_budget_with_iter1_refits(self):
        """User set ``iterations=1000`` and got best_iter=2: adaptive
        floor = max(1, 50) = 50, min(3, 50) = 3, best_iter=2 < 3 -> refit."""
        stub = _StubBooster(
            loss_param_key="loss_function",
            initial_loss="Huber:delta=1.345",
            best_iter_first=2,
            best_iter_after_refit=350,
        )
        stub._params["iterations"] = 1000
        new_best = self._call_helper(stub, model_type_name="CatBoostRegressor", best_iter=2)
        assert new_best == 350
        assert stub.fit_history, "large-budget degenerate ES must trigger refit"

    def test_refit_failure_returns_none_keeps_first_fit(self):
        """When set_params or fit raises (e.g. backend rejects RMSE on
        this build), the helper returns None and the original
        degenerate best_iter survives so the chart shows the truth."""

        class _RaisingStub(_StubBooster):
            """Groups tests covering raising stub."""
            def set_params(self, **kwargs):
                """Set params."""
                raise ValueError("simulated backend rejection")

        stub = _RaisingStub(
            loss_param_key="loss_function",
            initial_loss="Huber:delta=1.345",
            best_iter_first=0,
            best_iter_after_refit=120,
        )
        new_best = self._call_helper(stub, model_type_name="CatBoostRegressor", best_iter=0)
        assert new_best is None
        assert not stub.fit_history
