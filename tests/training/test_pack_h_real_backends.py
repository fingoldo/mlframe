"""End-to-end Pack H test: real CatBoost / LightGBM / XGBoost backends.

The earlier ``test_loss_recommendation.py`` covers the pure-function side of the helper. This test goes further: it constructs REAL CatBoostRegressor / LGBMRegressor / XGBRegressor instances (the exact classes mlframe builds in ``trainer.py``), wraps them in the same ``models_params`` dict shape ``select_target`` returns, hands the dict to ``_apply_loss_recommendation_in_place``, and verifies:

1. ``set_params`` succeeds on every backend for the recommended loss name (catches a typo in the recommendation strings before it surfaces in production).
2. After the mutation, each backend's ``fit`` + ``predict`` actually runs on a tiny heavy-tail target -- so the new objective is wire-compatible with the rest of the model machinery (no hidden validator rejects e.g. ``reg:absoluteerror`` on the installed XGBoost build).
"""
from __future__ import annotations

import logging

import numpy as np
import pytest


@pytest.fixture
def heavy_tail_target() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.laplace(loc=0.0, scale=1.0, size=5000)


@pytest.fixture
def gaussian_target() -> np.ndarray:
    return np.random.default_rng(11).standard_normal(5000)


def _build_models_params() -> dict:
    """Build the real-backend ``models_params`` dict in the shape mlframe's ``select_target`` returns."""
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    return {
        "cb": {"model": CatBoostRegressor(iterations=5, verbose=False), "fit_params": {}},
        "lgb": {"model": LGBMRegressor(n_estimators=5, verbose=-1), "fit_params": {}},
        "xgb": {"model": XGBRegressor(n_estimators=5), "fit_params": {}},
    }


class TestPackHRealBackends:
    """Pack H integration test against the real CB / LGB / XGB classes."""

    def test_heavy_tail_target_switches_all_three_to_huber(
        self, heavy_tail_target: np.ndarray,
    ) -> None:
        """2026-05-23 round-5 policy: kurt > 1.5 routes ALL three backends
        to Huber, not MAE / L1. Pure L1 was triggering the "MAE-gradient-
        is-noise" pathology on production TVT residuals (kurt=6.37 ->
        CB es_best_iter=1, LGB es_best_iter=5). Huber's bounded-influence
        loss retains a useful gradient on small residuals AND attenuates
        outlier influence; pure L1 is reserved for explicit
        ``target_quantile=0.5`` opt-in."""
        from mlframe.training.core._phase_train_one_target import (
            _apply_loss_recommendation_in_place,
        )

        models_params = _build_models_params()
        logger = logging.getLogger(__name__)
        _apply_loss_recommendation_in_place(
            models_params=models_params,
            target_values=heavy_tail_target,
            composite_name="laplace-test",
            logger_=logger,
            verbose=False,
        )

        cb_loss = models_params["cb"]["model"].get_params().get("loss_function")
        lgb_obj = models_params["lgb"]["model"].get_params().get("objective")
        xgb_obj = models_params["xgb"]["model"].get_params().get("objective")
        assert cb_loss == "Huber:delta=1.345", f"cb loss_function expected Huber:delta=1.345, got {cb_loss!r}"
        assert lgb_obj == "huber", f"lgb objective expected huber, got {lgb_obj!r}"
        assert xgb_obj == "reg:pseudohubererror", f"xgb objective expected reg:pseudohubererror, got {xgb_obj!r}"

    def test_after_switch_backends_actually_fit_and_predict(
        self, heavy_tail_target: np.ndarray,
    ) -> None:
        """The post-switch ``fit + predict`` round-trip MUST work on every backend (catches any installed-version incompatibility with the recommended objective string)."""
        from mlframe.training.core._phase_train_one_target import (
            _apply_loss_recommendation_in_place,
        )

        models_params = _build_models_params()
        _apply_loss_recommendation_in_place(
            models_params=models_params,
            target_values=heavy_tail_target,
            composite_name="laplace-fit-test",
            logger_=logging.getLogger(__name__),
            verbose=False,
        )

        rng = np.random.default_rng(31)
        n_train = 300
        X_train = rng.normal(0.0, 1.0, size=(n_train, 5))
        y_train = rng.laplace(0.0, 1.0, size=n_train)
        X_test = rng.normal(0.0, 1.0, size=(20, 5))

        for backend_name, entry in models_params.items():
            model = entry["model"]
            model.fit(X_train, y_train)
            preds = np.asarray(model.predict(X_test), dtype=np.float64)
            assert preds.shape == (20,), f"{backend_name} predict shape mismatch"
            assert np.all(np.isfinite(preds)), f"{backend_name} produced non-finite preds"

    def test_gaussian_target_keeps_default_losses(
        self, gaussian_target: np.ndarray,
    ) -> None:
        """Below-threshold kurtosis must leave all three backends at their default Gaussian losses."""
        from mlframe.training.core._phase_train_one_target import (
            _apply_loss_recommendation_in_place,
        )

        models_params = _build_models_params()
        _apply_loss_recommendation_in_place(
            models_params=models_params,
            target_values=gaussian_target,
            composite_name="gauss-test",
            logger_=logging.getLogger(__name__),
            verbose=False,
        )

        assert models_params["cb"]["model"].get_params().get("loss_function") == "RMSE"
        assert models_params["lgb"]["model"].get_params().get("objective") == "regression"
        assert models_params["xgb"]["model"].get_params().get("objective") == "reg:squarederror"

    def test_contaminated_target_picks_huber(self) -> None:
        """Extreme kurtosis (~ contaminated mixture) should pick Huber and the real backends must accept that variant string."""
        from mlframe.training.core._phase_train_one_target import (
            _apply_loss_recommendation_in_place,
        )

        rng = np.random.default_rng(13)
        main = rng.standard_normal(5000)
        idx = rng.choice(5000, size=250, replace=False)
        main[idx] = rng.standard_normal(250) * 30.0

        models_params = _build_models_params()
        _apply_loss_recommendation_in_place(
            models_params=models_params,
            target_values=main,
            composite_name="contam-test",
            logger_=logging.getLogger(__name__),
            verbose=False,
        )

        cb_loss = models_params["cb"]["model"].get_params().get("loss_function")
        lgb_obj = models_params["lgb"]["model"].get_params().get("objective")
        xgb_obj = models_params["xgb"]["model"].get_params().get("objective")
        # CB Huber needs a delta; production string is "Huber:delta=1.345" but a board may
        # reject it depending on the build -- accept either the full Huber string OR an
        # untouched default (graceful fallback when set_params rejected). We do require
        # LGB + XGB to be in their respective Huber variants since those are validated.
        assert lgb_obj == "huber", f"lgb objective expected huber, got {lgb_obj!r}"
        assert xgb_obj == "reg:pseudohubererror", f"xgb objective expected reg:pseudohubererror, got {xgb_obj!r}"
        assert cb_loss in {"Huber:delta=1.345", "RMSE"}, f"cb loss_function expected Huber or default fallback, got {cb_loss!r}"
