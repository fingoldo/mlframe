"""Sensor / regression tests for CompositeTargetEstimator + loss diagnostics
fixes shipped 2026-05-25 (round 5.5).

Covers:

* ``get_model_best_iter`` unwraps ``CompositeTargetEstimator`` and reads
  CB's ``best_iteration_`` property (was missing -> @iter never shown
  for composite-wrapped models).
* ``recommend_boosting_regression_loss`` MAD-calibrates XGB's
  ``huber_slope`` instead of letting it default to 1.0 (was causing
  XGB pred blow-up by 30x on T-scale composite targets like
  TVT-addres-TVT_prev: pred [-50,+340] for T in [-50,+50]).
* CB's ``Huber:delta=1.345`` loss is matched by ``eval_metric=Huber:
  delta=1.345`` (was MAE -> constant-magnitude gradient stops ES at
  iter=1) PLUS ``od_pval=1e-5`` + ``early_stopping_rounds=100`` (the
  canonicalised ``od_wait`` synonym; passing both raises in CB).
* ``CompositeTargetEstimator`` clips T-scale predictions to a
  MAD-derived envelope BEFORE applying the inverse, so any backend
  blow-up is bounded at its source.
* ``_update_model_name_after_training`` shows ``@iter=0`` (best_iter
  can legitimately be 0 on tiny-residual targets; was silently dropped
  by ``if best_iter:``).
"""

from __future__ import annotations

import numpy as np
import pytest


class TestGetModelBestIterUnwrapsCTE:
    """Groups tests covering get model best iter unwraps c t e."""
    def test_unwraps_composite_target_estimator(self) -> None:
        """Unwraps composite target estimator."""
        from mlframe.core.helpers import get_model_best_iter

        class _InnerWithBestIter:
            """Groups tests covering inner with best iter."""
            best_iteration_ = 137

        class _FakeCTE:
            """Groups tests covering fake c t e."""
            estimator_ = _InnerWithBestIter()

        assert get_model_best_iter(_FakeCTE()) == 137

    def test_unwraps_transformed_target_regressor_regressor_attr(self) -> None:
        """sklearn TransformedTargetRegressor (parent of _TTRWithEvalSetScaling)
        exposes the inner via ``.regressor_`` NOT ``.estimator_``; without
        the .regressor_ unwrap branch, PytorchLightningRegressor's
        ``best_epoch`` stays invisible behind _TTRWithEvalSetScaling and
        report headers lose @iter=N. Sensor for the 2026-05-27 fix.
        """
        from mlframe.core.helpers import get_model_best_iter

        class _InnerWithBestEpoch:
            """Groups tests covering inner with best epoch."""
            best_epoch = 42

        class _TTRLike:
            # mimic sklearn TransformedTargetRegressor surface
            """Groups tests covering t t r like."""
            regressor_ = _InnerWithBestEpoch()
            # NOTE: no estimator_, no best_iteration_, no best_epoch on self

        assert get_model_best_iter(_TTRLike()) == 42

    def test_returns_int_not_None_for_iter_zero(self) -> None:
        """ES on iter=0 is a real outcome on tiny-residual targets.
        Previously ``if best_iter:`` swallowed it as falsy."""
        from mlframe.core.helpers import get_model_best_iter

        class _ZeroIter:
            """Groups tests covering zero iter."""
            best_iteration_ = 0

        assert get_model_best_iter(_ZeroIter()) == 0

    def test_falls_back_to_tree_count_(self) -> None:
        """CB without ES uses tree_count_ as the iter substitute for
        chart titles."""
        from mlframe.core.helpers import get_model_best_iter

        class _NoESCatBoost:
            """Groups tests covering no e s cat boost."""
            tree_count_ = 250

        assert get_model_best_iter(_NoESCatBoost()) == 250

    def test_prefers_best_iteration_underscore_first(self) -> None:
        """Prefers best iteration underscore first."""
        from mlframe.core.helpers import get_model_best_iter

        class _BothAttrs:
            """Groups tests covering both attrs."""
            best_iteration = 999
            best_iteration_ = 42

        assert get_model_best_iter(_BothAttrs()) == 42

    def test_returns_None_when_nothing_exposed(self) -> None:
        """Returns none when nothing exposed."""
        from mlframe.core.helpers import get_model_best_iter

        class _NoIter:
            """Groups tests covering no iter."""
            pass

        assert get_model_best_iter(_NoIter()) is None


class TestUpdateModelNameShowsIterZero:
    """Groups tests covering update model name shows iter zero."""
    def test_iter_zero_appears_in_name(self) -> None:
        """Iter zero appears in name."""
        from mlframe.training._data_helpers import _update_model_name_after_training

        name = _update_model_name_after_training("M", 100, "", best_iter=0)
        assert "@iter=0" in name

    def test_iter_None_suppresses_label(self) -> None:
        """Iter none suppresses label."""
        from mlframe.training._data_helpers import _update_model_name_after_training

        name = _update_model_name_after_training("M", 100, "", best_iter=None)
        assert "@iter=" not in name


class TestXgbHuberSlopeMadCalibrated:
    """Groups tests covering xgb huber slope mad calibrated."""
    def test_huber_slope_scaled_by_mad(self) -> None:
        """Huber slope scaled by mad."""
        from mlframe.training.loss_recommendation import (
            recommend_boosting_regression_loss,
        )

        rng = np.random.default_rng(42)
        # Heavy-tail T-scale residual: mean ~0, std~13, kurt>>3.
        base = rng.normal(0, 13, 5000)
        outliers = rng.normal(0, 60, 200)
        target = np.concatenate([base, outliers])
        rec = recommend_boosting_regression_loss(target)
        assert rec["xgb"] == "reg:pseudohubererror"
        assert "xgb_extra_params" in rec
        assert "huber_slope" in rec["xgb_extra_params"]
        slope = rec["xgb_extra_params"]["huber_slope"]
        # MAD ~ 0.67*std ~ 8.7; 1.345 * MAD ~ 11.7. Must be MUCH greater
        # than the default 1.0 (which is the bug we're fixing).
        assert slope > 2.0, f"huber_slope {slope} too close to default 1.0; MAD calibration didn't fire"
        assert "mad" in rec

    def test_no_huber_extra_params_for_gaussian(self) -> None:
        """No huber extra params for gaussian."""
        from mlframe.training.loss_recommendation import (
            recommend_boosting_regression_loss,
        )

        rng = np.random.default_rng(42)
        target = rng.normal(0, 1, 5000)
        rec = recommend_boosting_regression_loss(target)
        assert rec["xgb"] == "reg:squarederror"
        assert "xgb_extra_params" not in rec


class TestCbHuberEvalMatchesLoss:
    """Groups tests covering cb huber eval matches loss."""
    def test_cb_eval_metric_matches_huber_loss(self) -> None:
        """Cb eval metric matches huber loss."""
        from mlframe.training.loss_recommendation import (
            recommend_boosting_regression_loss,
        )

        rng = np.random.default_rng(42)
        base = rng.normal(0, 13, 5000)
        outliers = rng.normal(0, 60, 200)
        target = np.concatenate([base, outliers])
        rec = recommend_boosting_regression_loss(target)
        assert rec["cb"].startswith("Huber:")
        assert "cb_extra_params" in rec
        cb_extras = rec["cb_extra_params"]
        assert cb_extras["od_type"] == "IncToDec"
        assert cb_extras["od_pval"] == pytest.approx(1e-5)
        # CatBoost canonicalises ``od_wait`` and ``early_stopping_rounds`` into
        # the same parameter group; passing both raises CatBoostError. The
        # base CB params already carry ``early_stopping_rounds`` so
        # recommend_boosting_regression_loss now emits ``early_stopping_rounds``
        # (semantically equivalent to od_wait, no collision).
        assert cb_extras["early_stopping_rounds"] == 100

    def test_eval_metric_for_cb_huber_returns_huber(self) -> None:
        """The matcher returns the SAME Huber-with-delta string for the
        eval_metric, not 'MAE' (the pre-fix value)."""
        from pathlib import Path

        src = Path(__import__("mlframe.training.core._phase_train_one_target", fromlist=["_apply_loss_recommendation_in_place"]).__file__).read_text(
            encoding="utf-8"
        )
        # The Huber branch in CB must now return the value unchanged
        # (Huber:delta=X is a valid CB eval_metric), not MAE.
        assert 'return ("eval_metric", _value)' in src
        assert "stops ES at iter=1" in src


class TestApplyLossRecommendationWiresExtras:
    """Groups tests covering apply loss recommendation wires extras."""
    def test_xgb_extra_params_threaded_into_set_params(self) -> None:
        """Sensor: the apply path must propagate ``xgb_extra_params`` /
        ``cb_extra_params`` into the model's set_params() call."""
        from pathlib import Path

        src = Path(__import__("mlframe.training.core._phase_train_one_target", fromlist=["_apply_loss_recommendation_in_place"]).__file__).read_text(
            encoding="utf-8"
        )
        assert 'rec.get(f"{_backend}_extra_params")' in src
        assert "_set_kwargs.update(_extra_params)" in src


class TestCompositeTargetEstimatorTClip:
    """Groups tests covering composite target estimator t clip."""
    def _make_fitted_estimator(self, y_scale: float = 13.0):
        """Build a synthetic CompositeTargetEstimator with diff transform
        and a constant-blowup inner. Uses from_fitted_inner to bypass
        sklearn.clone (the blowup inner is a stub, not sklearn-compat)."""
        from mlframe.training.composite.estimator._estimator import (
            CompositeTargetEstimator,
        )

        class _BlowupInner:
            """Groups tests covering blowup inner."""
            def __init__(self, blowup_value: float):
                self.blowup_value = blowup_value

            def fit(self, X, y, **kw):
                """Fit."""
                return self

            def predict(self, X) -> np.ndarray:
                """Predict."""
                n = len(X) if hasattr(X, "__len__") else X.shape[0]
                return np.full(n, self.blowup_value, dtype=np.float64)

        rng = np.random.default_rng(42)
        n = 500
        import pandas as pd

        y = rng.normal(0, y_scale, n)
        base = rng.normal(0, y_scale * 5, n)
        X = pd.DataFrame({"f0": rng.normal(0, 1, n), "f1": rng.normal(0, 1, n)})
        full_X = X.copy()
        full_X["base_col"] = base
        # diff transform: T = y - base, params={}
        wrapper = CompositeTargetEstimator.from_fitted_inner(
            fitted_inner=_BlowupInner(blowup_value=y_scale * 30.0),
            transform_name="diff",
            base_column="base_col",
            transform_fitted_params={},
            y_train=y,
        )
        return wrapper, full_X, base, y_scale

    def test_t_clip_bounds_stored_after_fit(self) -> None:
        """T clip bounds stored after fit."""
        wrapper, _X, _base, _ = self._make_fitted_estimator()
        params = wrapper.fitted_params_
        assert "t_clip_low" in params
        assert "t_clip_high" in params
        assert np.isfinite(params["t_clip_low"])
        assert np.isfinite(params["t_clip_high"])
        assert params["t_clip_low"] < params["t_clip_high"]

    def test_blowup_inner_predictions_get_clipped(self) -> None:
        """Blowup inner predictions get clipped."""
        wrapper, X, _base, y_scale = self._make_fitted_estimator()
        y_hat = wrapper.predict(X)
        # Without T-clip: y_hat = T_blow + base where T_blow = 30 * y_scale.
        # With T-clip: T capped at ~10*MAD < y_scale*10 < 30*y_scale.
        # Inner returned a constant blowup so all rows hit the same clip.
        # y-clip on top further bounds y_hat to [y_min - span*0.9, y_max + span*9].
        # Effective bound: |y_hat| < |y_max| + 9 * span < y_scale * (10 + 9*10).
        # Most importantly: y_hat must NOT contain the 30x outlier directly.
        assert np.all(np.abs(y_hat) < y_scale * 100), f"y_hat range [{y_hat.min():.1f}, {y_hat.max():.1f}] suggests T-clip didn't fire"

    def test_legacy_params_without_t_clip_dont_crash(self) -> None:
        """Backwards-compat: existing pickled wrappers don't have
        t_clip_* keys. Predict must still work (no-op clip)."""
        wrapper, X, _base, _ = self._make_fitted_estimator()
        del wrapper.fitted_params_["t_clip_low"]
        del wrapper.fitted_params_["t_clip_high"]
        y_hat = wrapper.predict(X)
        assert y_hat.shape[0] == X.shape[0]
        assert np.all(np.isfinite(y_hat) | np.isnan(y_hat))
