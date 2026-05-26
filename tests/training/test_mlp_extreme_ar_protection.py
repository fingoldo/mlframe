"""Sensor / regression tests for the MLP extreme-AR + group-aware
protection shipped 2026-05-26.

Two protections, two test classes:

1. **Skip-by-default**: ``TrainingBehaviorConfig.mlp_extreme_ar_group_aware_skip``
   default flipped from False to True. The Identity-MLP / LeakyReLU-MLP
   failure mode on extreme-AR (lag1_corr >= 0.99) + group-aware splits
   produces R²<-200 reliably across prod incidents (2026-05-22, -24, -26).
   The skip avoids ~3 min train + 126 MB checkpoint waste; the ensemble's
   dummy-floor gate drops the bad MLP from the blend anyway.

2. **Defensive y-clip**: ``_TTRWithEvalSetScaling.predict`` clips
   inverse-transformed predictions to ``[y_train_min - 3*std,
   y_train_max + 3*std]``. Catches the failure mode when the operator
   opts out of #1 (or runs an MLP outside the suite gate). Bounds the
   damage to "wrong by a few sigma" instead of "wrong by 1000 sigma".
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


class TestMlpExtremeArSkipDefault:
    def test_default_is_False(self) -> None:
        """Default OFF: turning MLP off is not the fix. Damage is bounded
        by the TTR predict clip + ensemble dummy-floor gate. User has
        asked the framework to make MLP actually work on extreme-AR +
        group-aware regimes (substantive fix paths in the comment), not
        silently skip."""
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig()
        assert cfg.mlp_extreme_ar_group_aware_skip is False
        assert cfg.mlp_extreme_ar_threshold == pytest.approx(0.99)

    def test_can_opt_in(self) -> None:
        """The knob still exists and accepts opt-in for users who
        explicitly want to skip the MLP fit (e.g. tight time budget)."""
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig(mlp_extreme_ar_group_aware_skip=True)
        assert cfg.mlp_extreme_ar_group_aware_skip is True

    def test_threshold_configurable(self) -> None:
        from mlframe.training._model_configs import TrainingBehaviorConfig
        cfg = TrainingBehaviorConfig(mlp_extreme_ar_threshold=0.95)
        assert cfg.mlp_extreme_ar_threshold == pytest.approx(0.95)


class TestTtrPredictClip:
    def _fit_ttr(self, y_train: np.ndarray):
        """Helper: fit a TTR with StandardScaler on synthetic y."""
        from sklearn.preprocessing import StandardScaler
        from mlframe.training._ttr_eval_set_scaling import _TTRWithEvalSetScaling

        class _MockRegressor:
            """sklearn-compat: scaled-space identity on the first feature.
            Used to inject controllable T_hat predictions."""

            def __init__(self):
                self._t_hat_to_return: np.ndarray | None = None

            def fit(self, X, y, **kw):
                self._n_features = X.shape[1] if hasattr(X, "shape") else 1
                return self

            def predict(self, X, **kw):
                if self._t_hat_to_return is not None:
                    return self._t_hat_to_return
                n = len(X) if hasattr(X, "__len__") else X.shape[0]
                return np.zeros(n, dtype=np.float64)

            def get_params(self, deep=True):
                return {}

            def set_params(self, **p):
                return self

        ttr = _TTRWithEvalSetScaling(
            regressor=_MockRegressor(),
            transformer=StandardScaler(),
        )
        rng = np.random.default_rng(0)
        X_train = rng.normal(0, 1, (len(y_train), 2)).astype(np.float64)
        ttr.fit(X_train, y_train)
        return ttr, X_train

    def test_clip_stats_stashed_after_fit(self) -> None:
        y_train = np.linspace(10500, 12800, 1000)
        ttr, _ = self._fit_ttr(y_train)
        assert hasattr(ttr, "_y_train_clip_low_")
        assert hasattr(ttr, "_y_train_clip_high_")
        # 3*std around the range; should bracket [y_min, y_max].
        assert ttr._y_train_clip_low_ < y_train.min()
        assert ttr._y_train_clip_high_ > y_train.max()

    def test_clip_fires_on_blow_up_predictions(self) -> None:
        """Inject T_hat = 100 sigma (way past 3 sigma envelope). After
        inverse_transform + clip, predictions land at the train-envelope
        boundary, not at the raw 100-sigma value."""
        y_train = np.linspace(10500, 12800, 1000).astype(np.float64)
        ttr, X_train = self._fit_ttr(y_train)
        # In scaled space, mean=0, std=1 (StandardScaler). Inject T_hat=100
        # which after inverse_transform = 100 * std + mean. With y_train
        # std ~ 660 and mean ~ 11650, raw prediction would be ~ 11650 +
        # 66000 = 77650. Clipping caps at y_max + 3 * std ~ 12800 + 1980
        # = 14780.
        n_pred = 50
        ttr.regressor_._t_hat_to_return = np.full(n_pred, 100.0)
        X_pred = X_train[:n_pred]
        preds = ttr.predict(X_pred)
        # Should NOT contain the un-clipped ~77650 value.
        assert preds.max() < ttr._y_train_clip_high_ + 1e-6
        # And the clip should leave preds near (not exactly at because
        # multiple clipped points all stack at the boundary) the high bound.
        assert preds.max() == pytest.approx(ttr._y_train_clip_high_, abs=1.0)

    def test_clip_noop_on_in_distribution_predictions(self) -> None:
        """When T_hat lands in-distribution, clip is a no-op."""
        y_train = np.linspace(10500, 12800, 1000).astype(np.float64)
        ttr, X_train = self._fit_ttr(y_train)
        ttr.regressor_._t_hat_to_return = np.zeros(50)  # mean of scaled = 0
        X_pred = X_train[:50]
        preds = ttr.predict(X_pred)
        # 0 in scaled space ~ y_mean ~ 11650 in y-space; well within
        # [10500-3*std, 12800+3*std].
        assert preds.min() > ttr._y_train_clip_low_
        assert preds.max() < ttr._y_train_clip_high_

    def test_clip_disable_env_var(self, monkeypatch) -> None:
        """Setting MLFRAME_TTR_DISABLE_PREDICT_CLIP=1 disables the clip
        (for benchmarking the failure mode)."""
        monkeypatch.setenv("MLFRAME_TTR_DISABLE_PREDICT_CLIP", "1")
        y_train = np.linspace(10500, 12800, 1000).astype(np.float64)
        ttr, X_train = self._fit_ttr(y_train)
        ttr.regressor_._t_hat_to_return = np.full(50, 100.0)
        preds = ttr.predict(X_train[:50])
        # Without clipping the raw 100-sigma value flows through.
        assert preds.max() > ttr._y_train_clip_high_ + 100


class TestSourceIntegrity:
    """Source-grep sensors so refactors that drop the protection get
    caught early."""

    def test_model_configs_has_skip_default_False(self) -> None:
        from mlframe.training import _model_configs as mc
        src = Path(mc.__file__).read_text(encoding="utf-8")
        assert "mlp_extreme_ar_group_aware_skip: bool = False" in src

    def test_ttr_module_has_y_train_clip(self) -> None:
        from mlframe.training import _ttr_eval_set_scaling as ttr
        src = Path(ttr.__file__).read_text(encoding="utf-8")
        assert "_y_train_clip_low_" in src
        assert "_y_train_clip_high_" in src
        assert "MLFRAME_TTR_DISABLE_PREDICT_CLIP" in src
