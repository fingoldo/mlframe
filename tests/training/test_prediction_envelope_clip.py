"""Tests for the generic prediction-envelope clip phase.

Bounds all regression-model predictions to a 3-sigma window around the
train target range BEFORE downstream metric / chart / ensemble work.

Documented failure modes the clip protects against:
  * Identity-MLP / LeakyReLU-MLP without batch-norm extrapolating on
    group-aware splits (R^2 < -200 observed in 3 separate runs).
  * Ridge / Lasso on composite (Yeo-Johnson residual) targets producing
    y-scale predictions hundreds of sigma below the train range
    (R^2=-6934, MaxError=1.4M observed).

The clip is a safety net. The MODEL still extrapolates badly; the clip
just stops the damage from poisoning charts, RMSE / MaxError, and the
ensemble stacker.
"""
from __future__ import annotations

import numpy as np
import pytest


class TestComputeTrainEnvelopeStats:
    def test_finite_y_train(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            compute_train_envelope_stats,
        )
        y = np.array([10.0, 11.0, 12.0, 13.0, 14.0] * 100)
        stats = compute_train_envelope_stats(y)
        assert stats is not None
        assert stats.y_min == pytest.approx(10.0)
        assert stats.y_max == pytest.approx(14.0)
        assert stats.y_std > 0

    def test_too_few_finite(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            compute_train_envelope_stats,
        )
        stats = compute_train_envelope_stats(np.array([1.0, 2.0, 3.0]))
        assert stats is None

    def test_zero_variance(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            compute_train_envelope_stats,
        )
        stats = compute_train_envelope_stats(np.full(100, 5.0))
        assert stats is None

    def test_skips_non_finite_rows(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            compute_train_envelope_stats,
        )
        y = np.array([10.0, 11.0, np.nan, np.inf, 14.0] * 50)
        stats = compute_train_envelope_stats(y)
        assert stats is not None
        assert stats.y_min == pytest.approx(10.0)
        assert stats.y_max == pytest.approx(14.0)


class TestClipPredictionsToTrainEnvelope:
    def _stats(self, y_min=10.0, y_max=20.0, y_std=2.0):
        from mlframe.training._prediction_envelope_clip import (
            TrainEnvelopeStats,
        )
        return TrainEnvelopeStats(y_min=y_min, y_max=y_max, y_std=y_std)

    def test_in_envelope_preds_pass_through(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )
        # y_train [10, 20], std=2 -> envelope [10 - 3*2, 20 + 3*2] = [4, 26]
        preds = np.array([11.0, 15.0, 19.0, 25.0, 5.0])
        out = clip_predictions_to_train_envelope(preds, self._stats())
        np.testing.assert_allclose(out, preds)

    def test_blow_up_above_envelope_clipped(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )
        preds = np.array([15.0, 1000.0, 10000.0])
        out = clip_predictions_to_train_envelope(preds, self._stats())
        # envelope upper = 20 + 3*2 = 26
        assert out[0] == pytest.approx(15.0)
        assert out[1] == pytest.approx(26.0)
        assert out[2] == pytest.approx(26.0)

    def test_blow_up_below_envelope_clipped(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )
        preds = np.array([-1e5, -5e4, 15.0])
        out = clip_predictions_to_train_envelope(preds, self._stats())
        # envelope lower = 10 - 3*2 = 4
        assert out[0] == pytest.approx(4.0)
        assert out[1] == pytest.approx(4.0)
        assert out[2] == pytest.approx(15.0)

    def test_none_stats_noop(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )
        preds = np.array([1e9, -1e9])
        out = clip_predictions_to_train_envelope(preds, None)
        np.testing.assert_allclose(out, preds)

    def test_apply_clip_false_noop(self) -> None:
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )
        preds = np.array([1e9, -1e9])
        out = clip_predictions_to_train_envelope(
            preds, self._stats(), apply_clip=False,
        )
        np.testing.assert_allclose(out, preds)

    def test_env_var_disable(self, monkeypatch) -> None:
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )
        monkeypatch.setenv("MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP", "1")
        preds = np.array([1e9, -1e9])
        out = clip_predictions_to_train_envelope(preds, self._stats())
        np.testing.assert_allclose(out, preds)

    def test_warn_message_on_clip_fire(self, caplog) -> None:
        import logging
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )
        with caplog.at_level(logging.WARNING):
            clip_predictions_to_train_envelope(
                np.array([1e6, 15.0, -1e6]),
                self._stats(),
                model_label="Ridge",
                split_label="TEST",
            )
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "Ridge" in joined
        assert "TEST" in joined
        assert "1 row(s) below" in joined
        assert "1 row(s) above" in joined

    def test_idempotent_double_clip(self) -> None:
        """Clipping twice must equal clipping once (no drift)."""
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )
        preds = np.array([1e6, 15.0, -1e6])
        stats = self._stats()
        once = clip_predictions_to_train_envelope(preds, stats)
        twice = clip_predictions_to_train_envelope(once, stats)
        np.testing.assert_allclose(once, twice)


class TestReportingIntegration:
    """Confirm report_regression_model_perf wires the clip when
    y_train_{min,max,std} kwargs are supplied."""

    def test_source_has_envelope_clip_wiring(self) -> None:
        from pathlib import Path
        from mlframe.training import _reporting as rep
        src = Path(rep.__file__).read_text(encoding="utf-8")
        assert "_prediction_envelope_clip" in src
        assert "clip_predictions_to_train_envelope" in src

    def test_clip_invoked_only_when_stats_supplied(self) -> None:
        """Legacy callers that didn't pass y_train_{min,max,std} get a
        no-op (back-compat)."""
        from pathlib import Path
        from mlframe.training import _reporting as rep
        src = Path(rep.__file__).read_text(encoding="utf-8")
        # The clip block must be gated on the three kwargs being non-None.
        assert "if (y_train_min is not None and y_train_max is not None" in src
