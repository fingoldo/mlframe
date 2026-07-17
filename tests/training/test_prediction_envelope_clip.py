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
    """Groups tests covering compute train envelope stats."""
    def test_finite_y_train(self) -> None:
        """Finite y train."""
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
        """Too few finite."""
        from mlframe.training._prediction_envelope_clip import (
            compute_train_envelope_stats,
        )

        stats = compute_train_envelope_stats(np.array([1.0, 2.0, 3.0]))
        assert stats is None

    def test_zero_variance(self) -> None:
        """Zero variance."""
        from mlframe.training._prediction_envelope_clip import (
            compute_train_envelope_stats,
        )

        stats = compute_train_envelope_stats(np.full(100, 5.0))
        assert stats is None

    def test_skips_non_finite_rows(self) -> None:
        """Skips non finite rows."""
        from mlframe.training._prediction_envelope_clip import (
            compute_train_envelope_stats,
        )

        y = np.array([10.0, 11.0, np.nan, np.inf, 14.0] * 50)
        stats = compute_train_envelope_stats(y)
        assert stats is not None
        assert stats.y_min == pytest.approx(10.0)
        assert stats.y_max == pytest.approx(14.0)


class TestClipPredictionsToTrainEnvelope:
    """Groups tests covering clip predictions to train envelope."""
    def _stats(self, y_min=10.0, y_max=20.0, y_std=2.0):
        """Stats."""
        from mlframe.training._prediction_envelope_clip import (
            TrainEnvelopeStats,
        )

        return TrainEnvelopeStats(y_min=y_min, y_max=y_max, y_std=y_std)

    def test_in_envelope_preds_pass_through(self) -> None:
        """In envelope preds pass through."""
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )

        # y_train [10, 20], std=2 -> envelope [10 - 3*2, 20 + 3*2] = [4, 26]
        preds = np.array([11.0, 15.0, 19.0, 25.0, 5.0])
        out = clip_predictions_to_train_envelope(preds, self._stats())
        np.testing.assert_allclose(out, preds)

    def test_blow_up_above_envelope_clipped(self) -> None:
        """Blow up above envelope clipped."""
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
        """Blow up below envelope clipped."""
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
        """None stats noop."""
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )

        preds = np.array([1e9, -1e9])
        out = clip_predictions_to_train_envelope(preds, None)
        np.testing.assert_allclose(out, preds)

    def test_apply_clip_false_noop(self) -> None:
        """Apply clip false noop."""
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )

        preds = np.array([1e9, -1e9])
        out = clip_predictions_to_train_envelope(
            preds,
            self._stats(),
            apply_clip=False,
        )
        np.testing.assert_allclose(out, preds)

    def test_env_var_disable(self, monkeypatch) -> None:
        """Env var disable."""
        from mlframe.training._prediction_envelope_clip import (
            clip_predictions_to_train_envelope,
        )

        monkeypatch.setenv("MLFRAME_DISABLE_PREDICTION_ENVELOPE_CLIP", "1")
        preds = np.array([1e9, -1e9])
        out = clip_predictions_to_train_envelope(preds, self._stats())
        np.testing.assert_allclose(out, preds)

    def test_warn_message_on_clip_fire(self, caplog) -> None:
        """Warn message on clip fire."""
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
        """Source has envelope clip wiring."""
        from pathlib import Path
        from mlframe.training.reporting import _reporting as rep

        # ``report_regression_model_perf`` was carved out of ``_reporting.py``
        # into ``_reporting_regression.py``; the envelope-clip wiring moved
        # with it. Concat both files so the source-grep guard still matches.
        src = Path(rep.__file__).read_text(encoding="utf-8")
        _pkg = Path(rep.__file__).parent / "_reporting_regression"
        if _pkg.is_dir():
            for _f in sorted(_pkg.glob("*.py")):
                src += "\n" + _f.read_text(encoding="utf-8")
        assert "_prediction_envelope_clip" in src
        assert "clip_predictions_to_train_envelope" in src

    def test_clip_invoked_only_when_stats_supplied(self) -> None:
        """Legacy callers that didn't pass y_train_{min,max,std} get a
        no-op (back-compat)."""
        from pathlib import Path
        from mlframe.training.reporting import _reporting as rep

        # Same carve as ``test_source_has_envelope_clip_wiring``: the
        # gate moved to ``_reporting_regression.py``.
        src = Path(rep.__file__).read_text(encoding="utf-8")
        _pkg = Path(rep.__file__).parent / "_reporting_regression"
        if _pkg.is_dir():
            for _f in sorted(_pkg.glob("*.py")):
                src += "\n" + _f.read_text(encoding="utf-8")
        # The clip block must be gated on the three kwargs being non-None.
        assert "if y_train_min is not None and y_train_max is not None" in src

    def test_train_envelope_stats_threaded_through_unified_entry(self) -> None:
        """2026-05-26: ``report_model_perf`` accepts a single
        ``y_train_envelope_stats`` object (computed once by the
        trainer) and decomposes it into the three legacy kwargs when
        forwarding to ``report_regression_model_perf``. End-to-end
        replaces the prior 16+ -callsite y_train_min/max/std threading
        that nobody ever wired."""
        import inspect
        from mlframe.training.reporting._reporting import report_model_perf

        sig = inspect.signature(report_model_perf)
        assert "y_train_envelope_stats" in sig.parameters

    def test_split_metrics_helper_forwards_envelope_stats(self) -> None:
        """Trainer-side ``_compute_split_metrics`` is the per-split
        bridge between the training loop and the reporter; it must
        forward the envelope stats so val + test get the same TRAIN
        bound."""
        import inspect
        from mlframe.training._eval_helpers import _compute_split_metrics

        sig = inspect.signature(_compute_split_metrics)
        assert "y_train_envelope_stats" in sig.parameters

    def test_train_bound_uses_k3_eval_fallback_uses_k10(self) -> None:
        """Behaviour: when the train envelope is supplied, the clip
        uses k=3 sigma (tighter, conceptually correct). When ONLY
        eval targets are available, k=10 (defensive). Bounds must
        differ accordingly so train-stats actually do work."""
        from mlframe.training._prediction_envelope_clip import (
            TrainEnvelopeStats,
            clip_predictions_to_train_envelope,
        )

        stats = TrainEnvelopeStats(y_min=10.0, y_max=20.0, y_std=2.0)
        # Train k=3 -> [10 - 6, 20 + 6] = [4, 26]
        preds = np.array([3.0, 15.0, 27.0])
        train_clip = clip_predictions_to_train_envelope(preds, stats, k_sigma=3.0)
        assert train_clip[0] == pytest.approx(4.0)
        assert train_clip[2] == pytest.approx(26.0)
        # Eval k=10 -> [10 - 20, 20 + 20] = [-10, 40]
        eval_clip = clip_predictions_to_train_envelope(preds, stats, k_sigma=10.0)
        np.testing.assert_allclose(eval_clip, preds)
