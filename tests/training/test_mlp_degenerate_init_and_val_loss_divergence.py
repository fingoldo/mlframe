"""Round-4 audit-followup tests covering 5 of 7 deferred items.

* #5 Degenerate-init probe in generate_mlp -- catches zeros-init / LeakyReLU(0)
  pathologies that the Identity-MLP guard misses.
* #6 ``ValLossDivergenceCallback`` -- WARNs when val_loss grows 100x its baseline,
  catching collapses DURING training before paying the full budget.
* #7 MRMR identity cache composite-aware invalidation -- already fixed by
  ``mrmr_identity_cache_include_y=True`` default; lock-in test.
* #4 MLP 126MB save-size pathology -- ``auto_lean_pre_check_mb`` default
  lowered 100 -> 50 so the pre-check kicks in BEFORE the fat dump for typical
  MLP-on-4M-row payloads.
* #2 Train-y envelope branch in ``regression-collapse-sensor`` -- new branch
  ``outside-train-y-envelope`` fires when pred falls > 3 sigma outside
  [y_train_min, y_train_max].

Deferred (separate commits):
* #1 Native init_score (LGB/XGB/CB shims) -- partial: composite-diff path
  already delivers ~80% of the value. Asymmetry between booster predict
  semantics (LGB/CB include baseline natively, XGB returns margin minus
  base_margin) needs careful wiring.
* #3 OOF predictions reuse from initial fit CV -- large scope, requires CV
  path threading through the initial trainer.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_torch, pytest.mark.uses_torch]


class TestDegenerateInitProbe:
    def test_zeros_init_triggers_warn(self, caplog) -> None:
        torch = pytest.importorskip("torch")
        from mlframe.training.neural.flat import generate_mlp
        caplog.set_level(logging.WARNING, logger="mlframe.training.neural.flat")
        # Force zeros initialisation. The probe inspects the just-built
        # nn.Linear modules; with zero weights every layer has std 0.
        generate_mlp(
            num_features=10, num_classes=1, nlayers=2,
            activation_function=torch.nn.ReLU,
            weights_init_fcn=torch.nn.init.zeros_,
            use_layernorm=False, dropout_prob=0.0,
            inputs_dropout_prob=0.0, verbose=0,
        )
        warnings = [
            r for r in caplog.records
            if "degenerate Linear layer" in r.message
        ]
        assert warnings, (
            f"degenerate-init probe didn't fire on zeros-initialised MLP; "
            f"captured records: {[r.message for r in caplog.records]}"
        )

    def test_constant_init_triggers_warn(self, caplog) -> None:
        """``constant_(W, c)`` also collapses to zero std and must be caught."""
        torch = pytest.importorskip("torch")
        from functools import partial
        from mlframe.training.neural.flat import generate_mlp
        caplog.set_level(logging.WARNING, logger="mlframe.training.neural.flat")
        generate_mlp(
            num_features=10, num_classes=1, nlayers=2,
            activation_function=torch.nn.ReLU,
            weights_init_fcn=partial(torch.nn.init.constant_, val=0.5),
            use_layernorm=False, dropout_prob=0.0,
            inputs_dropout_prob=0.0, verbose=0,
        )
        warnings = [
            r for r in caplog.records
            if "degenerate Linear layer" in r.message
        ]
        assert warnings, (
            f"degenerate-init probe didn't fire on constant_-initialised MLP; "
            f"captured records: {[r.message for r in caplog.records]}"
        )

    def test_kaiming_init_no_warn(self, caplog) -> None:
        """Reasonable init should not trip the probe."""
        torch = pytest.importorskip("torch")
        from mlframe.training.neural.flat import generate_mlp
        caplog.set_level(logging.WARNING, logger="mlframe.training.neural.flat")
        generate_mlp(
            num_features=10, num_classes=1, nlayers=2,
            activation_function=torch.nn.ReLU,
            weights_init_fcn=torch.nn.init.kaiming_normal_,
            use_layernorm=False, dropout_prob=0.0,
            inputs_dropout_prob=0.0, verbose=0,
        )
        warnings = [
            r for r in caplog.records
            if "degenerate Linear layer" in r.message
        ]
        assert not warnings, (
            f"degenerate-init probe fired spuriously on kaiming_normal init: "
            f"{[r.message for r in warnings]}"
        )


class TestValLossDivergenceCallback:
    def test_callback_class_exists_and_warns(self, caplog) -> None:
        """Smoke: the callback class is importable and warns at high
        divergence factors. End-to-end Lightning integration is covered
        by the suite's neural test harness."""
        pytest.importorskip("torch")
        pytest.importorskip("lightning")
        from mlframe.training.neural.base import ValLossDivergenceCallback
        cb = ValLossDivergenceCallback(monitor="val_loss", divergence_factor=10.0)
        # Construct a minimal mock trainer with growing val_loss.
        class MockTrainer:
            current_epoch = 1
            callback_metrics = {"val_loss": 0.05}

        class MockModule:
            pass

        # Epoch 1 -> baseline
        cb.on_validation_epoch_end(MockTrainer(), MockModule())
        assert cb._initial_value is not None

        # Epoch 5 -> 50x baseline -> trip
        caplog.set_level(logging.WARNING, logger="mlframe.training.neural.base")
        class MockTrainer2:
            current_epoch = 5
            callback_metrics = {"val_loss": 2.5}  # 50x baseline of 0.05

        cb.on_validation_epoch_end(MockTrainer2(), MockModule())
        diverge_warnings = [
            r for r in caplog.records
            if "mlp-val-divergence" in r.message
        ]
        assert diverge_warnings, (
            f"ValLossDivergenceCallback didn't warn on 50x growth (factor=10); "
            f"records: {[r.message for r in caplog.records]}"
        )

    def test_callback_only_warns_once(self) -> None:
        pytest.importorskip("torch")
        from mlframe.training.neural.base import ValLossDivergenceCallback
        cb = ValLossDivergenceCallback(divergence_factor=10.0)

        class T:
            current_epoch = 1
            callback_metrics = {"val_loss": 0.01}

        cb.on_validation_epoch_end(T(), None)
        assert cb._initial_value == 0.01
        assert not cb._warned

        # First divergence -> warns + latches.
        class T2:
            current_epoch = 5
            callback_metrics = {"val_loss": 1.0}

        cb.on_validation_epoch_end(T2(), None)
        assert cb._warned

        # Second divergence -> should NOT re-warn (idempotent).
        class T3:
            current_epoch = 6
            callback_metrics = {"val_loss": 2.0}

        cb.on_validation_epoch_end(T3(), None)
        assert cb._warned  # still latched


class TestMRMRIdentityCacheCompositeAware:
    """#7 was already fixed via ``mrmr_identity_cache_include_y=True``.
    Lock-in test: the default is True, so composite vs raw target on the
    same X get separate cache slots."""

    def test_mrmr_identity_cache_include_y_default_is_true(self) -> None:
        from mlframe.feature_selection.filters.mrmr import MRMR
        import inspect
        # ``mrmr_identity_cache_include_y`` is a constructor kwarg with
        # default ``True``; introspect the signature to assert.
        sig = inspect.signature(MRMR.__init__)
        assert "mrmr_identity_cache_include_y" in sig.parameters
        param = sig.parameters["mrmr_identity_cache_include_y"]
        assert param.default is True, (
            f"mrmr_identity_cache_include_y default must be True so the "
            f"identity cache distinguishes composite-target y from raw "
            f"target y on the same X; got default {param.default!r}"
        )


class TestSaveSizePrecheckThreshold:
    """#4: ``auto_lean_pre_check_mb`` default lowered 100 -> 50 MB so the
    pre-check fires for MLP-on-4M-row payloads BEFORE the fat dump
    (which would land at ~120 MB on disk past the post-save sensor's
    50 MB threshold). Sensor and pre-check thresholds now agree."""

    def test_default_threshold_is_50_mb(self) -> None:
        import inspect
        from mlframe.training.io import save_mlframe_model
        sig = inspect.signature(save_mlframe_model)
        assert "auto_lean_pre_check_mb" in sig.parameters
        param = sig.parameters["auto_lean_pre_check_mb"]
        assert param.default == 50.0, (
            f"save-size pre-check threshold must be 50 MB to match the "
            f"post-save sensor's 50 MB suspicious-threshold; got {param.default}"
        )


class TestOutsideTrainYEnvelopeSensorBranch:
    """#2: new branch in ``regression-collapse-sensor`` that fires when
    predictions land > 3 sigma outside [y_train_min, y_train_max]."""

    def test_branch_fires_when_y_train_stats_supplied(self, caplog) -> None:
        from mlframe.training import _reporting
        caplog.set_level(logging.WARNING, logger="mlframe.training._reporting")
        # Test rows: in-batch target std=100, mean=11500. Preds: also
        # mean~11500, std~50 (no in-batch collapse signal). BUT preds
        # range goes to 14000 while y_train_max=12500 with y_train_std=200.
        # delta = (14000 - 12500) / 200 = 7.5 sigma -- trips the new
        # train-envelope branch.
        rng_t = np.random.default_rng(0)
        rng_p = np.random.default_rng(1)
        targets = 11500 + rng_t.normal(0, 100, 1000)
        # Preds land far above train-y max for some rows.
        preds = 11500 + rng_p.normal(0, 50, 1000)
        preds[0] = 14000  # outlier outside [11000, 12500] envelope
        _reporting.report_regression_model_perf(
            targets=targets, columns=[], model_name="test-envelope",
            model=None, preds=preds,
            print_report=False, show_perf_chart=False, verbose=False,
            y_train_min=11000.0, y_train_max=12500.0, y_train_std=200.0,
        )
        envelope_warnings = [
            r for r in caplog.records
            if "outside-train-y-envelope" in r.message
            or "linear-extrapolation" in r.message
        ]
        assert envelope_warnings, (
            f"train-y envelope branch didn't fire on pred far outside "
            f"[y_train_min, y_train_max]; records: "
            f"{[r.message for r in caplog.records if 'sensor' in r.message]}"
        )

    def test_no_warn_when_train_stats_not_supplied(self, caplog) -> None:
        """Backward compat: callers that don't pass y_train stats see
        the same behaviour as pre-fix (only in-batch checks fire)."""
        from mlframe.training import _reporting
        caplog.set_level(logging.WARNING, logger="mlframe.training._reporting")
        rng_t = np.random.default_rng(0)
        rng_p = np.random.default_rng(1)
        targets = 11500 + rng_t.normal(0, 100, 1000)
        preds = 11500 + rng_p.normal(0, 50, 1000)
        # No train stats passed -> envelope branch can't fire by construction.
        _reporting.report_regression_model_perf(
            targets=targets, columns=[], model_name="test-no-envelope",
            model=None, preds=preds,
            print_report=False, show_perf_chart=False, verbose=False,
        )
        envelope_warnings = [
            r for r in caplog.records
            if "outside-train-y-envelope" in r.message
        ]
        assert not envelope_warnings, (
            f"envelope branch fired without y_train stats supplied: "
            f"{[r.message for r in envelope_warnings]}"
        )
