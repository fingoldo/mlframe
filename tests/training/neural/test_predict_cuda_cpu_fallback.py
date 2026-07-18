"""Regression: ``prediction_trainer.predict(...)`` CUDA-runtime errors
(``illegal memory access`` / OOM / ``device-side assert``) must trigger
a one-shot CPU fallback inside ``PytorchLightningEstimator._predict_
common`` instead of propagating and killing the whole suite.

Driven by iter293 (2026-05-26): concurrent GPU usage on a single host
during a multi-process fuzz run made predict step hit
``CUDA error: an illegal memory access was encountered`` and the
exception killed the whole train/predict cycle for that combo. The
narrow CPU retry catches CUDA-fingerprinted RuntimeError and runs
``Trainer(accelerator='cpu').predict`` exactly once; the underlying
CUDA fault stays visible as a WARNING.

These tests monkey-patch ``L.Trainer.predict`` so the fault is
deterministic on any host regardless of GPU availability.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")


def _make_regressor():
    """Mirror ``test_predict_path_eval_and_datamodule._make_regressor``."""
    import torch
    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningRegressor,
        TorchDataModule,
    )

    return PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 2},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={"max_epochs": 1, "logger": False, "accelerator": "cpu", "devices": 1},
    )


def test_predict_cpu_fallback_on_cuda_runtime_error(caplog, monkeypatch):
    """First Trainer.predict call raises a CUDA RuntimeError; estimator
    must retry on CPU and the second call must succeed."""
    import lightning as L
    import torch

    invocations = {"count": 0}

    _orig_predict = L.Trainer.predict

    def _fake_predict(self, *args, **kwargs):
        """Fake predict."""
        invocations["count"] += 1
        if invocations["count"] == 1:
            raise RuntimeError(
                "CUDA error: an illegal memory access was encountered\n"
                "CUDA kernel errors might be asynchronously reported at some "
                "other API call, so the stacktrace below might be incorrect."
            )
        # Second call (CPU fallback) -- return a dummy 1-batch prediction.
        # Use real torch.zeros so downstream torch.cat works.
        return [torch.zeros(8, 1)]

    np.random.seed(0)
    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    est = _make_regressor()
    est.fit(X, y, eval_set=(X, y))

    # Reset call count -- ``fit()`` calls trainer.fit() not predict(), so
    # the counter is still 0, but be defensive in case fit() did probe.
    invocations["count"] = 0
    monkeypatch.setattr(L.Trainer, "predict", _fake_predict)

    # Force the predict path to think the trainer wanted CUDA so the
    # narrow ``trainer_params.get('accelerator') in ('cuda', 'gpu', 'auto')``
    # filter fires. We mutate the post-fit instance.
    est.trainer_params = {**est.trainer_params, "accelerator": "cuda"}

    caplog.set_level(logging.WARNING, logger="mlframe.training.neural.base")
    predictions = est.predict(X)

    assert invocations["count"] == 2, f"expected 2 predict invocations (cuda fail + cpu retry), got {invocations['count']}"
    assert predictions is not None
    assert len(predictions) > 0
    fallback_logs = [r for r in caplog.records if "retrying on CPU" in r.message]
    assert fallback_logs, f"expected a WARNING about CPU retry after CUDA error; got: {[r.message for r in caplog.records]}"


def test_predict_3rd_tier_cuda_fail_moves_model_to_cpu_before_retry(caplog, monkeypatch):
    """iter420 regression: when the FIRST CPU fallback ALSO raises a CUDA
    error (context already invalidated), the recovery path hides CUDA at
    the torch module level AND moves ``self.model`` to CPU. Pre-fix the
    third-tier retry left model parameters on the invalidated GPU and
    Lightning re-raised the same illegal-memory-access. Surfaced on c0005
    LTR 2026-05-27."""
    import lightning as L
    import torch

    invocations = {"count": 0}
    moved_to_cpu = {"yes": False}

    def _fake_predict(self, *args, **kwargs):
        """Fake predict."""
        invocations["count"] += 1
        if invocations["count"] in (1, 2):
            # First call: GPU path. Second call: first CPU retry, which
            # also fails with CUDA fingerprint (context invalidated).
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        # Third call: only succeeds if model is now on CPU.
        return [torch.zeros(8, 1)]

    np.random.seed(0)
    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    est = _make_regressor()
    est.fit(X, y, eval_set=(X, y))

    # Wrap .to so we can observe whether the recovery path moved the model.
    _orig_to = est.model.to

    def _spy_to(*a, **kw):
        """Spy to."""
        if a and a[0] == "cpu":
            moved_to_cpu["yes"] = True
        return _orig_to(*a, **kw)

    monkeypatch.setattr(est.model, "to", _spy_to)

    invocations["count"] = 0
    monkeypatch.setattr(L.Trainer, "predict", _fake_predict)
    est.trainer_params = {**est.trainer_params, "accelerator": "cuda"}

    caplog.set_level(logging.ERROR, logger="mlframe.training.neural.base")
    predictions = est.predict(X)

    assert invocations["count"] == 3, f"expected 3 predict invocations (cuda fail + CPU fail + CPU retry), got {invocations['count']}"
    assert moved_to_cpu["yes"], "expected self.model.to('cpu') to be called in the 3rd-tier recovery"
    assert predictions is not None


def test_predict_non_cuda_runtime_error_still_raises(monkeypatch):
    """Narrow filter sanity: a non-CUDA RuntimeError must propagate so
    we don't mask genuine training bugs as transient CUDA faults."""
    import lightning as L

    np.random.seed(0)
    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    est = _make_regressor()
    est.fit(X, y, eval_set=(X, y))

    def _raise_non_cuda(self, *args, **kwargs):
        """Raise non cuda."""
        raise RuntimeError("Input shape mismatch: expected 4 features, got 7")

    monkeypatch.setattr(L.Trainer, "predict", _raise_non_cuda)

    with pytest.raises(RuntimeError, match="shape mismatch"):
        est.predict(X)
