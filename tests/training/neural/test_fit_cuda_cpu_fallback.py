"""Regression: ``trainer.fit(...)`` CUDA-runtime errors (``illegal memory access`` / OOM /
``device-side assert``) must trigger a one-shot CPU fallback inside ``_FitMixin._fit_common``
instead of propagating and killing the whole suite.

Mirrors ``test_predict_cuda_cpu_fallback.py``: predict already had this exact retry mechanism
(``run_with_cuda_cpu_fallback`` in ``_cuda_fallback.py``), but fit called ``trainer.fit(...)``
unguarded -- a CUDA context corrupted by an earlier test (driver instability, GPU contention under
xdist) then raised straight out of Lightning's own ``strategy.setup()``/``teardown()`` (e.g.
``torch.cuda.empty_cache()`` inside ``_clear_cuda_memory()``), observed live as 38 failures
concentrated in one xdist worker.

These tests monkey-patch ``L.Trainer.fit`` so the fault is deterministic on any host regardless
of GPU availability.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")


def _make_regressor():
    """Mirror ``test_predict_cuda_cpu_fallback._make_regressor``."""
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
        trainer_params={"max_epochs": 1, "logger": False, "accelerator": "cuda", "devices": 1},
    )


def test_fit_cpu_fallback_on_cuda_runtime_error(caplog, monkeypatch):
    """First Trainer.fit call raises a CUDA RuntimeError; estimator must retry on CPU and the
    second call must succeed."""
    import lightning as L

    invocations = {"count": 0}

    def _fake_fit(self, *args, **kwargs):
        """Fake fit."""
        invocations["count"] += 1
        if invocations["count"] == 1:
            raise RuntimeError(
                "CUDA error: an illegal memory access was encountered\n"
                "CUDA kernel errors might be asynchronously reported at some "
                "other API call, so the stacktrace below might be incorrect."
            )
        return None

    monkeypatch.setattr(L.Trainer, "fit", _fake_fit)

    np.random.seed(0)
    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    est = _make_regressor()

    caplog.set_level(logging.WARNING, logger="mlframe.training.neural.base")
    est.fit(X, y)

    assert invocations["count"] == 2, f"expected 2 fit invocations (cuda fail + cpu retry), got {invocations['count']}"
    fallback_logs = [r for r in caplog.records if "retrying on CPU" in r.message]
    assert fallback_logs, f"expected a WARNING about CPU retry after CUDA error; got: {[r.message for r in caplog.records]}"


def test_fit_3rd_tier_cuda_fail_moves_model_to_cpu_before_retry(caplog, monkeypatch):
    """When the FIRST CPU fallback ALSO raises a CUDA error (context already invalidated), the
    recovery path hides CUDA at the torch module level AND moves the model to CPU before the
    second retry, mirroring the predict-side 3rd-tier recovery."""
    import lightning as L

    invocations = {"count": 0}

    def _fake_fit(self, *args, **kwargs):
        """Fake fit."""
        invocations["count"] += 1
        if invocations["count"] in (1, 2):
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return None

    monkeypatch.setattr(L.Trainer, "fit", _fake_fit)

    np.random.seed(0)
    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    est = _make_regressor()

    moved_to_cpu = {"yes": False}

    import torch

    _orig_to = torch.nn.Module.to

    def _spy_to(self, *a, **kw):
        """Spy to."""
        if a and a[0] == "cpu":
            moved_to_cpu["yes"] = True
        return _orig_to(self, *a, **kw)

    monkeypatch.setattr(torch.nn.Module, "to", _spy_to)

    caplog.set_level(logging.ERROR, logger="mlframe.training.neural.base")
    est.fit(X, y)

    assert invocations["count"] == 3, f"expected 3 fit invocations (cuda fail + CPU fail + CPU retry), got {invocations['count']}"
    assert moved_to_cpu["yes"], "expected model.to('cpu') to be called in the 3rd-tier recovery"


def test_fit_non_cuda_runtime_error_still_raises(monkeypatch):
    """Narrow filter sanity: a non-CUDA RuntimeError must propagate so we don't mask genuine
    training bugs as transient CUDA faults."""
    import lightning as L

    def _raise_non_cuda(self, *args, **kwargs):
        """Raise non cuda."""
        raise RuntimeError("Input shape mismatch: expected 4 features, got 7")

    monkeypatch.setattr(L.Trainer, "fit", _raise_non_cuda)

    np.random.seed(0)
    X = np.random.randn(32, 4).astype(np.float32)
    y = np.random.randn(32).astype(np.float32)
    est = _make_regressor()

    with pytest.raises(RuntimeError, match="shape mismatch"):
        est.fit(X, y)
