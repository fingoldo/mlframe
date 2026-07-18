"""Wave 11a monolith-split sensor for ``mlframe.training.neural.base``.

Carve pattern: logging filters + tensor helpers + callbacks extracted to sibling files; PytorchLightningEstimator class stays in parent. Identity preservation verified so downstream isinstance / Trainer-callback-list checks keep working.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def parent_module():
    """Parent module."""
    from mlframe.training.neural import base

    return base


@pytest.fixture(scope="module")
def siblings():
    """Siblings."""
    from mlframe.training.neural import (
        _base_callbacks,
        _base_logging,
        _base_tensor_helpers,
    )

    return {
        "logging": _base_logging,
        "tensors": _base_tensor_helpers,
        "callbacks": _base_callbacks,
    }


def test_logging_identity(parent_module, siblings):
    """Logging identity."""
    lg = siblings["logging"]
    assert parent_module.MetricSpec is lg.MetricSpec
    assert parent_module._LightningRankZeroNoiseFilter is lg._LightningRankZeroNoiseFilter
    assert parent_module.suppress_lightning_workers_warning is lg.suppress_lightning_workers_warning
    assert parent_module._rmse_metric is lg._rmse_metric


def test_tensor_helpers_identity(parent_module, siblings):
    """Tensor helpers identity."""
    th = siblings["tensors"]
    assert parent_module.custom_collate_fn is th.custom_collate_fn
    assert parent_module.to_tensor_any is th.to_tensor_any
    assert parent_module.to_numpy_safe is th.to_numpy_safe
    assert parent_module._ensure_numpy is th._ensure_numpy


def test_callback_identity(parent_module, siblings):
    """Callback identity."""
    cb = siblings["callbacks"]
    assert parent_module.NetworkGraphLoggingCallback is cb.NetworkGraphLoggingCallback
    assert parent_module.AggregatingValidationCallback is cb.AggregatingValidationCallback
    assert parent_module.ValLossDivergenceCallback is cb.ValLossDivergenceCallback
    assert parent_module.BestEpochModelCheckpoint is cb.BestEpochModelCheckpoint
    assert parent_module.PeriodicLearningRateFinder is cb.PeriodicLearningRateFinder


def test_isinstance_callbacks_preserved(parent_module):
    """Trainer callback-list check pattern: must still recognise our subclasses."""
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint, LearningRateFinder

    cb1 = parent_module.NetworkGraphLoggingCallback()
    assert isinstance(cb1, Callback)
    cb2 = parent_module.BestEpochModelCheckpoint(monitor="val_loss", mode="min")
    assert isinstance(cb2, ModelCheckpoint)
    cb3 = parent_module.PeriodicLearningRateFinder(period=5)
    assert isinstance(cb3, LearningRateFinder)


def test_facade_loc_budget(parent_module):
    """Facade loc budget."""
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    # Budget raised 1000 -> 1200 (MLP-iter-3 burst) -> 1500 (binary_sigmoid_head,
    # class_weight, random_state, NaN/Inf guards, broader CUDA-fallback retry
    # block, seed_everything backward-compat) -> 1700 (2026-06-01: tanh_train_range
    # auto-derive + single-pass numba kernel wiring + cross-call CUDA-graph
    # sync). Next reasonable splits are the ``fit`` / ``predict`` body into
    # per-phase siblings -- tracked under the same FIXME(carve-wave-next) tag
    # as the LOC_BUDGET_EXEMPT entry.
    assert n_lines < 1700, f"facade is {n_lines} LOC, expected < 1700"


def test_tensor_helpers_smoke_round_trip(parent_module):
    """Exercise carved tensor helpers end-to-end (CLAUDE.md AST-audit gate: sensor must call into moved body)."""
    import torch

    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    t = parent_module.to_tensor_any(arr)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (2, 2)

    back = parent_module.to_numpy_safe(t)
    assert isinstance(back, np.ndarray)
    np.testing.assert_array_equal(back, arr)


def test_metric_spec_construct(parent_module):
    """Metric spec construct."""
    spec = parent_module.MetricSpec(name="rmse", fcn=parent_module._rmse_metric)
    assert spec.name == "rmse"
    assert spec.requires_cpu is True
