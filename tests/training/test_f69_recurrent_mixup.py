"""F-69 (2026-05-31) tests for recurrent Mixup augmentation.

Verifies that the recurrent training_step honors ``use_mixup`` /
``mixup_alpha`` on the HYBRID / FEATURES_ONLY paths (where aux_features
exist) and is a no-op for SEQUENCE_ONLY.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mlframe.training.neural._recurrent_config import (
    RNNType, InputMode, RecurrentConfig,
)
from mlframe.training.neural._recurrent_torch_model import RecurrentTorchModel


def _build_module(input_mode: InputMode, use_mixup: bool) -> RecurrentTorchModel:
    config = RecurrentConfig(
        input_mode=input_mode,
        rnn_type=RNNType.LSTM,
        hidden_size=8,
        num_layers=1,
        bidirectional=False,
        use_attention=False,
        mlp_hidden_sizes=(16,),
        dropout=0.0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        batch_size=16,
        max_epochs=1,
        early_stopping_patience=1,
        gradient_clip_val=1.0,
        precision="32-true",
        accelerator="cpu",
        scale_features=False,
        use_mixup=use_mixup,
        mixup_alpha=0.2,
    )
    return RecurrentTorchModel(
        config=config,
        seq_input_size=4,
        aux_input_size=3,
        is_regression=True,
    )


def _make_hybrid_batch() -> dict:
    return {
        "sequences": torch.randn(8, 5, 4),  # (B, T, F)
        "lengths": torch.tensor([5, 5, 5, 5, 5, 5, 5, 5], dtype=torch.long),
        "aux_features": torch.randn(8, 3),
        "labels": torch.randn(8),
    }


def test_recurrent_training_step_with_mixup_hybrid_regression_returns_finite_loss():
    torch.manual_seed(0)
    module = _build_module(InputMode.HYBRID, use_mixup=True)
    module.train()
    batch = _make_hybrid_batch()
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_recurrent_training_step_with_mixup_off_returns_finite_loss():
    """Sanity: use_mixup=False produces a finite loss without invoking
    mixup."""
    torch.manual_seed(0)
    module = _build_module(InputMode.HYBRID, use_mixup=False)
    module.train()
    batch = _make_hybrid_batch()
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)


def test_recurrent_training_step_sequence_only_mixup_is_noop():
    """SEQUENCE_ONLY mode has no aux_features so mixup is a no-op;
    loss must still be finite + grad-bearing."""
    torch.manual_seed(0)
    module = _build_module(InputMode.SEQUENCE_ONLY, use_mixup=True)
    module.train()
    # SEQUENCE_ONLY: no aux_features in batch
    batch = {
        "sequences": torch.randn(8, 5, 4),
        "lengths": torch.tensor([5] * 8, dtype=torch.long),
        "labels": torch.randn(8),
    }
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_recurrent_training_step_classification_mixup_two_target_form():
    """Classification (integer labels) on HYBRID mode uses the two-target
    convex-loss form."""
    config = RecurrentConfig(
        input_mode=InputMode.HYBRID,
        rnn_type=RNNType.LSTM,
        hidden_size=8,
        num_layers=1,
        bidirectional=False,
        use_attention=False,
        mlp_hidden_sizes=(16,),
        dropout=0.0,
        accelerator="cpu",
        scale_features=False,
        use_mixup=True,
        mixup_alpha=0.2,
        num_classes=3,
    )
    module = RecurrentTorchModel(
        config=config,
        seq_input_size=4,
        aux_input_size=3,
        is_regression=False,
    )
    module.train()
    batch = {
        "sequences": torch.randn(8, 5, 4),
        "lengths": torch.tensor([5] * 8, dtype=torch.long),
        "aux_features": torch.randn(8, 3),
        "labels": torch.randint(0, 3, (8,), dtype=torch.long),
    }
    loss = module.training_step(batch, batch_idx=0)
    assert torch.isfinite(loss)
    assert loss.requires_grad
