"""F-10 regression test: ``_compute_weighted_loss`` must emit a
once-per-LightningModule WARN on the first all-zero ``sample_weight``
batch.

Pre-fix the safe-divide path (``raw / clamp(weight_sum, 1e-12)``)
silently returned 0 loss + 0 gradient with no log message. A user who
accidentally fed all-zero weights (sample_weight pipeline bug, masked
boolean → integer overflow, etc.) saw a flat val curve with no clue
why. F-10 adds a one-shot warning so the operator gets a signal.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import MLPTorchModel, generate_mlp


def _make_module() -> MLPTorchModel:
    torch.manual_seed(0)
    network = generate_mlp(
        num_features=4,
        num_classes=1,
        nlayers=1,
        first_layer_num_neurons=8,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        use_layernorm=False,
        use_batchnorm=False,
        activation_function=nn.ReLU,
        verbose=0,
    )
    return MLPTorchModel(loss_fn=nn.MSELoss(), metrics=[], network=network)


def test_zero_weight_batch_emits_warning(caplog):
    """First call with all-zero ``sample_weight`` must produce a WARN
    log record from the mlframe neural-flat logger."""
    module = _make_module()
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=False)
    labels = torch.tensor([1.5, 1.5, 1.5, 1.5])
    zero_w = torch.zeros(4)

    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.flat"):
        loss = module._compute_weighted_loss(preds, labels, zero_w)

    # Loss is still numerically 0 (zero gradient flows naturally).
    assert loss.item() == pytest.approx(0.0, abs=1e-9)

    # Warning fired once and mentions zero weight.
    warning_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("All-zero sample_weight" in m for m in warning_msgs), f"expected an 'All-zero sample_weight' warning; got {warning_msgs}"


def test_warning_fires_only_once(caplog):
    """Second + subsequent zero-weight calls do NOT re-emit the WARN."""
    module = _make_module()
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([1.5, 1.5, 1.5, 1.5])
    zero_w = torch.zeros(4)

    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.flat"):
        module._compute_weighted_loss(preds, labels, zero_w)
        module._compute_weighted_loss(preds, labels, zero_w)
        module._compute_weighted_loss(preds, labels, zero_w)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "All-zero sample_weight" in r.getMessage()]
    assert len(warning_records) == 1, f"expected exactly 1 'All-zero sample_weight' warning across 3 calls; got {len(warning_records)}"


def test_nonzero_weight_does_not_warn(caplog):
    """Sanity: a normal (nonzero) weight vector must NOT trigger the
    warning -- the F-10 fix is opt-in to the zero-weight pathology and
    must not pollute logs for healthy fits."""
    module = _make_module()
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([1.5, 1.5, 1.5, 1.5])
    w = torch.ones(4)

    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.flat"):
        loss = module._compute_weighted_loss(preds, labels, w)

    assert loss.item() > 0  # nonzero loss expected (preds != labels)
    warnings_about_zero = [r for r in caplog.records if r.levelno == logging.WARNING and "All-zero sample_weight" in r.getMessage()]
    assert warnings_about_zero == [], f"unexpected zero-weight warning on healthy weights: {warnings_about_zero}"


def test_warning_resets_for_fresh_module(caplog):
    """A FRESH LightningModule should not inherit the warned-flag from a
    sibling -- each fit gets one warning. (No reset is needed; the flag
    lives on the instance.)"""
    module_a = _make_module()
    module_b = _make_module()
    preds = torch.tensor([1.0, 2.0])
    labels = torch.tensor([1.5, 1.5])
    zero_w = torch.zeros(2)

    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.flat"):
        module_a._compute_weighted_loss(preds, labels, zero_w)
        module_b._compute_weighted_loss(preds, labels, zero_w)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and "All-zero sample_weight" in r.getMessage()]
    # Both modules warn once each.
    assert len(warning_records) == 2
