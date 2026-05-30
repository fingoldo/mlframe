"""F-07 regression: L1 regularisation must NOT be applied to normalisation
(BatchNorm / LayerNorm / GroupNorm) parameters.

Pre-fix ``MLPTorchModel.training_step`` summed ``p.abs().sum()`` over
``self.network.parameters()`` -- which includes the gamma / beta of every
BN / LN / GN layer. The L1 penalty then drove those gammas to zero,
effectively killing the normalisation layer (gamma=0 means the layer
output is just the bias). Standard practice (the same convention used
by PyTorch's decoupled weight-decay in AdamW): L1 / weight-decay on
Linear weights, NOT on normalisation gamma/beta.

The test compares the gradient on the BN.weight parameter under two
L1 settings: l1_alpha=0 (no L1) and l1_alpha=1.0 (strong L1). Pre-fix
the gradient at l1_alpha=1.0 differs from l1_alpha=0 by exactly
``sign(BN.weight)`` (the L1 sub-gradient). Post-fix the two gradients
are bit-identical: L1 contributes nothing to the BN parameter's
gradient.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import MLPTorchModel, MetricSpec, generate_mlp


def _make_model(l1_alpha: float, n_features: int = 4, n_classes: int = 1) -> MLPTorchModel:
    """Build a tiny MLP with batch normalisation enabled, wrapped in
    MLPTorchModel. Single fixed seed so the network has identical weights
    for both l1_alpha=0 and l1_alpha=1.0 runs."""
    torch.manual_seed(0)
    network = generate_mlp(
        num_features=n_features,
        num_classes=n_classes,
        nlayers=2,
        first_layer_num_neurons=8,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        use_layernorm=False,
        use_batchnorm=True,
        use_layernorm_per_layer=False,
        activation_function=nn.ReLU,
        verbose=0,
    )
    return MLPTorchModel(
        loss_fn=nn.MSELoss(),
        metrics=[],
        network=network,
        learning_rate=1e-3,
        l1_alpha=l1_alpha,
    )


def _grads_after_one_step(l1_alpha: float, batch) -> dict:
    """Run one forward + backward pass at the given l1_alpha and return
    a dict of {param_qualified_name: grad_clone} for all Linear and BN
    parameters."""
    model = _make_model(l1_alpha=l1_alpha)
    model.train()  # BN needs batch_size>=2 + train mode
    # Zero grads.
    for p in model.network.parameters():
        if p.grad is not None:
            p.grad = None
    # training_step expects (features, labels) tuple.
    out = model.training_step(batch, batch_idx=0)
    loss = out["loss"]
    loss.backward()
    return {
        name: (p.grad.detach().clone() if p.grad is not None else None)
        for name, p in model.network.named_parameters()
    }


def _find_bn_weight_name(grads: dict) -> str:
    """Locate the first BatchNorm weight parameter name from a grad dict."""
    # generate_mlp's BN appears with a numeric-indexed name inside the
    # Sequential, e.g. "2.weight". The shape disambiguates: BN weight is
    # 1-D over channels; Linear weight is 2-D.
    for name, g in grads.items():
        if g is None:
            continue
        if name.endswith(".weight") and g.dim() == 1:
            return name
    raise RuntimeError(f"no 1-D weight gradient found; got names {list(grads.keys())}")


@pytest.fixture
def batch():
    """Deterministic batch of 16 samples with 4 features, scalar regression
    target. Batch >= 2 so BN train-mode works."""
    torch.manual_seed(42)
    X = torch.randn(16, 4)
    y = torch.randn(16)
    return (X, y)


def test_l1_does_not_affect_batchnorm_weight_gradient(batch):
    """The BN.weight gradient at l1_alpha=1.0 must EQUAL the BN.weight
    gradient at l1_alpha=0 (within float-precision noise). If L1 leaks
    into the BN parameter, the difference would be exactly the L1 sub-
    gradient ``sign(BN.weight)`` times l1_alpha=1.0.
    """
    grads_no_l1 = _grads_after_one_step(l1_alpha=0.0, batch=batch)
    grads_l1 = _grads_after_one_step(l1_alpha=1.0, batch=batch)

    bn_name = _find_bn_weight_name(grads_no_l1)
    g_no_l1 = grads_no_l1[bn_name]
    g_l1 = grads_l1[bn_name]

    # Diff should be ~0 if L1 correctly EXCLUDES BN. Pre-fix the diff
    # equals sign(BN.weight) * l1_alpha = sign(gamma_init=1.0) * 1.0 = +1
    # for every channel (BN gamma is initialised to all ones).
    diff = (g_l1 - g_no_l1).abs().max().item()
    print(f"\nBN gradient diff (l1=1.0) - (l1=0): max|diff| = {diff:.6f}")
    assert diff < 1e-5, (
        f"L1 leaked into BN.weight gradient: max|diff|={diff:.6f} "
        "(expected <1e-5 if BN is correctly excluded). Pre-fix the diff "
        "is ~1.0 because BN gamma init is 1.0 and the L1 sub-gradient is "
        "sign(gamma)*l1_alpha = 1.0 per channel."
    )


def test_l1_still_affects_linear_weight_gradient(batch):
    """Sanity: L1 must STILL apply to Linear weights. The Linear gradient
    at l1_alpha=1.0 should differ from l1_alpha=0 by approximately
    ``sign(Linear.weight) * l1_alpha`` -- this confirms the fix doesn't
    accidentally turn L1 off entirely."""
    grads_no_l1 = _grads_after_one_step(l1_alpha=0.0, batch=batch)
    grads_l1 = _grads_after_one_step(l1_alpha=1.0, batch=batch)

    # Find the first 2-D (Linear) weight gradient.
    linear_name = None
    for name, g in grads_no_l1.items():
        if g is None:
            continue
        if name.endswith(".weight") and g.dim() == 2:
            linear_name = name
            break
    assert linear_name is not None

    g_no_l1 = grads_no_l1[linear_name]
    g_l1 = grads_l1[linear_name]

    # Recover the Linear weight values to compute the expected sign(W) contribution.
    model = _make_model(l1_alpha=0.0)
    weight = dict(model.network.named_parameters())[linear_name].detach()
    expected_l1_contribution = torch.sign(weight) * 1.0  # l1_alpha=1.0
    diff = (g_l1 - g_no_l1) - expected_l1_contribution
    print(f"Linear gradient diff matches sign(W) * l1_alpha within "
          f"max|residual|={diff.abs().max().item():.6f}")
    assert diff.abs().max().item() < 1e-5, (
        f"Linear L1 contribution does not match sign(W) * l1_alpha; "
        f"max|residual|={diff.abs().max().item():.6f}. Either the L1 "
        "term is not being added to the Linear weight (fix overshot) or "
        "the gradient has unrelated noise."
    )
