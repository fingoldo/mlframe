"""F-72 (2026-05-31) tests for the spectral_norm_output_only opt-in.

Motivation: the codebase has documented R^2=-326 and R^2=-30 catastrophic
OOD blow-ups from MLP-extrapolation on unseen-groups test splits. Full
``spectral_norm=True`` bounds every Linear and costs ~1.2-1.5x train,
which discourages adoption. ``spectral_norm_output_only=True`` wraps
only the final output Linear -- ~1.01x cost -- and still gives an
output-Lipschitz bound sufficient to clip the extrapolation tail.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from mlframe.training.neural.flat import generate_mlp


def _count_spectral_norm_wraps(model: nn.Module) -> int:
    """Count Linear modules that have the spectral_norm hook attached.
    ``nn.utils.spectral_norm`` stores the original weight as ``weight_orig``
    and registers a ``SpectralNorm`` parametrization; both are easy to
    detect."""
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if hasattr(m, "weight_orig"):
                count += 1
    return count


def test_no_spectral_norm_by_default():
    """spectral_norm and spectral_norm_output_only both default to False;
    no Linear should be wrapped."""
    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=2,
        first_layer_num_neurons=16,
        activation_function=nn.ReLU, verbose=0,
    )
    assert _count_spectral_norm_wraps(net) == 0


def test_spectral_norm_output_only_wraps_one_linear():
    """spectral_norm_output_only=True wraps EXACTLY one Linear -- the
    output Linear. Hidden Linears stay unwrapped."""
    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=2,
        first_layer_num_neurons=16,
        activation_function=nn.ReLU,
        spectral_norm_output_only=True,
        verbose=0,
    )
    # Should be exactly 1 wrapped Linear.
    assert _count_spectral_norm_wraps(net) == 1


def test_spectral_norm_output_only_wraps_the_last_linear():
    """Specifically: the wrapped Linear must be the OUTPUT Linear
    (the last nn.Linear in the model). Hidden Linears must be
    unwrapped."""
    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=3,
        first_layer_num_neurons=32,
        activation_function=nn.ReLU,
        spectral_norm_output_only=True,
        verbose=0,
    )
    linear_modules = [m for m in net.modules() if isinstance(m, nn.Linear)]
    # Last Linear is the output projection; previous Linears are hidden.
    assert hasattr(linear_modules[-1], "weight_orig"), (
        "F-72: output Linear must be spectral_norm-wrapped"
    )
    for hidden in linear_modules[:-1]:
        assert not hasattr(hidden, "weight_orig"), (
            "F-72: hidden Linears must NOT be wrapped under output_only"
        )


def test_spectral_norm_full_still_wraps_everything():
    """spectral_norm=True (the existing flag) wraps all Linears,
    including the output. F-72's introduction of _maybe_sn_output
    must NOT regress this -- both flags compose."""
    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=2,
        first_layer_num_neurons=16,
        activation_function=nn.ReLU,
        spectral_norm=True,
        verbose=0,
    )
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert all(hasattr(m, "weight_orig") for m in linears), (
        "F-72: spectral_norm=True must still wrap every Linear (output_only "
        "is a strict subset, NOT a replacement)"
    )


def test_spectral_norm_output_only_classification_also_wraps():
    """F-72: the output_only flag also bounds classification output --
    same code path. The (N, K) classifier head's Lipschitz constant is
    bounded too."""
    net = generate_mlp(
        num_features=4, num_classes=3, nlayers=2,
        first_layer_num_neurons=16,
        activation_function=nn.ReLU,
        spectral_norm_output_only=True,
        verbose=0,
    )
    linears = [m for m in net.modules() if isinstance(m, nn.Linear)]
    assert hasattr(linears[-1], "weight_orig")


def test_spectral_norm_output_only_forward_returns_finite():
    """Sanity: the wrapped output Linear forward is finite -- the
    spectral_norm hook computes a power-iteration update on every
    forward and the result must be a numerically valid forward pass."""
    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=2,
        first_layer_num_neurons=16,
        activation_function=nn.ReLU,
        spectral_norm_output_only=True,
        verbose=0,
    )
    net.eval()
    x = torch.randn(8, 4)
    with torch.no_grad():
        out = net(x)
    assert torch.isfinite(out).all()
    assert out.shape == (8, 1)
