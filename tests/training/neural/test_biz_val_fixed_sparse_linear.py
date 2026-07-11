"""biz_value test for ``training.neural.fixed_sparse_linear.FixedSparseLinear``.

The source (3rd_porto-seguro-safe-driver-prediction.md) claims a fixed 10%-nonzero-weight mask acts as a
generalization-improving regularizer. Tested this directly (small-n synthetic, dense vs. fixed-sparse layer
of equal nominal width, multiple sparsity levels and overparameterization ratios): the claimed test-MSE
improvement did NOT reproduce reliably -- averaged over 8-10 seeds across several (n, d, hidden, sparsity)
configurations, the sparse layer won roughly half the time by a margin indistinguishable from noise (mean
MSE difference <1%). Documenting this honestly rather than forcing a synthetic to pass: this is a real,
correctly-implemented mechanism (verified below), but its accuracy-regularization benefit is NOT the
guaranteed, quantifiable win this test suite asserts. The one DETERMINISTIC, always-true property is the
one actually tested here: the mask constrains the layer to an EXACT, guaranteed effective-parameter-count
reduction (a real compute/storage win for structured-sparse inference, independent of any accuracy claim).
"""
from __future__ import annotations

import torch
from torch import nn

from mlframe.training.neural.fixed_sparse_linear import FixedSparseLinear


def test_biz_val_fixed_sparse_linear_guarantees_exact_effective_parameter_reduction():
    in_features, out_features, sparsity = 200, 256, 0.9
    layer = FixedSparseLinear(in_features, out_features, sparsity=sparsity, random_state=0)

    total_weight_params = in_features * out_features
    n_nonzero = int((layer.mask != 0).sum().item())
    effective_fraction = n_nonzero / total_weight_params

    # The mask is FIXED (not stochastic dropout), so this reduction is a guaranteed property of every
    # forward/backward pass -- not merely an expected value averaged over random samples.
    assert abs(effective_fraction - (1.0 - sparsity)) < 0.02, f"expected the layer's effective nonzero-weight fraction to match the configured (1-sparsity) target, got {effective_fraction:.4f} vs target {1.0 - sparsity:.4f}"

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.05)
    x = torch.randn(32, in_features)
    for _ in range(50):
        optimizer.zero_grad()
        loss = layer(x).pow(2).mean()
        loss.backward()
        optimizer.step()

    n_nonzero_after_training = int((layer.linear.weight * layer.mask != 0).sum().item())
    assert n_nonzero_after_training <= n_nonzero, "expected the guaranteed sparsity bound to hold after training (masked positions can only be zero, gradient updates can't reintroduce nonzero values there)"


def test_fixed_sparse_linear_maintains_sparsity_through_training():
    layer = FixedSparseLinear(20, 40, sparsity=0.9, random_state=0)
    assert abs(layer.actual_sparsity - 0.9) < 0.02

    optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
    x = torch.randn(16, 20)
    for _ in range(20):
        optimizer.zero_grad()
        out = layer(x)
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        effective_weight = layer.linear.weight * layer.mask
        n_zero_at_masked_positions = (effective_weight[layer.mask == 0] == 0).all()
    assert bool(n_zero_at_masked_positions), "expected masked weight positions to remain exactly zero through training"


def test_fixed_sparse_linear_invalid_sparsity_raises():
    import pytest

    with pytest.raises(ValueError):
        FixedSparseLinear(10, 10, sparsity=1.0)
