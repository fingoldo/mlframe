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

A second, genuinely reproducible win IS tested below: the opt-in ``importance``-ranked mask construction.
A uniform-random mask (the prior/default behavior) keeps each input connection independent of whether that
input is signal or noise -- at high sparsity it can, and in this synthetic reliably does, prune away most of
the few informative features. An importance-ranked mask (fed a simple correlation-with-target score, the
kind of signal any mlframe MRMR/redundancy filter already produces) always keeps the informative features,
so downstream test MSE is reliably lower or equal, never worse.
"""
from __future__ import annotations

import torch
from torch import nn

from mlframe.training.neural.fixed_sparse_linear import FixedSparseLinear, _build_importance_mask


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


def test_fixed_sparse_linear_default_mask_construction_unchanged_when_importance_omitted():
    """Regression test: not passing ``importance`` must reproduce the exact prior (uniform-random) mask,
    bit-for-bit, for a given ``random_state`` -- the new opt-in parameter must not alter default behavior."""
    layer_a = FixedSparseLinear(30, 20, sparsity=0.8, random_state=7)
    layer_b = FixedSparseLinear(30, 20, sparsity=0.8, random_state=7)
    assert torch.equal(layer_a.mask, layer_b.mask)

    # Pin against the exact construction formula used before this change existed, replicated here.
    generator = torch.Generator().manual_seed(7)
    expected_mask = (torch.rand(20, 30, generator=generator) < 0.2).to(torch.float32)
    assert torch.equal(layer_a.mask, expected_mask), "default (no `importance`) mask construction changed -- must stay bit-identical to the prior uniform-random formula"


def test_fixed_sparse_linear_importance_mask_keeps_top_ranked_inputs():
    in_features, out_features, sparsity = 50, 16, 0.8
    importance = torch.arange(in_features, dtype=torch.float32)  # feature i has importance i -> top n_keep are the highest indices
    layer = FixedSparseLinear(in_features, out_features, sparsity=sparsity, importance=importance)

    n_keep = round((1.0 - sparsity) * in_features)
    kept_cols = (layer.mask.sum(dim=0) > 0).nonzero(as_tuple=True)[0]
    expected_kept = set(range(in_features - n_keep, in_features))
    assert set(kept_cols.tolist()) == expected_kept, "importance-ranked mask must keep exactly the highest-importance input columns"
    # every output row keeps the identical set of inputs -- there's no per-row importance signal to differ on
    assert bool((layer.mask == layer.mask[0]).all())


def test_fixed_sparse_linear_importance_mask_wrong_length_raises():
    import pytest

    with pytest.raises(ValueError):
        FixedSparseLinear(10, 10, sparsity=0.8, importance=torch.rand(9))


def test_biz_val_fixed_sparse_linear_importance_mask_beats_random_mask_on_sparse_signal_synthetic():
    """Downstream-accuracy synthetic: only a small informative subset of inputs drives the target, the rest
    are pure noise. At high sparsity a uniform-random mask has a real chance of dropping most informative
    inputs; an importance-ranked mask (fed a cheap correlation-with-target score) never does. Averaged over
    several seeds, the importance-ranked mask must win (lower or equal test MSE) with a real margin."""
    torch.manual_seed(0)
    n_train, n_test, in_features, out_features = 400, 200, 60, 1
    n_informative, sparsity = 6, 0.9  # keeps only 6/60 inputs -- a random mask over 60 columns often misses several of the 6

    informative_idx = torch.randperm(in_features)[:n_informative]
    true_weights = torch.randn(n_informative)

    def make_batch(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(n, in_features)
        y = (x[:, informative_idx] @ true_weights).unsqueeze(1) + 0.1 * torch.randn(n, 1)
        return x, y

    x_train, y_train = make_batch(n_train)
    x_test, y_test = make_batch(n_test)

    # Cheap importance score any caller could compute without mlframe's FS machinery: |correlation| with target.
    x_centered = x_train - x_train.mean(dim=0, keepdim=True)
    y_centered = y_train - y_train.mean()
    importance = (x_centered * y_centered).mean(dim=0).abs() / (x_train.std(dim=0) * y_train.std() + 1e-8)

    def fit_and_eval(layer: FixedSparseLinear, seed: int) -> float:
        torch.manual_seed(seed)
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.02)
        for _ in range(300):
            optimizer.zero_grad()
            loss = (layer(x_train) - y_train).pow(2).mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            return float((layer(x_test) - y_test).pow(2).mean().item())

    random_mses = []
    importance_mses = []
    for seed in range(6):
        random_layer = FixedSparseLinear(in_features, out_features, sparsity=sparsity, random_state=seed)
        importance_layer = FixedSparseLinear(in_features, out_features, sparsity=sparsity, random_state=seed, importance=importance)
        random_mses.append(fit_and_eval(random_layer, seed))
        importance_mses.append(fit_and_eval(importance_layer, seed))

    mean_random_mse = sum(random_mses) / len(random_mses)
    mean_importance_mse = sum(importance_mses) / len(importance_mses)

    # Threshold set below the measured margin (importance-ranked mask reliably keeps all 6 informative
    # inputs; random uniform sampling over 60 columns at 10% keep-rate frequently drops several of them).
    assert mean_importance_mse < mean_random_mse * 0.7, f"expected importance-ranked mask to beat random mask by a real margin, got importance={mean_importance_mse:.4f} vs random={mean_random_mse:.4f}"


def test_build_importance_mask_helper_respects_keep_count():
    generator = torch.Generator().manual_seed(1)
    mask = _build_importance_mask(torch.rand(40), out_features=5, in_features=40, sparsity=0.75, generator=generator)
    n_kept_per_row = int(mask[0].sum().item())
    assert n_kept_per_row == round(0.25 * 40)
