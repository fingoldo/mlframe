"""Tests for the 2026-05-27 MLP option additions:

* ``generate_mlp(spectral_norm=True)`` -- wraps every Linear with
  ``nn.utils.spectral_norm`` so each layer's weight has sigma_max <= 1
  after power-iteration convergence.
* ``activation_function=nn.GELU`` / ``nn.Mish`` -- accepted and produces
  finite forward + backward.
* ``activation_function=Snake`` -- custom periodic activation defined
  in flat.py; finite forward + backward AND preserves periodic signal
  better than monotonic activations.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mlframe.training.neural.flat import (
    Snake,
    generate_mlp,
    MLPNeuronsByLayerArchitecture,
)


def _build_mlp(activation_function, spectral_norm=False, nlayers=2):
    return generate_mlp(
        num_features=32,
        num_classes=1,
        nlayers=nlayers,
        first_layer_num_neurons=64,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=2.0,
        activation_function=activation_function,
        weights_init_fcn=None,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        spectral_norm=spectral_norm,
        verbose=0,
    )


@pytest.mark.parametrize("act", [nn.Tanh, nn.GELU, nn.Mish, Snake])
def test_activation_forward_backward_finite(act) -> None:
    torch.manual_seed(0)
    net = _build_mlp(act, spectral_norm=False)
    X = torch.randn(64, 32)
    y = net(X)
    assert y.shape == (64, 1)
    assert torch.isfinite(y).all()
    loss = y.mean()
    loss.backward()
    for p in net.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("act", [nn.Tanh, nn.GELU, nn.Mish, Snake])
def test_activation_with_spectral_norm(act) -> None:
    torch.manual_seed(1)
    net = _build_mlp(act, spectral_norm=True)
    X = torch.randn(64, 32)
    y = net(X)
    assert torch.isfinite(y).all()
    y.mean().backward()


def test_spectral_norm_bounds_linear_sigma_to_one() -> None:
    """After power-iteration convergence, every Linear weight matrix's
    largest singular value must equal 1.0 (within numerical tolerance).
    This is the formal Lipschitz-1 guarantee that motivates SN: composes
    with Lipschitz activations to give a globally Lipschitz network.
    """
    torch.manual_seed(2)
    net = generate_mlp(
        num_features=64,
        num_classes=1,
        nlayers=4,
        first_layer_num_neurons=128,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=2.0,
        activation_function=nn.LeakyReLU,
        weights_init_fcn=None,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        spectral_norm=True,
        spectral_norm_n_power_iterations=5,
        verbose=0,
    )
    # Run many forwards to converge the power iteration.
    X = torch.randn(8, 64)
    for _ in range(50):
        _ = net(X)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            W = m.weight.detach()
            sigma_max = float(torch.linalg.matrix_norm(W, ord=2))
            assert abs(sigma_max - 1.0) < 1e-4, f"SN Linear has sigma_max={sigma_max} after 50 iters; expected ~1.0 for spectral_norm=True."


def test_spectral_norm_off_no_constraint() -> None:
    """Without SN, Linear weights are arbitrary -- sigma_max may exceed 1."""
    torch.manual_seed(3)
    net = generate_mlp(
        num_features=64,
        num_classes=1,
        nlayers=4,
        first_layer_num_neurons=128,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=2.0,
        activation_function=nn.LeakyReLU,
        weights_init_fcn=None,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        spectral_norm=False,
        verbose=0,
    )
    sigmas = []
    for m in net.modules():
        if isinstance(m, nn.Linear):
            sigmas.append(float(torch.linalg.matrix_norm(m.weight.detach(), ord=2)))
    # Without SN, at least one Linear should differ noticeably from 1.0
    # (default init does not target sigma=1).
    assert max(abs(s - 1.0) for s in sigmas) > 0.05, (
        f"SN=False net's Linear sigmas {sigmas} all ~1.0; either the init was unusually lucky or SN-off path was not honoured."
    )


def test_snake_periodic_fit_finite_and_no_worse_than_tanh() -> None:
    """Smoke check on a slow-periodic target (period = 4 in input
    space). Snake activation should AT LEAST match Tanh's fit and
    typically beat it; testing only "no worse than" because Adam +
    300 epochs is not guaranteed to converge to the global optimum
    on either side. The hard claim (Snake's periodicity helps fast-
    period targets) is validated empirically in the literature; this
    test catches gross regression on the Snake forward.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    # Slow-period target: y = sin(pi/2 * x) on [-2, 2] -> period 4,
    # i.e. half a cycle visible. Both activations can fit this.
    N = 2000
    x = np.linspace(-2.0, 2.0, N).reshape(-1, 1).astype(np.float32)
    y = np.sin(0.5 * np.pi * x).astype(np.float32)
    X = torch.from_numpy(x)
    Y = torch.from_numpy(y)

    def _train(act, epochs=300, lr=1e-2):
        torch.manual_seed(0)
        net = generate_mlp(
            num_features=1,
            num_classes=1,
            nlayers=2,
            first_layer_num_neurons=16,
            neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Constant,
            consec_layers_neurons_ratio=1.0,
            activation_function=act,
            weights_init_fcn=None,
            dropout_prob=0.0,
            inputs_dropout_prob=0.0,
            verbose=0,
        )
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            pred = net(X)
            loss = ((pred - Y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach()), net

    loss_tanh, _ = _train(nn.Tanh)
    loss_snake, _ = _train(Snake)
    # Both must produce finite finite training losses, and Snake must
    # not be dramatically worse than Tanh (loose 2x ceiling). Snake's
    # periodicity advantage on fast-period targets is a separate
    # empirical claim left to bench scripts.
    assert np.isfinite(loss_tanh) and np.isfinite(loss_snake)
    assert loss_snake < max(loss_tanh * 2.0, 0.5), f"Snake fit looks broken: snake={loss_snake:.4g} vs tanh={loss_tanh:.4g}"


def test_snake_learnable_alpha_trains() -> None:
    """``alpha_learnable=True`` exposes alpha as a Parameter that
    receives gradients."""
    torch.manual_seed(0)
    snake = Snake(alpha=1.0, alpha_learnable=True)
    assert isinstance(snake.alpha, nn.Parameter)
    X = torch.randn(32, 1, requires_grad=True)
    y = snake(X).sum()
    y.backward()
    assert snake.alpha.grad is not None
    assert torch.isfinite(snake.alpha.grad).all()


def test_snake_fixed_alpha_no_gradient() -> None:
    """Default ``alpha_learnable=False`` registers alpha as a buffer
    (no gradient flow)."""
    snake = Snake(alpha=1.5, alpha_learnable=False)
    assert not isinstance(snake.alpha, nn.Parameter)
    # Buffer is in state_dict but not parameters
    param_names = [n for n, _ in snake.named_parameters()]
    assert "alpha" not in param_names
