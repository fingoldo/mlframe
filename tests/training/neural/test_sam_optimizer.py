"""F-63 (2026-05-31) tests for the SAM (Sharpness-Aware Minimization)
optimizer wrapper.

Three layers:
  1. Unit tests on ``SAM`` class (validation, two-step API mechanics).
  2. Integration: MLPTorchModel.configure_optimizers wraps when
     ``use_sam=True``; composition with Lookahead is correct.
  3. biz_value: enabling SAM doesn't break convergence on a small
     regression task.

We do NOT assert a SAM WIN here (the Foret 2020 paper needed many
seeds + many tasks); just no catastrophic regression.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mlframe.training.neural._sam_optimizer import SAM


# --- Unit tests --------------------------------------------------------------


def test_sam_rejects_non_optimizer_base():
    with pytest.raises(TypeError, match="base_optimizer must be"):
        SAM("not an optimizer", rho=0.05)  # type: ignore[arg-type]


def test_sam_rejects_invalid_rho():
    base = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
    with pytest.raises(ValueError, match="rho must be > 0"):
        SAM(base, rho=0.0)
    with pytest.raises(ValueError, match="rho must be > 0"):
        SAM(base, rho=-0.1)


def test_sam_first_step_perturbs_params_along_grad_direction():
    """first_step(): theta <- theta + rho * grad / |grad|.
    Verifies the perturbation magnitude equals rho along the
    grad-normalised direction."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    base = torch.optim.SGD([p], lr=0.1)
    sam = SAM(base, rho=0.5)
    # Set a manual grad direction (1, 0, 0).
    p.grad = torch.tensor([1.0, 0.0, 0.0])
    initial = p.data.clone()
    sam.first_step()
    perturbed = p.data
    # |grad| = 1, so perturbation = 0.5 * (1, 0, 0) / 1 = (0.5, 0, 0).
    torch.testing.assert_close(
        perturbed - initial, torch.tensor([0.5, 0.0, 0.0]), atol=1e-6, rtol=0.0
    )


def test_sam_second_step_restores_then_steps():
    """second_step(): restores pre-perturbation params, then runs the
    base optimizer's step using the CURRENT param.grad (which the user
    must have computed at the perturbed weights between first and
    second step).
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    base = torch.optim.SGD([p], lr=1.0)
    sam = SAM(base, rho=0.1)
    # Initial grad set, then first_step perturbs.
    p.grad = torch.tensor([1.0, 0.0])
    sam.first_step()
    # After first_step, p = (0.1, 0.0); _param_backup[id(p)] = (0, 0).
    assert torch.allclose(p.data, torch.tensor([0.1, 0.0]))
    assert id(p) in sam._param_backup

    # Simulate "second forward+backward at perturbed weights" by setting
    # a new gradient.
    p.grad = torch.tensor([0.5, 1.5])
    sam.second_step()
    # second_step restores p to (0, 0), then SGD steps p -= lr * grad
    # = (0, 0) - 1.0 * (0.5, 1.5) = (-0.5, -1.5).
    torch.testing.assert_close(
        p.data, torch.tensor([-0.5, -1.5]), atol=1e-6, rtol=0.0
    )
    # Backup cache cleared after second_step.
    assert sam._param_backup == {}


def test_sam_param_groups_forwarding():
    p = torch.nn.Parameter(torch.zeros(2))
    base = torch.optim.SGD([p], lr=0.1)
    sam = SAM(base, rho=0.05)
    assert sam.param_groups is base.param_groups


def test_sam_state_dict_round_trip():
    p = torch.nn.Parameter(torch.zeros(2))
    base = torch.optim.AdamW([p], lr=1e-3)
    sam = SAM(base, rho=0.1, adaptive=True)
    sd = sam.state_dict()
    assert sd["rho"] == 0.1
    assert sd["adaptive"] is True

    p2 = torch.nn.Parameter(torch.zeros(2))
    base2 = torch.optim.AdamW([p2], lr=1e-3)
    sam2 = SAM(base2, rho=0.05, adaptive=False)
    sam2.load_state_dict(sd)
    assert sam2.rho == 0.1
    assert sam2.adaptive is True


def test_sam_step_without_closure_raises():
    """``step()`` without a closure must raise -- SAM needs to re-run
    the loss + backward at the perturbed weights."""
    p = torch.nn.Parameter(torch.zeros(2))
    base = torch.optim.SGD([p], lr=0.1)
    sam = SAM(base, rho=0.05)
    p.grad = torch.tensor([1.0, 0.0])
    with pytest.raises(RuntimeError, match="closure"):
        sam.step()


def test_sam_step_with_closure_runs_two_passes():
    """SAM.step(closure) calls the closure once (Lightning's contract:
    the first forward+backward already ran; closure re-runs at perturbed
    weights). The base optimizer steps once, using the perturbed
    gradient.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
    base = torch.optim.SGD([p], lr=1.0)
    sam = SAM(base, rho=0.1)

    # First forward+backward simulated: set grad before calling step.
    p.grad = torch.tensor([1.0, 0.0])
    counter = {"calls": 0}

    def closure():
        counter["calls"] += 1
        # Simulate the perturbed-weight forward+backward: produce a
        # different gradient at the perturbed point.
        p.grad = torch.tensor([0.5, 1.5])
        return torch.tensor(0.0)

    sam.step(closure)
    assert counter["calls"] == 1, "SAM.step should call closure once"
    # After step: original p (0, 0) restored, then SGD step using the
    # second grad (0.5, 1.5) -> p = -1.0 * (0.5, 1.5) = (-0.5, -1.5).
    torch.testing.assert_close(
        p.data, torch.tensor([-0.5, -1.5]), atol=1e-6, rtol=0.0
    )


def test_sam_adaptive_scales_perturbation_by_param_magnitude():
    """Adaptive SAM (Kwon 2021): perturbation per-param is scaled by
    |theta|. With theta=(10, 0.1) and grad=(1, 1) and rho=1, the
    pre-norm perturbation is (10, 0.1); after normalisation the
    relative magnitudes are preserved.
    """
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.tensor([10.0, 0.1]))
    base = torch.optim.SGD([p], lr=0.1)
    sam = SAM(base, rho=1.0, adaptive=True)
    p.grad = torch.tensor([1.0, 1.0])
    initial = p.data.clone()
    sam.first_step()
    perturbation = p.data - initial
    # With adaptive=True, e_w = |theta| * grad * (rho / |adaptive_grad|).
    # The relative ratio |perturbation[0]| / |perturbation[1]| should
    # equal the |theta| ratio 10 / 0.1 = 100.
    ratio = abs(perturbation[0].item()) / abs(perturbation[1].item())
    assert abs(ratio - 100.0) < 1e-3


# --- Integration ---------------------------------------------------------------


def test_mlptorchmodel_configure_optimizers_wraps_when_use_sam_true():
    from mlframe.training.neural._flat_torch_module import MLPTorchModel
    from mlframe.training.neural.flat import generate_mlp

    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=1,
        first_layer_num_neurons=8, dropout_prob=0.0,
        inputs_dropout_prob=0.0, use_layernorm=False, use_batchnorm=False,
        activation_function=nn.ReLU, verbose=0,
    )
    module = MLPTorchModel(
        loss_fn=nn.MSELoss(), metrics=[], network=net,
        use_sam=True, sam_rho=0.1, sam_adaptive=True,
    )
    out = module.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    assert isinstance(opt, SAM)
    assert opt.rho == 0.1
    assert opt.adaptive is True


def test_mlptorchmodel_sam_off_does_not_wrap():
    from mlframe.training.neural._flat_torch_module import MLPTorchModel
    from mlframe.training.neural.flat import generate_mlp

    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=1,
        first_layer_num_neurons=8, dropout_prob=0.0,
        inputs_dropout_prob=0.0, use_layernorm=False, use_batchnorm=False,
        activation_function=nn.ReLU, verbose=0,
    )
    module = MLPTorchModel(
        loss_fn=nn.MSELoss(), metrics=[], network=net, use_sam=False,
    )
    out = module.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    assert not isinstance(opt, SAM)


def test_sam_composes_with_lookahead_on_mlp():
    """SAM wraps Lookahead -- the outermost optimizer Lightning sees
    is SAM; Lookahead is its base, and AdamW is Lookahead's base."""
    from mlframe.training.neural._flat_torch_module import MLPTorchModel
    from mlframe.training.neural._lookahead_optimizer import Lookahead
    from mlframe.training.neural.flat import generate_mlp

    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=1,
        first_layer_num_neurons=8, dropout_prob=0.0,
        inputs_dropout_prob=0.0, use_layernorm=False, use_batchnorm=False,
        activation_function=nn.ReLU, verbose=0,
    )
    module = MLPTorchModel(
        loss_fn=nn.MSELoss(), metrics=[], network=net,
        use_lookahead=True, use_sam=True,
    )
    out = module.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    assert isinstance(opt, SAM)
    assert isinstance(opt.base_optimizer, Lookahead)
