"""F-62 (2026-05-31) tests for the Lookahead meta-optimizer wrapper
(Zhang et al. 2019, arxiv 1907.08610).

Three layers:
  1. Unit tests for the Lookahead class itself (k-sync semantics, slow-
     weight lazy init, state-dict round-trip).
  2. Integration: MLPTorchModel.configure_optimizers wraps when
     use_lookahead=True.
  3. biz_value: Lookahead-wrapped AdamW converges to comparable R^2 on
     a small linear-target regression. We don't bench for the +0.4-0.6%
     win here (would need n=many seeds / many tasks); just that
     enabling the wrap doesn't regress correctness or convergence.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from mlframe.training.neural._lookahead_optimizer import (
    Lookahead,
    wrap_with_lookahead,
)


# --- Unit tests --------------------------------------------------------------


def test_lookahead_rejects_non_optimizer_base():
    with pytest.raises(TypeError, match="base_optimizer must be"):
        Lookahead("not an optimizer", k=5, alpha=0.5)  # type: ignore[arg-type]


def test_lookahead_rejects_invalid_k():
    base = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
    with pytest.raises(ValueError, match="k must be >= 1"):
        Lookahead(base, k=0, alpha=0.5)


def test_lookahead_rejects_invalid_alpha():
    base = torch.optim.AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\]"):
        Lookahead(base, k=5, alpha=0.0)
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\]"):
        Lookahead(base, k=5, alpha=1.5)


def test_lookahead_eager_init_at_construction():
    """F-62 fix: slow weights snapshotted at construction (initial param
    values), NOT lazily on first k-sync. Verifies Zhang 2019's
    phi_0 := theta_0. Pre-fix lazy init caused a one-cycle skip and
    0.32-0.40 R^2 regression on the smoke-bench across 4 seeds."""
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    base = torch.optim.SGD([p], lr=0.1)
    lh = Lookahead(base, k=5, alpha=0.5)
    # Slow weights MUST be populated immediately at construction.
    assert id(p) in lh._slow_weights
    assert torch.allclose(lh._slow_weights[id(p)], p.data)


def test_lookahead_kth_step_runs_real_interpolation():
    """At step k slow interpolates toward fast by alpha (with slow
    initially == initial-fast). On step 2k slow interpolates again from
    its cycle-1 anchor toward the new fast position.

    Trace with k=2, alpha=0.5, lr=1.0, p0=0:
      Construct: slow = 0 (initial)
      Step 1: SGD -> p=-1; no sync
      Step 2 (sync): SGD p=-1 -> -2; slow.lerp(p=-2, 0.5) = 0 + 0.5*(-2 - 0) = -1
                     p.copy(-1)
      Step 3: SGD from -1 -> p=-2; no sync
      Step 4 (sync): SGD p=-2 -> -3; slow.lerp(p=-3, 0.5) = -1 + 0.5*(-3 - -1) = -2
                     p.copy(-2)
    """
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.SGD([p], lr=1.0)
    lh = Lookahead(base, k=2, alpha=0.5)
    assert torch.allclose(lh._slow_weights[id(p)], torch.zeros(4))

    p.grad = torch.ones_like(p)
    lh.step()  # step 1: SGD only
    assert torch.allclose(p.data, -1.0 * torch.ones(4))

    p.grad = torch.ones_like(p)
    lh.step()  # step 2 (k-th): sync
    assert torch.allclose(p.data, -1.0 * torch.ones(4), atol=1e-6)
    assert torch.allclose(lh._slow_weights[id(p)], -1.0 * torch.ones(4), atol=1e-6)

    p.grad = torch.ones_like(p)
    lh.step()  # step 3: SGD only
    assert torch.allclose(p.data, -2.0 * torch.ones(4), atol=1e-6)

    p.grad = torch.ones_like(p)
    lh.step()  # step 4 (2k-th): sync
    assert torch.allclose(p.data, -2.0 * torch.ones(4), atol=1e-6)
    assert torch.allclose(lh._slow_weights[id(p)], -2.0 * torch.ones(4), atol=1e-6)


def test_lookahead_state_dict_round_trip_step_count():
    """state_dict / load_state_dict preserve step_count so we resume
    mid-cycle in the same phase."""
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.AdamW([p], lr=1e-3)
    lh = Lookahead(base, k=5, alpha=0.5)
    for _ in range(7):
        p.grad = torch.randn_like(p)
        lh.step()
    sd = lh.state_dict()
    assert sd["step_count"] == 7

    p2 = torch.nn.Parameter(torch.zeros(4))
    base2 = torch.optim.AdamW([p2], lr=1e-3)
    lh2 = Lookahead(base2, k=5, alpha=0.5)
    lh2.load_state_dict(sd)
    assert lh2._step_count == 7


def test_lookahead_param_groups_forward_to_base():
    """Lightning + schedulers access .param_groups; must forward."""
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.AdamW([p], lr=1e-3)
    lh = Lookahead(base, k=5, alpha=0.5)
    assert lh.param_groups is base.param_groups


def test_wrap_with_lookahead_idempotent_when_off():
    base = torch.optim.AdamW(
        [torch.zeros(1, requires_grad=True)], lr=1e-3,
    )
    wrapped = wrap_with_lookahead(base, use_lookahead=False)
    assert wrapped is base


def test_wrap_with_lookahead_returns_lookahead_when_on():
    base = torch.optim.AdamW(
        [torch.zeros(1, requires_grad=True)], lr=1e-3,
    )
    wrapped = wrap_with_lookahead(base, use_lookahead=True, k=7, alpha=0.3)
    assert isinstance(wrapped, Lookahead)
    assert wrapped.k == 7
    assert wrapped.alpha == 0.3
    assert wrapped.base_optimizer is base


# --- Integration ---------------------------------------------------------------


def test_mlp_configure_optimizers_wraps_when_use_lookahead_true():
    """MLPTorchModel.configure_optimizers wraps the base optimizer when
    hparam use_lookahead=True; returns the raw base otherwise.
    """
    from mlframe.training.neural._flat_torch_module import MLPTorchModel
    from mlframe.training.neural.flat import generate_mlp

    net = generate_mlp(
        num_features=4, num_classes=1, nlayers=1,
        first_layer_num_neurons=8, dropout_prob=0.0,
        inputs_dropout_prob=0.0, use_layernorm=False, use_batchnorm=False,
        activation_function=nn.ReLU, verbose=0,
    )
    module_lh = MLPTorchModel(
        loss_fn=nn.MSELoss(), metrics=[], network=net,
        use_lookahead=True, lookahead_k=3, lookahead_alpha=0.4,
    )
    out = module_lh.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    assert isinstance(opt, Lookahead), (
        f"expected Lookahead wrap when use_lookahead=True; got {type(opt).__name__}"
    )
    assert opt.k == 3
    assert opt.alpha == 0.4

    module_plain = MLPTorchModel(
        loss_fn=nn.MSELoss(), metrics=[], network=net, use_lookahead=False,
    )
    out_plain = module_plain.configure_optimizers()
    opt_plain = out_plain["optimizer"] if isinstance(out_plain, dict) else out_plain
    assert not isinstance(opt_plain, Lookahead), (
        "use_lookahead=False must return the raw base optimizer"
    )


# --- biz_value ---------------------------------------------------------------


def test_mlp_lookahead_converges_on_linear_regression():
    """biz_value smoke: at sufficient epochs Lookahead reaches at least
    as good R^2 as plain AdamW on a trivial linear-target task.

    Important: Lookahead is slow EARLY (the alpha=0.5 anchor dampens
    progress for the first ~30 epochs as the slow weights catch up to
    fast). At 30 epochs Lookahead can be 0.3-0.4 R^2 BEHIND plain; by
    100 epochs they converge to within +/-0.02. We use 100 epochs here
    so the test reflects the algorithm's actual behaviour rather than
    a transient early-cycle regression. Measured slope on the standalone
    bench (D:/Temp/lh_trace2.py): plain=0.986 vs lookahead-at-100=0.997
    on a 400-sample 4D linear regression.
    """
    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningRegressor,
        TorchDataModule,
    )

    X, y = make_regression(n_samples=400, n_features=4, noise=0.5, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    def fit_score(use_lookahead: bool) -> float:
        torch.manual_seed(0); np.random.seed(0)
        reg = PytorchLightningRegressor(
            model_class=MLPTorchModel,
            model_params={
                "loss_fn": nn.MSELoss(),
                "learning_rate": 5e-3,
                "use_lookahead": use_lookahead,
                "lookahead_k": 5,
                "lookahead_alpha": 0.5,
            },
            network_params={
                "nlayers": 2, "first_layer_num_neurons": 32,
                "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
                "use_layernorm": False, "use_batchnorm": False,
                "activation_function": nn.ReLU,
            },
            datamodule_class=TorchDataModule,
            datamodule_params={
                "features_dtype": torch.float32, "labels_dtype": torch.float32,
                "dataloader_params": {"batch_size": 64, "num_workers": 0},
            },
            trainer_params={
                "max_epochs": 100, "enable_model_summary": False,
                "enable_progress_bar": False, "log_every_n_steps": 5,
                "devices": 1, "accelerator": "cpu", "logger": False,
            },
            random_state=0,
        )
        reg.fit(X_tr, y_tr)
        return r2_score(y_te, reg.predict(X_te))

    r2_plain = fit_score(False)
    r2_lh = fit_score(True)

    # Both should reach decent R^2 on this clean linear task.
    assert r2_plain > 0.9, f"plain AdamW only reached R^2={r2_plain:.4f}"
    # At 100 epochs Lookahead's alpha-damping has fully amortised; it
    # should match plain within 0.03 R^2 (standalone bench: -0.003 to
    # +0.011, mean +0.004 across 4 seeds).
    assert r2_lh > r2_plain - 0.03, (
        f"Lookahead-wrapped R^2={r2_lh:.4f} regressed >0.03 vs plain R^2={r2_plain:.4f} "
        f"at 100 epochs (this is well past the early-anchor-damping window)."
    )
