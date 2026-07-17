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


def test_lookahead_state_dict_round_trip_slow_weights():
    """F-A fix (2026-05-31, audit follow-up): slow weights round-trip
    via state_dict so a resumed run resumes Zhang 2019's algorithm at
    the correct phase. Pre-fix the slow_weights were dropped on save
    and re-initialised lazily from fast on load -- combined with the
    persisted step_count, the first post-load step would hit
    ``step_count % k == 0``, take the snapshot branch (slow == fast),
    and skip the alpha-interpolation entirely. The resumed run then
    runs one full cycle at effective alpha=1 (pure fast), silently
    losing the anchoring effect.
    """
    # Drive the optimiser to a non-trivial slow-weight state.
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
    base = torch.optim.SGD([p], lr=0.1)
    lh = Lookahead(base, k=3, alpha=0.5)
    for _ in range(7):  # 2 full sync cycles + 1 extra step
        p.grad = torch.ones_like(p)
        lh.step()
    expected_slow = lh._slow_weights[id(p)].clone()
    sd = lh.state_dict()
    # Serialised slow_weights are present + match the live tensor.
    assert "slow_weights" in sd
    assert len(sd["slow_weights"]) == 1
    torch.testing.assert_close(sd["slow_weights"][0]["tensor"], expected_slow)

    # Restore into a fresh optimizer and verify slow_weights bind.
    p2 = torch.nn.Parameter(torch.tensor([10.0, 20.0, 30.0]))  # DIFFERENT init
    base2 = torch.optim.SGD([p2], lr=0.1)
    lh2 = Lookahead(base2, k=3, alpha=0.5)
    lh2.load_state_dict(sd)
    # After load, lh2's slow weights are bound to p2 (the current arch's
    # param) and contain the saved values.
    assert id(p2) in lh2._slow_weights
    torch.testing.assert_close(lh2._slow_weights[id(p2)], expected_slow)
    assert lh2._step_count == 7


def test_lookahead_state_dict_load_resumes_step_count_correctly_for_next_sync():
    """F-A integration: after a load_state_dict from a mid-cycle state,
    the very next ``step()`` continues the cycle phase rather than
    re-snapping (which is what the pre-fix code did).

    Concretely: save at step_count=5 (mid-cycle for k=3, so next sync
    is at step_count=6). Load + take one step -> step_count=6 -> sync
    fires AND must run real interpolation against the loaded slow,
    not snap-to-fast.
    """
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.SGD([p], lr=1.0)
    lh = Lookahead(base, k=3, alpha=0.5)
    # Drive for 5 steps; the k-th sync at step 3 ran, leaving slow != fast.
    for _ in range(5):
        p.grad = torch.ones_like(p)
        lh.step()
    sd = lh.state_dict()
    pre_slow = lh._slow_weights[id(p)].clone()
    pre_fast = p.data.clone()
    assert not torch.allclose(pre_slow, pre_fast)

    # Restore into a fresh optimizer with the same initial params.
    p2 = torch.nn.Parameter(torch.zeros(4))
    base2 = torch.optim.SGD([p2], lr=1.0)
    lh2 = Lookahead(base2, k=3, alpha=0.5)
    lh2.load_state_dict(sd)
    # Force p2 to match the saved fast value so the next step sees the
    # same state as the original run would have post-load.
    p2.data.copy_(pre_fast)
    torch.testing.assert_close(lh2._slow_weights[id(p2)], pre_slow)
    assert lh2._step_count == 5

    # Next step (step 6) is a k-th sync. With F-A fix, slow is the LOADED
    # tensor and interpolation runs normally:
    #   pre-step:  fast = pre_fast (loaded), slow = pre_slow (loaded)
    #   SGD:       fast -= 1
    #   sync:      slow.lerp(fast, 0.5); fast.copy(slow)
    p2.grad = torch.ones_like(p2)
    lh2.step()
    # Recompute expected: SGD makes fast = pre_fast - 1; then lerp:
    # slow_new = pre_slow + 0.5 * ((pre_fast - 1) - pre_slow)
    expected_slow_new = pre_slow + 0.5 * ((pre_fast - 1.0) - pre_slow)
    torch.testing.assert_close(lh2._slow_weights[id(p2)], expected_slow_new)
    torch.testing.assert_close(p2.data, expected_slow_new)


def test_lookahead_param_groups_forward_to_base():
    """Lightning + schedulers access .param_groups; must forward."""
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.AdamW([p], lr=1e-3)
    lh = Lookahead(base, k=5, alpha=0.5)
    assert lh.param_groups is base.param_groups


def test_wrap_with_lookahead_idempotent_when_off():
    base = torch.optim.AdamW(
        [torch.zeros(1, requires_grad=True)],
        lr=1e-3,
    )
    wrapped = wrap_with_lookahead(base, use_lookahead=False)
    assert wrapped is base


def test_wrap_with_lookahead_returns_lookahead_when_on():
    base = torch.optim.AdamW(
        [torch.zeros(1, requires_grad=True)],
        lr=1e-3,
    )
    wrapped = wrap_with_lookahead(base, use_lookahead=True, k=7, alpha=0.3)
    assert isinstance(wrapped, Lookahead)
    assert wrapped.k == 7
    assert wrapped.alpha == 0.3
    assert wrapped.base_optimizer is base


# --- F-B (audit follow-up): commit_slow_to_fast ------------------------------


def test_commit_slow_to_fast_makes_param_equal_slow():
    """F-B: after commit_slow_to_fast(), p.data MUST equal the slow
    anchor for every tracked param. Verifies the post-fit
    fast<-slow projection used in on_train_end."""
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.SGD([p], lr=1.0)
    lh = Lookahead(base, k=2, alpha=0.5)
    # Drive a few steps so fast (p) diverges from slow (== initial).
    p.grad = torch.ones_like(p)
    lh.step()  # step 1: p = -1; no sync
    # At this point: slow = 0 (initial), p = -1 (fast)
    assert torch.allclose(lh._slow_weights[id(p)], torch.zeros(4))
    assert torch.allclose(p.data, -1.0 * torch.ones(4))

    lh.commit_slow_to_fast()
    # After commit: p MUST equal slow (which was 0, the initial anchor).
    assert torch.allclose(p.data, torch.zeros(4)), f"F-B: commit_slow_to_fast must overwrite p with slow; got p={p.data}"


def test_commit_slow_to_fast_is_idempotent_when_already_synced():
    """At a k-th step fast == slow already; commit is a no-op."""
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.SGD([p], lr=1.0)
    lh = Lookahead(base, k=2, alpha=0.5)
    # Step twice to land on a k-th sync.
    p.grad = torch.ones_like(p)
    lh.step()
    p.grad = torch.ones_like(p)
    lh.step()  # k-th sync: fast == slow after this
    snapshot = p.data.clone()
    lh.commit_slow_to_fast()
    # Idempotent: no change.
    assert torch.allclose(p.data, snapshot)


def test_mlp_on_train_end_commits_slow_to_fast_when_use_lookahead():
    """F-B integration: after a Lookahead-wrapped MLP fit completes,
    the network's params MUST equal the optimizer's slow weights (not
    the mid-cycle fast values). This guards against the regression
    where ``predict()`` after fit returns wrong-anchor predictions.
    """
    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningRegressor,
        TorchDataModule,
    )

    X, y = make_regression(n_samples=128, n_features=4, noise=0.5, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    torch.manual_seed(0)
    np.random.seed(0)
    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={
            "loss_fn": nn.MSELoss(),
            "learning_rate": 5e-3,
            "use_lookahead": True,
            "lookahead_k": 7,  # >1 so the last step is unlikely to be a sync
            "lookahead_alpha": 0.5,
            # Don't reload best checkpoint -- we want to inspect the
            # post-fit live weights, not a reloaded snapshot.
            "load_best_weights_on_train_end": False,
        },
        network_params={
            "nlayers": 1,
            "first_layer_num_neurons": 8,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": False,
            "activation_function": nn.ReLU,
        },
        datamodule_class=TorchDataModule,
        datamodule_params={
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 5,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 5,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
        random_state=0,
    )
    reg.fit(X_tr, y_tr)

    # After fit, every param tensor must match its corresponding slow
    # entry in the Lookahead optimizer. If they differ the F-B fix
    # didn't fire (or the optimizer wasn't a Lookahead).
    # Note: the Lookahead is a configure_optimizers return that Lightning
    # may have wrapped further; reach through if so.
    opts = reg.model.optimizers()
    opt_iter = opts if isinstance(opts, (list, tuple)) else [opts]
    found_lookahead = False
    for opt in opt_iter:
        base = getattr(opt, "optimizer", opt)
        if not isinstance(base, Lookahead):
            continue
        found_lookahead = True
        for group in base.base_optimizer.param_groups:
            for p in group["params"]:
                slow = base._slow_weights.get(id(p))
                if slow is None:
                    continue
                # Allow tiny numeric noise from device-side ops.
                assert torch.allclose(p.data, slow, atol=1e-5), (
                    f"F-B: post-fit p does not equal slow (max diff "
                    f"{(p.data - slow).abs().max().item():.6f}). The "
                    f"on_train_end commit_slow_to_fast hook did not fire."
                )
    assert found_lookahead, "F-B test: configure_optimizers did not return a Lookahead"


# --- Integration ---------------------------------------------------------------


def test_mlp_configure_optimizers_wraps_when_use_lookahead_true():
    """MLPTorchModel.configure_optimizers wraps the base optimizer when
    hparam use_lookahead=True; returns the raw base otherwise.
    """
    from mlframe.training.neural._flat_torch_module import MLPTorchModel
    from mlframe.training.neural.flat import generate_mlp

    net = generate_mlp(
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
    module_lh = MLPTorchModel(
        loss_fn=nn.MSELoss(),
        metrics=[],
        network=net,
        use_lookahead=True,
        lookahead_k=3,
        lookahead_alpha=0.4,
    )
    out = module_lh.configure_optimizers()
    opt = out["optimizer"] if isinstance(out, dict) else out
    assert isinstance(opt, Lookahead), f"expected Lookahead wrap when use_lookahead=True; got {type(opt).__name__}"
    assert opt.k == 3
    assert opt.alpha == 0.4

    module_plain = MLPTorchModel(
        loss_fn=nn.MSELoss(),
        metrics=[],
        network=net,
        use_lookahead=False,
    )
    out_plain = module_plain.configure_optimizers()
    opt_plain = out_plain["optimizer"] if isinstance(out_plain, dict) else out_plain
    assert not isinstance(opt_plain, Lookahead), "use_lookahead=False must return the raw base optimizer"


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
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    def fit_score(use_lookahead: bool) -> float:
        torch.manual_seed(0)
        np.random.seed(0)
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
                "nlayers": 2,
                "first_layer_num_neurons": 32,
                "dropout_prob": 0.0,
                "inputs_dropout_prob": 0.0,
                "use_layernorm": False,
                "use_batchnorm": False,
                "activation_function": nn.ReLU,
            },
            datamodule_class=TorchDataModule,
            datamodule_params={
                "features_dtype": torch.float32,
                "labels_dtype": torch.float32,
                "dataloader_params": {"batch_size": 64, "num_workers": 0},
            },
            trainer_params={
                "max_epochs": 100,
                "enable_model_summary": False,
                "enable_progress_bar": False,
                "log_every_n_steps": 5,
                "devices": 1,
                "accelerator": "cpu",
                "logger": False,
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
        f"Lookahead-wrapped R^2={r2_lh:.4f} regressed >0.03 vs plain R^2={r2_plain:.4f} at 100 epochs (this is well past the early-anchor-damping window)."
    )
