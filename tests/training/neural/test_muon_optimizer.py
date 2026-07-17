"""C8 (F-33): Muon optimizer + MuonAdamWHybrid for tabular MLPs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
)
from mlframe.training.neural._muon_optimizer import (
    Muon,
    MuonAdamWHybrid,
    _zeropower_via_newtonschulz5,
)


def test_newton_schulz_returns_approximate_orthogonal():
    """Newton-Schulz output should be approximately orthogonal:
    U U^T ~ I. Keller Jordan's 5-iteration quintic with hand-tuned
    coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) is APPROXIMATE
    (designed for fp32/bf16 GEMM speed, not exact orthogonality);
    tolerance is generous on purpose. The test mainly catches the
    DEGENERATE case where Newton-Schulz returns NaN / zeros / the
    raw input."""
    torch.manual_seed(0)
    G = torch.randn(16, 16) * 0.5
    U = _zeropower_via_newtonschulz5(G, steps=5)
    UUt = U @ U.T
    eye = torch.eye(16, dtype=UUt.dtype)
    # Loose absolute tolerance -- 5-iter quintic is approximate.
    assert torch.allclose(UUt, eye, atol=0.5), f"U U^T should be near-identity; max off-diag = {(UUt - eye).abs().max().item():.4f}"
    # Sanity: output must not be the raw input (NS did SOMETHING).
    assert not torch.allclose(U, G, atol=1e-3)
    # Sanity: output must not contain NaN.
    assert torch.isfinite(U).all()


def test_muon_step_updates_2d_params():
    """One Muon step on a tiny 2D Linear weight decreases the loss."""
    torch.manual_seed(0)
    linear = nn.Linear(8, 4, bias=False)
    opt = Muon([linear.weight], lr=0.1)
    x = torch.randn(32, 8)
    target = torch.randn(32, 4)
    loss_before = ((linear(x) - target) ** 2).mean()
    loss_before.backward()
    opt.step()
    loss_after = ((linear(x) - target) ** 2).mean()
    assert loss_after.item() < loss_before.item(), f"Muon failed to decrease loss: before={loss_before.item():.4f}, after={loss_after.item():.4f}"


def test_muon_rejects_1d_params():
    """Muon on a 1D param (bias / BN gamma) must raise — use the
    hybrid instead."""
    bias = nn.Parameter(torch.randn(8))
    opt = Muon([bias], lr=0.1)
    bias.grad = torch.randn(8)
    with pytest.raises(RuntimeError, match=r"only handles 2D"):
        opt.step()


def test_muon_adamw_hybrid_routes_by_param_shape():
    """MuonAdamWHybrid splits params: 2D -> Muon, 1D -> AdamW."""
    net = nn.Sequential(
        nn.Linear(8, 16),  # 2D weight + 1D bias
        nn.BatchNorm1d(16),  # 1D weight + 1D bias
        nn.Linear(16, 4),
    )
    opt = MuonAdamWHybrid(net.parameters(), lr=1e-3, muon_lr=0.02)
    # All 2D params should be in Muon's param groups.
    muon_count = sum(p.numel() for g in opt._muon.param_groups for p in g["params"])
    adamw_count = sum(p.numel() for g in opt._adamw.param_groups for p in g["params"])
    # Expected: 8*16=128 + 16*4=64 = 192 params in Muon; 16+16+16+4 = 52 in AdamW.
    assert muon_count == 8 * 16 + 16 * 4
    assert adamw_count == 16 + 16 + 16 + 4


def test_muon_adamw_hybrid_step_updates_both_groups():
    """One hybrid step decreases loss on a tiny MLP with both 2D and
    1D parameters (BN gammas)."""
    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(8, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )
    opt = MuonAdamWHybrid(net.parameters(), lr=1e-2, muon_lr=0.1)
    x = torch.randn(32, 8)
    target = torch.randn(32, 4)
    for _ in range(3):
        opt.zero_grad()
        loss = ((net(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
    final = ((net(x) - target) ** 2).mean()
    assert final.item() < 1.5, f"loss should decrease; got final={final.item():.4f}"


def test_muon_adamw_hybrid_integration_with_lightning_module():
    """End-to-end: pass MuonAdamWHybrid as the optimizer to MLPTorchModel
    via PytorchLightningRegressor. Fit + predict must complete and
    produce finite predictions."""
    X, y = make_regression(n_samples=160, n_features=5, noise=0.1, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={
            "loss_fn": torch.nn.MSELoss(),
            "learning_rate": 1e-3,
            "optimizer": MuonAdamWHybrid,
            # Hybrid expects muon_lr in kwargs.
            "optimizer_kwargs": {"muon_lr": 0.02},
        },
        network_params={
            "nlayers": 1,
            "first_layer_num_neurons": 16,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": True,
            "activation_function": torch.nn.ReLU,
        },
        datamodule_class=TorchDataModule,
        datamodule_params={
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 5,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
        random_state=0,
    )
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (X_te.shape[0],)
    assert np.isfinite(preds).all()
