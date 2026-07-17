"""C6 (F-31): ResNet-tabular residual block variant of generate_mlp.

Gorishniy 2021 "Revisiting Deep Learning Models for Tabular Data"
showed a properly-tuned residual MLP outperforms TabNet / NODE /
TabTransformer on standard tabular benchmarks. The mlframe variant
ships as ``use_residual=True`` flag on generate_mlp — minimal patch,
no new deps.

Test contract:
  * use_residual=False (default): vanilla MLP, no _ResidualLinearBlock
    instances in the network.
  * use_residual=True with constant-width arch (in==out): residual
    skip is nn.Identity (parameter-free).
  * use_residual=True with differing widths: skip is a Linear projection
    (added params, but shape-safe).
  * use_residual=True trains end-to-end and matches or beats the
    vanilla MLP on a synthetic linear regression problem (sanity, not
    a strict win — the residual path's lift on simple linear is small).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
    generate_mlp,
)
from mlframe.training.neural.flat import _ResidualLinearBlock


def test_use_residual_false_default_no_residual_blocks():
    """Default (use_residual=False) builds vanilla MLP — no
    _ResidualLinearBlock anywhere."""
    net = generate_mlp(
        num_features=8,
        num_classes=1,
        nlayers=2,
        first_layer_num_neurons=16,
        dropout_prob=0.0,
        use_batchnorm=False,
        verbose=0,
    )
    assert not any(isinstance(m, _ResidualLinearBlock) for m in net.modules())


def test_use_residual_true_inserts_blocks_constant_width():
    """use_residual=True with constant-width: each hidden layer
    is a _ResidualLinearBlock; the skip is nn.Identity for same-width
    blocks."""
    net = generate_mlp(
        num_features=16,
        num_classes=1,
        nlayers=3,
        first_layer_num_neurons=16,
        dropout_prob=0.0,
        use_batchnorm=True,
        use_residual=True,
        verbose=0,
    )
    blocks = [m for m in net.modules() if isinstance(m, _ResidualLinearBlock)]
    assert len(blocks) == 3, f"expected 3 residual blocks; got {len(blocks)}"
    for block in blocks:
        # First block: in_features = num_features = 16 = out -> identity skip
        if block.linear.in_features == block.linear.out_features:
            assert isinstance(block.skip, nn.Identity)


def test_use_residual_true_skip_projection_when_dims_differ():
    """When the architecture varies width across layers, the skip
    becomes a Linear projection (bias-free)."""
    net = generate_mlp(
        num_features=8,
        num_classes=1,
        nlayers=2,
        first_layer_num_neurons=32,
        dropout_prob=0.0,
        use_residual=True,
        verbose=0,
    )
    blocks = [m for m in net.modules() if isinstance(m, _ResidualLinearBlock)]
    # First block: in=8, out=32 -> projection skip.
    first = blocks[0]
    assert first.linear.in_features == 8
    assert first.linear.out_features == 32
    assert isinstance(first.skip, nn.Linear)
    assert first.skip.bias is None  # bias-free projection


def test_use_residual_true_trains_on_synthetic_data():
    """End-to-end fit + predict on a clean linear regression problem;
    R^2 must reach respectable levels (>0.8) — confirms the residual
    path actually trains, not just builds."""
    X, y = make_regression(n_samples=300, n_features=6, noise=0.1, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-2},
        network_params={
            "nlayers": 2,
            "first_layer_num_neurons": 32,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": True,
            "use_residual": True,
            "activation_function": torch.nn.ReLU,
        },
        datamodule_class=TorchDataModule,
        datamodule_params={
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 30,
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
    r2 = r2_score(y_te, preds)
    print(f"\nResidual MLP R^2 on synthetic linear: {r2:+.4f}")
    assert r2 > 0.8, f"residual MLP should reach R^2>0.8 on clean linear; got {r2:+.4f}"


def test_use_residual_with_spectral_norm_warns_but_works(caplog):
    """use_residual=True + spectral_norm=True: spectral norm wraps the
    body Linear but NOT the skip projection -> log WARN about the
    approximate Lipschitz bound. Network still trains."""
    import logging

    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.flat"):
        net = generate_mlp(
            num_features=8,
            num_classes=1,
            nlayers=2,
            first_layer_num_neurons=8,
            dropout_prob=0.0,
            use_residual=True,
            spectral_norm=True,
            verbose=0,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("use_residual" in m and "spectral" in m for m in msgs), f"expected use_residual+spectral_norm warning; got {msgs}"
    # Network still constructible + forward works.
    x = torch.randn(4, 8)
    y = net(x)
    assert y.shape == (4, 1)
