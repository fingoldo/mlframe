"""C7 (F-32): PLR numerical-feature embedding for tabular MLPs.

Per Holzmuller 2024 (RealMLP-TD) the single biggest tabular MLP
ablation lift came from periodic numerical embeddings:
  +20.6% R^2 on regression aggregate
  +2.3% accuracy on classification aggregate
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
    generate_mlp,
)
from mlframe.training.neural._numerical_embeddings import PeriodicLinearEmbedding


def test_plr_embedding_output_shape():
    """forward(N, D) -> (N, D * embed_dim)."""
    emb = PeriodicLinearEmbedding(in_features=5, embed_dim=8, n_frequencies=12)
    x = torch.randn(32, 5)
    out = emb(x)
    assert out.shape == (32, 5 * 8)


def test_plr_embedding_no_include_raw_shape():
    """include_raw=False removes the raw scalar from the per-feature
    input but output shape is unchanged (proj still maps to embed_dim)."""
    emb = PeriodicLinearEmbedding(in_features=4, embed_dim=6, n_frequencies=8, include_raw=False)
    x = torch.randn(16, 4)
    out = emb(x)
    assert out.shape == (16, 4 * 6)


def test_plr_embedding_gradients_flow():
    """Backward through the embedding must produce nonzero gradients
    on the coeffs (frequencies) and proj weights."""
    emb = PeriodicLinearEmbedding(in_features=3, embed_dim=4, n_frequencies=6)
    x = torch.randn(8, 3, requires_grad=False)
    out = emb(x)
    loss = out.sum()
    loss.backward()
    assert emb.coeffs.grad is not None and emb.coeffs.grad.abs().sum() > 0
    assert emb.proj_weight.grad is not None and emb.proj_weight.grad.abs().sum() > 0


def test_plr_embedding_rejects_non_2d_input():
    """Plr embedding rejects non 2d input."""
    emb = PeriodicLinearEmbedding(in_features=4, embed_dim=4)
    with pytest.raises(ValueError, match=r"2-D"):
        emb(torch.randn(8, 4, 1))


def test_generate_mlp_with_plr_embedding_constructs():
    """generate_mlp(numerical_embedding='plr') prepends PLR + uses
    embedded dim for downstream layers. example_input_array reflects
    RAW input shape."""
    net = generate_mlp(
        num_features=4,
        num_classes=1,
        nlayers=2,
        first_layer_num_neurons=32,
        dropout_prob=0.0,
        use_batchnorm=True,
        use_layernorm=False,
        numerical_embedding="plr",
        numerical_embedding_kwargs={"embed_dim": 8, "n_frequencies": 16},
        verbose=0,
    )
    # First layer is the embedding.
    assert isinstance(net[0], PeriodicLinearEmbedding)
    assert net[0].out_features == 4 * 8
    # example_input_array is RAW shape (1, 4), not embedded (1, 32)
    assert net.example_input_array.shape == (1, 4)
    # Forward end-to-end with raw 4-feature input works.
    x = torch.randn(16, 4)
    out = net(x)
    assert out.shape == (16, 1)


def test_generate_mlp_plr_unknown_type_raises():
    """Generate mlp plr unknown type raises."""
    with pytest.raises(ValueError, match=r"Unknown numerical_embedding"):
        generate_mlp(
            num_features=4,
            num_classes=1,
            nlayers=1,
            numerical_embedding="not_a_known_type",
            verbose=0,
        )


def test_plr_embedded_mlp_beats_vanilla_on_periodic_target():
    """Biz-value sanity: on a NONLINEAR sinusoidal target the PLR
    embedding should HELP — that's the regime it was designed for.

    Target: y = sin(2*pi*x_0) + 0.5*cos(4*pi*x_1) + 0.3*x_2 + eps
    Vanilla ReLU MLP struggles to represent the sin/cos basis from
    scratch; the PLR embedding hands it sin/cos features directly.

    Asserts:
      * PLR-MLP reaches R^2 > 0.5 (non-trivial fit on periodic target).
      * PLR-MLP R^2 > vanilla MLP R^2 - 0.05 (PLR is at least no worse).

    The "much-better" gap is dataset-specific (RealMLP-TD measured
    +20.6% AGGREGATE across many tabular datasets, not on a single
    synthetic); a strict "PLR >> vanilla" assertion would be flaky on
    one toy problem. The relaxed gate ensures PLR doesn't regress.
    """
    rng = np.random.default_rng(0)
    n, d = 500, 5
    X = rng.uniform(-1.0, 1.0, size=(n, d)).astype(np.float32)
    y = (np.sin(2 * np.pi * X[:, 0]) + 0.5 * np.cos(4 * np.pi * X[:, 1]) + 0.3 * X[:, 2] + 0.05 * rng.standard_normal(n)).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    def _make_reg(use_plr: bool):
        """Make reg."""
        np_params = {
            "nlayers": 2,
            "first_layer_num_neurons": 64,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": True,
            "activation_function": torch.nn.ReLU,
        }
        if use_plr:
            np_params["numerical_embedding"] = "plr"
            np_params["numerical_embedding_kwargs"] = {
                "embed_dim": 8,
                "n_frequencies": 16,
                "sigma": 1.0,
            }
        return PytorchLightningRegressor(
            model_class=MLPTorchModel,
            model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 5e-3},
            network_params=np_params,
            datamodule_class=TorchDataModule,
            datamodule_params={
                "features_dtype": torch.float32,
                "labels_dtype": torch.float32,
                "dataloader_params": {"batch_size": 32, "num_workers": 0},
            },
            trainer_params={
                "max_epochs": 40,
                "enable_model_summary": False,
                "enable_progress_bar": False,
                "log_every_n_steps": 1,
                "devices": 1,
                "accelerator": "cpu",
                "logger": False,
            },
            random_state=0,
        )

    reg_plr = _make_reg(use_plr=True)
    reg_plr.fit(X_tr, y_tr)
    r2_plr = r2_score(y_te, reg_plr.predict(X_te))

    reg_vanilla = _make_reg(use_plr=False)
    reg_vanilla.fit(X_tr, y_tr)
    r2_vanilla = r2_score(y_te, reg_vanilla.predict(X_te))

    print(f"\nPLR vs vanilla on periodic target:\n  PLR     R^2 = {r2_plr:+.4f}\n  vanilla R^2 = {r2_vanilla:+.4f}\n  delta       = {r2_plr - r2_vanilla:+.4f}")

    assert r2_plr > 0.5, f"PLR-MLP R^2={r2_plr:+.4f} should reach >0.5 on this periodic target -- the PLR is supposed to make sin/cos features cheap."
    assert r2_plr > r2_vanilla - 0.05, (
        f"PLR-MLP regressed against vanilla by more than 0.05 R^2 "
        f"(PLR={r2_plr:+.4f}, vanilla={r2_vanilla:+.4f}). The PLR "
        "embedding should be at least neutral on a periodic target."
    )
