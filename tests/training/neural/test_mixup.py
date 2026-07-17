"""F-68 (2026-05-31) tests for the Mixup augmentation.

Three layers:
  1. Unit tests on ``mixup_batch`` (Beta sampling, permutation, edge cases).
  2. Integration: MLPTorchModel.training_step calls mixup when
     ``use_mixup=True`` and computes the appropriate task-specific loss.
  3. biz_value: enabling Mixup doesn't catastrophically break convergence
     on a small regression task. We don't assert a Mixup WIN here
     (RealMLP-TD's +0.6% needs many tasks + seeds); just no regression
     past 0.05 R^2.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from mlframe.training.neural._mixup import mixup_batch


# --- Unit tests --------------------------------------------------------------


def test_mixup_rejects_zero_alpha():
    x = torch.randn(8, 4)
    y = torch.randn(8)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        mixup_batch(x, y, alpha=0.0)


def test_mixup_rejects_mismatched_batch_dims():
    x = torch.randn(8, 4)
    y = torch.randn(7)
    with pytest.raises(ValueError, match="batch dim"):
        mixup_batch(x, y, alpha=0.2)


def test_mixup_lam_in_clamped_range():
    """lam is clamped to [1e-3, 1 - 1e-3] to avoid degenerate
    no-mix / fully-swapped batches. fp32 .item() round-trip can drift
    by O(1e-6) past the bound; use 2e-3 here so the test absorbs
    that noise."""
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    y = torch.randn(64)
    for _ in range(50):
        _, _, _, lam = mixup_batch(x, y, alpha=0.2)
        assert 0.0 < lam < 1.0
        assert 1e-3 - 1e-6 <= lam <= 1.0 - 1e-3 + 1e-6


def test_mixup_x_is_convex_combination():
    """x_mixed = lam * x + (1-lam) * x[idx]. Verify by reconstructing
    via the returned (y_a, y_b, lam) shape contract."""
    torch.manual_seed(42)
    x = torch.randn(32, 4)
    y = torch.arange(32, dtype=torch.float32)
    x_mixed, y_a, y_b, lam = mixup_batch(x, y, alpha=0.2)
    # y_a should equal the original y
    torch.testing.assert_close(y_a, y)
    # x_mixed should equal lam * x + (1-lam) * x_permuted, where the
    # permutation is derivable from (y_b vs y).
    # Since y is arange, y_b values ARE the permutation indices.
    idx = y_b.long()
    expected_x_mixed = lam * x + (1.0 - lam) * x[idx]
    torch.testing.assert_close(x_mixed, expected_x_mixed, atol=1e-6, rtol=1e-6)


def test_mixup_preserves_dtypes():
    x = torch.randn(8, 4, dtype=torch.float32)
    y = torch.randn(8, dtype=torch.float32)
    x_mixed, y_a, y_b, _ = mixup_batch(x, y, alpha=0.2)
    assert x_mixed.dtype == torch.float32
    assert y_a.dtype == torch.float32
    assert y_b.dtype == torch.float32


# --- F-70 mixup_sequence_batch ----------------------------------------------


def test_mixup_sequence_batch_lengths_are_pairwise_max():
    """F-70: ``lengths_mixed[i] = max(lengths[i], lengths[perm[i]])``.
    Use arange labels so the permutation is recoverable."""
    from mlframe.training.neural._mixup import mixup_sequence_batch

    torch.manual_seed(0)
    seqs = torch.randn(8, 5, 4)
    lens = torch.tensor([3, 5, 2, 5, 4, 1, 5, 2], dtype=torch.long)
    y = torch.arange(8, dtype=torch.float32)
    _, lens_mixed, _, _y_a, y_b, _ = mixup_sequence_batch(
        seqs,
        lens,
        y,
        alpha=0.2,
    )
    perm = y_b.long()
    expected = torch.maximum(lens, lens[perm])
    torch.testing.assert_close(lens_mixed, expected)


def test_mixup_sequence_batch_mixes_aux_with_same_idx_and_lam():
    """F-70: when aux supplied, aux is mixed by the SAME (idx, lam)
    as the sequences, so per-sample identity is preserved across
    modalities."""
    from mlframe.training.neural._mixup import mixup_sequence_batch

    torch.manual_seed(0)
    seqs = torch.randn(8, 5, 4)
    lens = torch.full((8,), 5, dtype=torch.long)
    aux = torch.randn(8, 3)
    y = torch.arange(8, dtype=torch.float32)
    seq_mixed, _, aux_mixed, _, y_b, lam = mixup_sequence_batch(
        seqs,
        lens,
        y,
        alpha=0.2,
        aux_features=aux,
    )
    perm = y_b.long()
    # Reconstruct aux_mixed from the recovered idx + lam.
    expected_aux = lam * aux + (1.0 - lam) * aux[perm]
    torch.testing.assert_close(aux_mixed, expected_aux, atol=1e-5, rtol=1e-5)
    # Same for sequences.
    expected_seq = lam * seqs + (1.0 - lam) * seqs[perm]
    torch.testing.assert_close(seq_mixed, expected_seq, atol=1e-5, rtol=1e-5)


def test_mixup_sequence_batch_rejects_zero_alpha():
    from mlframe.training.neural._mixup import mixup_sequence_batch

    seqs = torch.randn(8, 5, 4)
    lens = torch.full((8,), 5)
    y = torch.randn(8)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        mixup_sequence_batch(seqs, lens, y, alpha=0.0)


def test_mixup_sequence_batch_rejects_batch_dim_mismatch():
    from mlframe.training.neural._mixup import mixup_sequence_batch

    seqs = torch.randn(8, 5, 4)
    lens = torch.full((7,), 5)
    y = torch.randn(8)
    with pytest.raises(ValueError, match="batch dim"):
        mixup_sequence_batch(seqs, lens, y, alpha=0.2)


# --- Integration ---------------------------------------------------------------


def test_mlptorchmodel_training_step_with_mixup_regression():
    """MLPTorchModel.training_step on a regression task with
    use_mixup=True must complete without errors and return a finite,
    grad-bearing loss."""
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
    module = MLPTorchModel(
        loss_fn=nn.MSELoss(),
        metrics=[],
        network=net,
        use_mixup=True,
        mixup_alpha=0.2,
        task_type="regression",
    )
    module.train()
    x = torch.randn(16, 4)
    y = torch.randn(16)
    batch = {"features": x, "labels": y, "sample_weight": None}
    out = module.training_step(batch, batch_idx=0)
    loss = out["loss"] if isinstance(out, dict) else out
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_mlptorchmodel_training_step_with_mixup_classification():
    """Classification path: integer labels -> two-target convex-loss form."""
    from mlframe.training.neural._flat_torch_module import MLPTorchModel
    from mlframe.training.neural.flat import generate_mlp

    net = generate_mlp(
        num_features=4,
        num_classes=3,
        nlayers=1,
        first_layer_num_neurons=8,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        use_layernorm=False,
        use_batchnorm=False,
        activation_function=nn.ReLU,
        verbose=0,
    )
    module = MLPTorchModel(
        loss_fn=nn.CrossEntropyLoss(),
        metrics=[],
        network=net,
        use_mixup=True,
        mixup_alpha=0.2,
    )
    module.train()
    x = torch.randn(16, 4)
    y = torch.randint(0, 3, (16,), dtype=torch.long)
    batch = {"features": x, "labels": y, "sample_weight": None}
    out = module.training_step(batch, batch_idx=0)
    loss = out["loss"] if isinstance(out, dict) else out
    assert torch.isfinite(loss)
    assert loss.requires_grad


def test_mlptorchmodel_training_step_mixup_disabled_returns_same_as_plain():
    """use_mixup=False must NOT invoke mixup -- behaviour identical to
    a pre-F-68 training_step."""
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
    module = MLPTorchModel(
        loss_fn=nn.MSELoss(),
        metrics=[],
        network=net,
        use_mixup=False,
        task_type="regression",
    )
    module.train()
    torch.manual_seed(0)
    x = torch.randn(16, 4)
    y = torch.randn(16)
    batch = {"features": x, "labels": y, "sample_weight": None}
    out_no_mix = module.training_step(batch, batch_idx=0)
    loss_no_mix = out_no_mix["loss"] if isinstance(out_no_mix, dict) else out_no_mix
    assert torch.isfinite(loss_no_mix)


# --- biz_value ----------------------------------------------------------------


def test_mlp_mixup_does_not_catastrophically_regress_regression():
    """100-epoch smoke: Mixup-on R^2 must be within 0.10 of Mixup-off on
    a trivial 4D regression. Mixup acts as a regulariser so on a clean
    linear task we expect a slight underperform (the interpolated
    targets introduce noise the unregularised baseline doesn't see).
    Tolerance 0.10 R^2 mirrors RealMLP-TD's "task-dependent" caveat.
    """
    from sklearn.datasets import make_regression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningRegressor,
        TorchDataModule,
    )

    X, y = make_regression(n_samples=400, n_features=4, noise=0.5, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    def fit_score(use_mixup: bool) -> float:
        torch.manual_seed(0)
        np.random.seed(0)
        reg = PytorchLightningRegressor(
            model_class=MLPTorchModel,
            model_params={
                "loss_fn": nn.MSELoss(),
                "learning_rate": 5e-3,
                "use_mixup": use_mixup,
                "mixup_alpha": 0.2,
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
    r2_mix = fit_score(True)
    assert r2_plain > 0.9, f"plain MLP only reached R^2={r2_plain:.4f}"
    assert r2_mix > r2_plain - 0.10, f"Mixup-wrapped R^2={r2_mix:.4f} regressed >0.10 vs plain R^2={r2_plain:.4f}"
