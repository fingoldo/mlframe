"""F-C + F-G + F-H regression tests (2026-05-31 silent-correctness audit
follow-up batch 3).

F-C: Lookahead.add_param_group must initialise slow weights for the new
     params eagerly, not lazily on first k-sync.
F-G: PytorchLightningEstimator._predict_raw caches the accelerator name
     on self before nullifying self.trainer, so a subsequent
     predict() resolves to the same device.
F-H: RecurrentDataset.__getitem__ produces a sequence tensor whose
     storage is INDEPENDENT of self.sequences[idx], not a zero-copy
     view (which would silently corrupt the dataset if a downstream
     subclass introduces an in-place op).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from mlframe.training.neural._lookahead_optimizer import Lookahead


# --- F-C --------------------------------------------------------------------


def test_lookahead_add_param_group_eager_inits_slow():
    """F-C fix: ``add_param_group`` must populate ``_slow_weights`` for
    the new params immediately (matches the construction-time eager init
    in F-A), so the first k-sync after the add runs a real interpolation.
    """
    p_init = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.SGD([p_init], lr=0.1)
    lh = Lookahead(base, k=3, alpha=0.5)
    assert id(p_init) in lh._slow_weights  # eager-init from construction

    # Mid-fit: add a new param group (e.g. a freshly-instantiated head).
    p_new = torch.nn.Parameter(torch.ones(2) * 7.0)
    lh.add_param_group({"params": [p_new]})
    assert id(p_new) in lh._slow_weights, (
        "F-C: add_param_group must eager-init slow weights for new params"
    )
    # The cached slow MUST equal the current data of the new param.
    torch.testing.assert_close(lh._slow_weights[id(p_new)], p_new.data)


def test_lookahead_add_param_group_does_not_overwrite_existing_slow():
    """Idempotency: adding the SAME param twice (rare but possible via
    user error) must NOT overwrite the existing slow snapshot."""
    p = torch.nn.Parameter(torch.zeros(4))
    base = torch.optim.SGD([p], lr=0.1)
    lh = Lookahead(base, k=3, alpha=0.5)
    # Mutate slow to a distinctive value.
    lh._slow_weights[id(p)] = torch.full((4,), 99.0)
    # add_param_group on the same param (would be rejected by base
    # optimizer, but our pre-check is by id-presence).
    try:
        lh.add_param_group({"params": [p]})
    except ValueError:
        pass  # base optimizer rejects duplicate; that's fine
    # Slow value untouched.
    torch.testing.assert_close(
        lh._slow_weights[id(p)], torch.full((4,), 99.0),
    )


# --- F-G --------------------------------------------------------------------


def test_predict_raw_caches_accelerator_name_for_next_call():
    """F-G fix: after _predict_raw returns, self._last_predict_accelerator
    holds the accelerator name used. Pre-fix the next predict() would
    fall through to ``accelerator="auto"`` (might silently switch device).
    """
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningRegressor,
        TorchDataModule,
    )

    X, y = make_regression(n_samples=64, n_features=4, random_state=0)
    X = X.astype(np.float32); y = y.astype(np.float32)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)

    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-2,
                      "load_best_weights_on_train_end": False},
        network_params={
            "nlayers": 1, "first_layer_num_neurons": 8,
            "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
            "use_layernorm": False, "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        datamodule_class=TorchDataModule,
        datamodule_params={
            "features_dtype": torch.float32, "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={
            "max_epochs": 1, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        random_state=0,
    )
    reg.fit(X_tr, y_tr)
    # First predict establishes the cached accelerator name.
    _ = reg.predict(X_te)
    assert hasattr(reg, "_last_predict_accelerator")
    assert reg._last_predict_accelerator == "cpu", (
        f"F-G: expected cached accelerator='cpu', got "
        f"{reg._last_predict_accelerator!r}"
    )

    # Second predict (trainer is now None) must use the cached value
    # rather than falling through to 'auto'. We can't directly inspect
    # the trainer created inside _predict_raw, but we can verify the
    # second predict doesn't crash AND returns finite predictions.
    pred2 = reg.predict(X_te)
    assert pred2.shape == (X_te.shape[0],)
    assert np.isfinite(pred2).all()


# --- F-H --------------------------------------------------------------------


def test_recurrent_dataset_getitem_independent_of_source_numpy():
    """F-H fix: ``__getitem__`` returns a sequence tensor that does NOT
    share storage with ``self.sequences[idx]``. Mutating the returned
    tensor in place must NOT corrupt the source numpy array."""
    from mlframe.training.neural._recurrent_data import RecurrentDataset

    seqs = [
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
    ]
    labels = np.array([0.5, -0.5], dtype=np.float32)
    ds = RecurrentDataset(
        sequences=seqs, aux_features=None,
        labels=labels, sample_weights=None,
        is_regression=True,
    )
    item = ds[0]
    seq_t = item["sequence"]
    assert seq_t.shape == (3, 2)

    # The tensor's storage MUST NOT alias the numpy array's buffer.
    # In PyTorch 2.x, a non-zero-copy convert produces a tensor whose
    # .data_ptr() differs from the source array.
    src_ptr = seqs[0].__array_interface__["data"][0]
    assert seq_t.data_ptr() != src_ptr, (
        "F-H: __getitem__ returned a zero-copy view; an in-place op on "
        "the per-sample tensor would silently corrupt the source array."
    )

    # Mutate the returned tensor in place; source array must stay intact.
    seq_t.add_(100.0)
    np.testing.assert_array_equal(
        seqs[0],
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )


def test_recurrent_dataset_getitem_still_returns_float32():
    """F-H must NOT change the dtype contract."""
    from mlframe.training.neural._recurrent_data import RecurrentDataset

    seqs = [np.array([[1.0]], dtype=np.float64)]  # float64 source
    labels = np.array([0.0], dtype=np.float32)
    ds = RecurrentDataset(
        sequences=seqs, aux_features=None,
        labels=labels, sample_weights=None,
        is_regression=True,
    )
    item = ds[0]
    assert item["sequence"].dtype == torch.float32
