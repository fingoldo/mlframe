"""C4: verify torch.optim.lr_scheduler.CosineAnnealingWarmRestarts plumbs
through the existing lr_scheduler / lr_scheduler_kwargs path.

The research-agent recommendation (2026-05-31) was to expose this
scheduler alongside OneCycleLR because RealMLP-TD uses multi-cycle
cosine (``coslog4``) as its default for 50-100 epoch fits and it
gives better final minima than a single OneCycle.

No new code is required — the existing
``MLPTorchModel(lr_scheduler=..., lr_scheduler_kwargs=...)`` API
constructs any user-supplied scheduler class. The test PINS that
contract end-to-end so a future scheduler-handling refactor surfaces
visibly if it breaks the CosineAnnealingWarmRestarts path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel, PytorchLightningRegressor, TorchDataModule,
)


def _params(lr_scheduler=None, lr_scheduler_kwargs=None,
            lr_scheduler_interval="epoch", lr_scheduler_monitor=None):
    return {
        "model_class": MLPTorchModel,
        "model_params": {
            "loss_fn": torch.nn.MSELoss(),
            "learning_rate": 1e-2,
            "lr_scheduler": lr_scheduler,
            "lr_scheduler_kwargs": lr_scheduler_kwargs,
            "lr_scheduler_interval": lr_scheduler_interval,
            "lr_scheduler_monitor": lr_scheduler_monitor,
        },
        "network_params": {
            "nlayers": 1, "first_layer_num_neurons": 8,
            "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
            "use_layernorm": False, "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        "datamodule_class": TorchDataModule,
        "datamodule_params": {
            "features_dtype": torch.float32, "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 4, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        "random_state": 0,
    }


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=128, n_features=4, random_state=0)
    X_tr, X_te, y_tr, _ = train_test_split(
        X.astype(np.float32), y.astype(np.float32), test_size=0.3, random_state=0,
    )
    return X_tr, X_te, y_tr


def test_cosine_warm_restarts_completes_fit(reg_data):
    """CosineAnnealingWarmRestarts(T_0, T_mult, eta_min) must plumb
    through the lr_scheduler API and the fit must complete."""
    X_tr, X_te, y_tr = reg_data
    params = _params(
        lr_scheduler=CosineAnnealingWarmRestarts,
        lr_scheduler_kwargs={"T_0": 2, "T_mult": 2, "eta_min": 1e-5},
        lr_scheduler_interval="epoch",
    )
    reg = PytorchLightningRegressor(**params)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (X_te.shape[0],)
    assert np.isfinite(preds).all()


def test_one_cycle_still_works(reg_data):
    """Regression check: the OneCycleLR special-case path still works
    after the C4 documentation change."""
    X_tr, X_te, y_tr = reg_data
    params = _params(
        lr_scheduler=OneCycleLR,
        lr_scheduler_kwargs={"max_lr": 1e-2},
        lr_scheduler_interval="step",
    )
    reg = PytorchLightningRegressor(**params)
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (X_te.shape[0],)


def test_reduce_on_plateau_still_works(reg_data):
    """ReduceLROnPlateau requires a monitor metric -- regression check
    that the lr_scheduler_monitor path is intact."""
    X_tr, X_te, y_tr = reg_data
    # Need validation for the monitored metric to exist.
    X_tr2, X_val, y_tr2, y_val = train_test_split(X_tr, y_tr, test_size=0.3, random_state=1)
    params = _params(
        lr_scheduler=ReduceLROnPlateau,
        lr_scheduler_kwargs={"mode": "min", "factor": 0.5, "patience": 1},
        lr_scheduler_interval="epoch",
        lr_scheduler_monitor="val_loss",
    )
    reg = PytorchLightningRegressor(**params)
    reg.fit(X_tr2, y_tr2, eval_set=(X_val, y_val))
    preds = reg.predict(X_te)
    assert preds.shape == (X_te.shape[0],)
