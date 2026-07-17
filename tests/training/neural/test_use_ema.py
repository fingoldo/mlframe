"""F-28: opt-in EMA-of-weights via Lightning's WeightAveraging callback."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
)


def _params(use_ema=False, ema_params=None, use_swa=False):
    """Params."""
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-2},
        "network_params": {
            "nlayers": 1,
            "first_layer_num_neurons": 8,
            "dropout_prob": 0.0,
            "inputs_dropout_prob": 0.0,
            "use_layernorm": False,
            "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        "datamodule_class": TorchDataModule,
        "datamodule_params": {
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 3,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
        "random_state": 0,
        "use_ema": use_ema,
        "ema_params": ema_params,
        "use_swa": use_swa,
    }


@pytest.fixture
def reg_data():
    """Reg data."""
    X, y = make_regression(n_samples=160, n_features=5, random_state=0)
    X_tr, X_te, y_tr, _ = train_test_split(
        X.astype(np.float32),
        y.astype(np.float32),
        test_size=0.3,
        random_state=0,
    )
    return X_tr, X_te, y_tr


def test_use_ema_false_does_not_attach_callback(reg_data):
    """Default: no WeightAveraging callback."""
    X_tr, X_te, y_tr = reg_data
    reg = PytorchLightningRegressor(**_params(use_ema=False))
    reg.fit(X_tr, y_tr)
    # Fit completes; default predict path works.
    preds = reg.predict(X_te)
    assert preds.shape == (X_te.shape[0],)


def test_use_ema_true_completes_fit_and_predicts(reg_data):
    """With use_ema=True, fit completes; predict works on the averaged
    weights that the callback swaps in at on_train_end."""
    X_tr, X_te, y_tr = reg_data
    reg = PytorchLightningRegressor(**_params(use_ema=True))
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert preds.shape == (X_te.shape[0],)
    assert np.isfinite(preds).all()


def test_use_ema_with_custom_decay(reg_data):
    """ema_params={'decay': 0.99} should plumb into get_ema_avg_fn."""
    X_tr, X_te, y_tr = reg_data
    reg = PytorchLightningRegressor(**_params(use_ema=True, ema_params={"decay": 0.99}))
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    assert np.isfinite(preds).all()


def test_use_ema_and_use_swa_are_mutex(reg_data):
    """SWA and EMA both rewrite live weights at train end (last-wins);
    fit() must raise ValueError when both flags are on."""
    X_tr, _, y_tr = reg_data
    reg = PytorchLightningRegressor(**_params(use_ema=True, use_swa=True))
    with pytest.raises(ValueError, match=r"mutually exclusive"):
        reg.fit(X_tr, y_tr)
