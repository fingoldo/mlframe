"""Integration: the default-on MonotonicDeclineStopCallback ends a real MLP fit early on an
overfit-prone target, using FEWER epochs than the no-monotonic baseline at the same holdout.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mlframe.training.neural import PytorchLightningRegressor
from mlframe.training.neural.flat import MLPTorchModel
from mlframe.training.neural.data import TorchDataModule


def _params(max_epochs=50, monotonic=3):
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.MSELoss(), "learning_rate": 5e-2},
        "network_params": {
            "nlayers": 3,
            "first_layer_num_neurons": 128,
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
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": max_epochs,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
        "random_state": 0,
        "early_stopping_rounds": 1000,  # huge patience -> isolate the monotonic effect
        "monotonic_decline_patience": monotonic,
    }


def _ood_data(seed=0, n=200, d=30):
    """Group-OOD: the net memorizes the train sign-relation, so val (sign-flipped target) keeps
    diverging once train is fit -- a clean monotone-rising val curve the detector should catch."""
    rng = np.random.RandomState(seed)
    Xtr = rng.randn(n, d).astype(np.float32)
    ytr = (Xtr[:, 0] * 3.0).astype(np.float32)
    Xv = rng.randn(60, d).astype(np.float32)
    yv = (-Xv[:, 0] * 3.0).astype(np.float32)
    return Xtr, ytr, Xv, yv


def _fit(monotonic):
    import pandas as pd

    Xtr, ytr, Xv, yv = _ood_data()
    Xtr_df, Xv_df = pd.DataFrame(Xtr), pd.DataFrame(Xv)
    est = PytorchLightningRegressor(**_params(monotonic=monotonic))
    est.fit(Xtr_df, pd.Series(ytr), eval_set=(Xv_df, pd.Series(yv)))
    n_epochs = est.model.current_epoch if hasattr(est.model, "current_epoch") else None
    rmse = float(np.sqrt(np.mean((est.predict(Xv_df).reshape(-1) - yv) ** 2)))
    return est, n_epochs, rmse


@pytest.mark.timeout(240)
def test_monotonic_decline_stops_mlp_early():
    _est_mono, ep_mono, rmse_mono = _fit(monotonic=3)
    _est_none, ep_none, rmse_none = _fit(monotonic=None)

    # The trained-epoch count is recorded on the lightning module; the monotonic run must use
    # strictly fewer epochs than the baseline that ran toward the full budget.
    assert ep_mono is not None and ep_none is not None
    assert ep_mono < ep_none, f"monotonic stop did not shorten training: {ep_mono} vs {ep_none} epochs"
    # Holdout must not be hurt -- BestEpochModelCheckpoint restores the global-best epoch either way.
    assert rmse_mono <= rmse_none + 0.10, f"monotonic stop hurt holdout RMSE: {rmse_mono:.3f} vs {rmse_none:.3f}"
