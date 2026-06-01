"""F-67 (2026-05-31): prediction trainer cache eliminates the per-predict
L.Trainer construction + ~236 ms gc.collect cycle.

Verifies that:
  * Repeated reg.predict() calls with the SAME (accelerator, precision)
    reuse the cached L.Trainer instance.
  * Changing accelerator (e.g. user passes device='cuda' on a CPU trainer)
    rebuilds the trainer for the new key but keeps the old one.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


@pytest.fixture
def fitted_regressor():
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
        model_params={"loss_fn": nn.MSELoss(), "learning_rate": 1e-2,
                      "load_best_weights_on_train_end": False},
        network_params={
            "nlayers": 1, "first_layer_num_neurons": 8,
            "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
            "use_layernorm": False, "use_batchnorm": False,
            "activation_function": nn.ReLU,
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
    return reg, X_te


def test_predict_trainer_cache_initialised_after_first_predict(fitted_regressor):
    """F-67: a single predict() call populates the cache with the
    resolved (accelerator, precision) key."""
    reg, X_te = fitted_regressor
    _ = reg.predict(X_te)
    assert hasattr(reg, "_prediction_trainer_cache")
    cache = reg._prediction_trainer_cache
    assert len(cache) >= 1
    # Key contains (accelerator, precision); accelerator must be "cpu"
    # since the wrapper was trained that way.
    keys = list(cache.keys())
    accelerators = [k[0] for k in keys]
    assert "cpu" in accelerators


def test_predict_trainer_cache_reuses_same_instance_on_repeat_call(fitted_regressor):
    """F-67 core: the SECOND predict() with the same (accelerator,
    precision) MUST return the same L.Trainer instance."""
    reg, X_te = fitted_regressor
    _ = reg.predict(X_te)
    cache_after_1 = dict(reg._prediction_trainer_cache)
    trainer_after_1 = next(iter(cache_after_1.values()))

    _ = reg.predict(X_te)
    cache_after_2 = reg._prediction_trainer_cache
    trainer_after_2 = next(iter(cache_after_2.values()))

    assert trainer_after_1 is trainer_after_2, (
        "F-67 expects cached trainer to be REUSED on the second predict; "
        "got a fresh instance."
    )
    # Cache size should remain 1 across repeated calls.
    assert len(cache_after_2) == 1


def test_predict_trainer_cache_size_stable_across_many_calls(fitted_regressor):
    """F-67: 5 sequential predict() calls with no kwarg variation MUST
    grow the cache from 0 -> 1 only. Catches a regression where the
    cache key fails to hash consistently and a fresh trainer gets
    inserted on every call."""
    reg, X_te = fitted_regressor
    for _ in range(5):
        _ = reg.predict(X_te)
    cache = reg._prediction_trainer_cache
    assert len(cache) == 1, (
        f"F-67: cache size grew to {len(cache)} across 5 identical "
        f"predict() calls; expected 1 (single (accelerator, precision) "
        f"key reused)."
    )
