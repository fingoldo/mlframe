"""F-67 prediction-trainer caching was REVERTED (2026-06-02): reusing one
L.Trainer across predict() calls accumulates Lightning's prediction-loop state
and every predict past the first raised "Mismatch in number of limits". A fresh
Trainer is now built per predict(); ``_prediction_trainer_cache`` exists only
for pickle-state symmetry and stays empty.

These tests pin the revert's contract: repeated predicts are correct, the cache
never populates, and the pickle round-trip stays clean (F-73b).
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
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    reg = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": nn.MSELoss(), "learning_rate": 1e-2, "load_best_weights_on_train_end": False},
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
            "max_epochs": 1,
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
    return reg, X_te


def test_prediction_trainer_cache_attribute_exists_and_stays_empty(fitted_regressor):
    """The cache attribute must exist on a freshly-constructed estimator (no
    AttributeError) and stay empty after predict() -- caching is reverted."""
    reg, X_te = fitted_regressor
    assert hasattr(reg, "_prediction_trainer_cache")
    assert reg._prediction_trainer_cache == {}
    _ = reg.predict(X_te)
    assert reg._prediction_trainer_cache == {}


def test_repeated_predict_is_consistent_and_uncached(fitted_regressor):
    """The revert's core guarantee: many predicts in a row all succeed and
    return identical results (the caching bug failed every predict past the
    first), and the cache never grows."""
    reg, X_te = fitted_regressor
    first = reg.predict(X_te)
    for _ in range(4):
        np.testing.assert_allclose(reg.predict(X_te), first, rtol=1e-4, atol=1e-5)
    assert reg._prediction_trainer_cache == {}


def test_f73b_predict_then_pickle_roundtrips_clean(fitted_regressor):
    """F-73b regression: pickling the estimator AFTER a predict() must succeed
    (no live L.Trainer / WarningCache leak) and the restored estimator predicts
    the same values. __getstate__ drops the cache; the restored estimator starts
    with an empty cache.
    """
    import pickle

    reg, X_te = fitted_regressor
    pred_before = reg.predict(X_te)

    state = reg.__getstate__()
    assert "_prediction_trainer_cache" not in state, "__getstate__ must drop the runtime trainer cache"
    assert state.get("trainer") is None

    reg2 = pickle.loads(pickle.dumps(reg))
    assert reg2._prediction_trainer_cache == {}, "restored estimator must start with an empty trainer cache"
    pred_after = reg2.predict(X_te)
    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-4, atol=1e-5)
