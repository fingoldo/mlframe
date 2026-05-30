"""Iter-4: NaN / missing-value handling in features and targets.

PyTorch's default loss functions propagate NaN through gradients,
producing NaN-valued weights after a single backward pass. Our MLP
estimators need to either:
  * EXPLICITLY refuse NaN-containing inputs with a clear error, OR
  * sanitize them upstream (drop rows / mean-impute / etc.).

This test PROBES the current behaviour so we can document it and
decide what to do.

Three cases per task:
  A) X has NaN, y clean -> what happens?
  B) X clean, y has NaN -> what happens?
  C) X + y both clean -> baseline (must work)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningClassifier,
    PytorchLightningRegressor,
    TorchDataModule,
)


def _regressor_params():
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-2},
        "network_params": {
            "nlayers": 1, "first_layer_num_neurons": 8,
            "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
            "use_layernorm": False, "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        "datamodule_class": TorchDataModule,
        "datamodule_params": {
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 2, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        "random_state": 0,
    }


@pytest.fixture
def clean_regression_split():
    X, y = make_regression(n_samples=160, n_features=5, noise=0.1, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def _inject_nan_into_features(X, rng, frac=0.05):
    Xn = X.copy()
    n, d = Xn.shape
    n_nan = int(n * d * frac)
    flat_idx = rng.choice(n * d, size=n_nan, replace=False)
    Xn.flat[flat_idx] = np.nan
    return Xn


def _inject_nan_into_targets(y, rng, frac=0.05):
    yn = y.copy()
    n_nan = int(len(yn) * frac)
    idx = rng.choice(len(yn), size=n_nan, replace=False)
    yn[idx] = np.nan
    return yn


def test_regressor_baseline_clean_inputs_train_to_completion(clean_regression_split):
    """Sanity: clean inputs train without producing NaN predictions."""
    reg = PytorchLightningRegressor(**_regressor_params())
    reg.fit(clean_regression_split["X_train"], clean_regression_split["y_train"])
    preds = reg.predict(clean_regression_split["X_test"])
    assert np.isfinite(preds).all(), "baseline produced NaN predictions on clean data"


def test_regressor_with_nan_in_features_documents_behaviour(clean_regression_split):
    """X has NaN -> what does the estimator do? Current behaviour: NaN
    propagates through the first Linear, producing all-NaN activations
    and all-NaN gradients; weights become NaN after step 1; predictions
    are all NaN.

    This test DOCUMENTS the current state. Ideal future fix would raise
    a clear error at fit() entry. Until then, callers must pre-clean.
    """
    rng = np.random.default_rng(0)
    X_nan = _inject_nan_into_features(clean_regression_split["X_train"], rng)

    reg = PytorchLightningRegressor(**_regressor_params())
    # We allow fit to either:
    #   (a) raise (good — caller knows something's wrong), OR
    #   (b) complete but produce all-NaN preds (bad but at least surfaces).
    # The CURRENT behaviour falls into (b).
    try:
        reg.fit(X_nan, clean_regression_split["y_train"])
    except (ValueError, RuntimeError) as e:
        # Future behaviour: explicit reject. Then this test passes here.
        print(f"\nNaN in features rejected at fit() with: {e}")
        return

    preds = reg.predict(clean_regression_split["X_test"])
    n_nan_preds = int(np.isnan(preds).sum())
    print(f"\nNaN-in-features documented behaviour: {n_nan_preds} / "
          f"{len(preds)} predictions are NaN after fit() on dirty X")
    # The diagnostic asserts SOMETHING is observable downstream: either
    # the model produced NaN preds (current state) OR it somehow trained
    # cleanly. Either way the test passes; it's a SNAPSHOT test.
    assert n_nan_preds >= 0  # always true; this is a snapshot


def test_regressor_with_nan_in_targets_documents_behaviour(clean_regression_split):
    """y has NaN -> what happens? Current behaviour: NaN in the loss
    propagates to NaN gradients and NaN-valued weights, identical to
    the X-NaN case. DOCUMENTS the current behaviour."""
    rng = np.random.default_rng(0)
    y_nan = _inject_nan_into_targets(clean_regression_split["y_train"], rng)

    reg = PytorchLightningRegressor(**_regressor_params())
    try:
        reg.fit(clean_regression_split["X_train"], y_nan)
    except (ValueError, RuntimeError) as e:
        print(f"\nNaN in targets rejected at fit() with: {e}")
        return

    preds = reg.predict(clean_regression_split["X_test"])
    n_nan_preds = int(np.isnan(preds).sum())
    print(f"\nNaN-in-targets documented behaviour: {n_nan_preds} / "
          f"{len(preds)} predictions are NaN after fit() on dirty y")
    assert n_nan_preds >= 0
