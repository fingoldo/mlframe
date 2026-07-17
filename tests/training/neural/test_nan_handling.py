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
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
)


def _regressor_params():
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
            "max_epochs": 2,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
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


def test_regressor_rejects_nan_in_features(clean_regression_split):
    """F-23: fit() must reject NaN in X with a clear ValueError pointing
    to sklearn imputation helpers. Pre-fix NaN propagated silently to
    all-NaN predictions."""
    rng = np.random.default_rng(0)
    X_nan = _inject_nan_into_features(clean_regression_split["X_train"], rng)

    reg = PytorchLightningRegressor(**_regressor_params())
    with pytest.raises(ValueError, match=r"NaN.*SimpleImputer|SimpleImputer.*NaN|contains \d+ NaN"):
        reg.fit(X_nan, clean_regression_split["y_train"])


def test_regressor_rejects_nan_in_targets(clean_regression_split):
    """F-23: fit() must reject NaN in y with a clear ValueError."""
    rng = np.random.default_rng(0)
    y_nan = _inject_nan_into_targets(clean_regression_split["y_train"], rng)

    reg = PytorchLightningRegressor(**_regressor_params())
    with pytest.raises(ValueError, match=r"contains \d+ NaN"):
        reg.fit(clean_regression_split["X_train"], y_nan)


def test_regressor_rejects_inf_in_features(clean_regression_split):
    """F-23: inf values are equally bad for training (+inf * 0 = NaN in
    matmul); must also be rejected."""
    X_inf = clean_regression_split["X_train"].copy()
    X_inf[0, 0] = np.inf
    X_inf[1, 1] = -np.inf
    reg = PytorchLightningRegressor(**_regressor_params())
    with pytest.raises(ValueError, match=r"inf"):
        reg.fit(X_inf, clean_regression_split["y_train"])


def test_regressor_rejects_nan_in_eval_set(clean_regression_split):
    """F-23: eval_set must also be validated -- pre-fix the val
    DataLoader silently propagated NaN."""
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        clean_regression_split["X_train"],
        clean_regression_split["y_train"],
        test_size=0.3,
        random_state=0,
    )
    rng = np.random.default_rng(0)
    X_val_nan = _inject_nan_into_features(X_val, rng)

    reg = PytorchLightningRegressor(**_regressor_params())
    with pytest.raises(ValueError, match=r"X_val.*NaN|NaN.*X_val|contains \d+ NaN"):
        reg.fit(X_tr, y_tr, eval_set=(X_val_nan, y_val))


def test_regressor_accepts_clean_int_features(clean_regression_split):
    """F-23 sanity: int-dtype features are valid (int arrays can't carry
    NaN); the validator must NOT raise on them."""
    X_int = clean_regression_split["X_train"].astype(np.int64)
    reg = PytorchLightningRegressor(**_regressor_params())
    # Must not raise; produces some prediction.
    reg.fit(X_int, clean_regression_split["y_train"])
    preds = reg.predict(clean_regression_split["X_test"].astype(np.int64))
    assert preds.shape[0] == clean_regression_split["X_test"].shape[0]
