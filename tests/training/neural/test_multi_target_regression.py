"""Iter-4: multi-target regression with PytorchLightningRegressor.

sklearn convention: a regressor that accepts y of shape (N, K) for K
target columns must:
  * train K output heads sharing the trunk
  * predict() returns (N, K) -- one prediction per target

Current code at base.py:394 hardcodes ``num_classes = 1`` for any
regressor. The network has output_dim=1; MSE between (N, 1)
predictions and (N, K) labels broadcasts catastrophically (torch
right-aligns shapes, treating (N, K) as (N, K) and (N, 1) as (N, 1),
broadcasting to (N, K) -- producing K copies of the same loss per
sample, ignoring K-1 of the target columns).

This test PROBES current behaviour. Expected: either crashes with
shape mismatch OR silently degenerates to predicting a constant /
ignoring K-1 target columns.

If the test surfaces a bug, the fix is to detect (N, K>=2) y in
_fit_common and route num_classes -> K (with regression metric
applied per-column).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
)


def _make_multi_target_data(n=240, d=5, k=3, seed=0):
    """y_k = linear(X) with K different coefficient vectors. A real
    multi-target regressor should learn all K functions."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    coefs = rng.normal(size=(d, k)).astype(np.float32)
    y = (X @ coefs + 0.05 * rng.normal(size=(n, k))).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed)
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def _regressor_params():
    """Regressor params."""
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.MSELoss(), "learning_rate": 5e-3},
        "network_params": {
            "nlayers": 2,
            "first_layer_num_neurons": 32,
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
            "max_epochs": 20,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
        "random_state": 0,
    }


def test_multi_target_regression_k3_native_support():
    """F-24 (2026-05-31): PytorchLightningRegressor natively supports
    multi-target y of shape (N, K>=2). Detection lives in _fit_common's
    regressor branch -> num_classes = K -> network outputs K heads
    sharing the trunk; MSE between (N, K) preds and (N, K) labels
    works without loss-shape gymnastics.

    Strong assertions:
      * fit() completes without raising
      * predict() returns shape exactly (N, K)
      * per-column R^2 > 0.5 (network actually learned each target)
      * the estimator's ``_is_multi_target_regression`` flag is True
    """
    data = _make_multi_target_data(k=3)
    reg = PytorchLightningRegressor(**_regressor_params())
    reg.fit(data["X_train"], data["y_train"])

    assert getattr(reg, "_is_multi_target_regression", False) is True

    preds = reg.predict(data["X_test"])
    assert preds.shape == data["y_test"].shape

    per_col_r2 = [r2_score(data["y_test"][:, k], preds[:, k]) for k in range(preds.shape[1])]
    print(f"\nF-24 multi-target (K=3) per-column R^2: {per_col_r2}")
    assert (
        min(per_col_r2) > 0.5
    ), f"per-column R^2 = {per_col_r2}; multi-target MLP did not learn each target. Expected per-column R^2 > 0.5 on this clean linear synthetic problem."


def test_multi_target_regression_k2_native_support():
    """Same property at K=2 (smallest non-trivial multi-target)."""
    data = _make_multi_target_data(k=2)
    reg = PytorchLightningRegressor(**_regressor_params())
    reg.fit(data["X_train"], data["y_train"])
    assert reg._is_multi_target_regression is True
    preds = reg.predict(data["X_test"])
    assert preds.shape == (data["X_test"].shape[0], 2)
    per_col_r2 = [r2_score(data["y_test"][:, k], preds[:, k]) for k in range(2)]
    print(f"\nF-24 multi-target (K=2) per-column R^2: {per_col_r2}")
    assert min(per_col_r2) > 0.5


def test_multi_target_predict_proba_does_not_apply():
    """Sanity: multi-target REGRESSOR doesn't have predict_proba."""
    data = _make_multi_target_data(k=3)
    reg = PytorchLightningRegressor(**_regressor_params())
    reg.fit(data["X_train"], data["y_train"])
    assert not hasattr(reg, "predict_proba")


def test_single_target_regression_with_2d_y_n_1_still_works():
    """y of shape (N, 1) is a SINGLE-target regression delivered as a
    2-D frame (common pandas pattern). Must work end-to-end."""
    n, d = 160, 4
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y_2d = (X @ rng.normal(size=d) + 0.05 * rng.normal(size=n)).astype(np.float32).reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_2d, test_size=0.3, random_state=0)

    reg = PytorchLightningRegressor(**_regressor_params())
    reg.fit(X_tr, y_tr)
    preds = reg.predict(X_te)
    r2 = r2_score(y_te.ravel(), preds.ravel())
    print(f"\n(N, 1) single-target regression: R^2 = {r2:+.4f}")
    assert r2 > 0.85
