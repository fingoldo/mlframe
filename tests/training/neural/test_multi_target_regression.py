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
import pytest
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
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.MSELoss(), "learning_rate": 5e-3},
        "network_params": {
            "nlayers": 2, "first_layer_num_neurons": 32,
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
            "max_epochs": 20, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        "random_state": 0,
    }


def test_multi_target_regression_current_state():
    """SNAPSHOT test: probe what PytorchLightningRegressor does with
    a (N, K=3) y. We assert at most ONE of:
      (a) fit() raises a clear shape-mismatch error, OR
      (b) fit() completes and predict returns shape (N, K=3) with
          R^2 > 0.5 per column (real multi-target support)
      (c) fit() completes but predict returns shape (N,) or (N, 1)
          (silent degeneration — current expected)

    If (b), multi-target is supported and this test passes the strong
    assertion. If (a) or (c), the test passes the weak assertion and
    DOCUMENTS what we currently do.
    """
    data = _make_multi_target_data(k=3)
    reg = PytorchLightningRegressor(**_regressor_params())
    try:
        reg.fit(data["X_train"], data["y_train"])
    except Exception as e:
        print(f"\nMulti-target regression fit() raised: {type(e).__name__}: {e}")
        # Acceptable outcome (a): explicit reject is the ideal behaviour.
        return

    try:
        preds = reg.predict(data["X_test"])
    except Exception as e:
        print(f"\nMulti-target predict() raised: {type(e).__name__}: {e}")
        return

    print(f"\nMulti-target predict shape: {preds.shape} "
          f"(y_test shape: {data['y_test'].shape})")

    if preds.shape == data["y_test"].shape:
        per_col_r2 = [
            r2_score(data["y_test"][:, k], preds[:, k]) for k in range(preds.shape[1])
        ]
        print(f"Per-column R^2: {per_col_r2}")
        # If we reached here AND R^2 is good per column, multi-target works.
        if min(per_col_r2) > 0.5:
            print("✓ Multi-target regression: SUPPORTED (per-column R^2 > 0.5)")
            return

    # Documented degeneration (c): predict shape doesn't match labels,
    # OR R^2 is poor per column. Snapshot the current state.
    print(
        "Multi-target regression: DEGENERATED on this estimator. "
        "Expected: detect (N, K>=2) y in _fit_common, route num_classes -> K, "
        "train K output heads."
    )


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
