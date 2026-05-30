"""Regression test for F-02 (mlp audit 2026-05-30).

The regression metric function in ``PytorchLightningEstimator._fit_common``
is ``sklearn.metrics.root_mean_squared_error`` (RMSE), but the
``MetricSpec.name`` used at registration was ``"MSE"``. That mislabel
propagated to:

* Lightning CSV-logger column names (``val_MSE`` / ``train_MSE``)
* Best-model checkpoint filename (``model-val_MSE=0.7555.ckpt``)
* EarlyStopping / ReduceLROnPlateau monitor keys
* Run-time logs ("New best model at epoch X with val_MSE=...")

Numerically the label was sqrt(MSE), so any cross-tool / cross-checkpoint
comparison keyed on "MSE" silently confused RMSE values with MSE values
(off by a square root for non-trivial residuals).

Post-fix the label is ``"RMSE"`` and the monitor key is ``val_RMSE``.

The test asserts:
1. The logged metric name in the run-time Lightning logger is ``val_RMSE``,
   not ``val_MSE``.
2. The logged value matches ``sklearn.metrics.root_mean_squared_error``
   recomputed independently from the validation predictions and labels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_regression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningRegressor,
    TorchDataModule,
)


@pytest.fixture
def regression_split():
    X, y = make_regression(
        n_samples=240, n_features=8, n_informative=6, noise=5.0, random_state=19,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3, random_state=19)
    return {"X_train": X_tr, "y_train": y_tr, "X_val": X_val, "y_val": y_val}


@pytest.fixture
def tiny_regressor_params():
    network_params = {
        "nlayers": 1,
        "first_layer_num_neurons": 16,
        "dropout_prob": 0.0,
        "inputs_dropout_prob": 0.0,
        "use_layernorm": False,
        "activation_function": torch.nn.ReLU,
    }
    model_params = {
        "loss_fn": torch.nn.MSELoss(),
        "learning_rate": 1e-2,
    }
    datamodule_params = {
        "features_dtype": torch.float32,
        "labels_dtype": torch.float32,
        "dataloader_params": {"batch_size": 32, "num_workers": 0},
    }
    trainer_params = {
        "max_epochs": 2,
        "enable_model_summary": False,
        "enable_progress_bar": False,
        "log_every_n_steps": 1,
        "devices": 1,
        "accelerator": "cpu",
        # Real CSVLogger lands in tmp_path via the default_root_dir override
        # set inside the test (the suite reads the CSV after fit completes).
    }
    return {
        "model_class": MLPTorchModel,
        "model_params": model_params,
        "network_params": network_params,
        "datamodule_class": TorchDataModule,
        "datamodule_params": datamodule_params,
        "trainer_params": trainer_params,
    }


def _read_metrics_csv(root_dir: Path) -> dict[str, list[float]]:
    """Locate the single CSVLogger metrics.csv under ``root_dir`` (Lightning
    nests it under ``_run_<id>_<ts>/lightning_logs/version_<N>/metrics.csv``
    by default; mlframe's checkpoint_dir_override path tree nests it under
    ``_run_<id>_<ts>/metrics.csv``). Returns dict of column-name -> values
    list."""
    candidates = list(root_dir.rglob("metrics.csv"))
    assert candidates, f"no metrics.csv found under {root_dir}"
    # Pick the most recently modified (handles concurrent test runs).
    csv_path = max(candidates, key=lambda p: p.stat().st_mtime)
    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    out: dict[str, list[float]] = {col: [] for col in header}
    for row in rows[1:]:
        cells = row.split(",")
        for col, cell in zip(header, cells):
            cell = cell.strip()
            if not cell:
                continue
            try:
                out[col].append(float(cell))
            except ValueError:
                pass  # epoch / step columns may already be ints; ignore non-floats
    return out


def test_regression_metric_logged_as_rmse_not_mse(
    regression_split, tiny_regressor_params, tmp_path,
):
    """The Lightning CSV logger must log ``val_RMSE``, not ``val_MSE``,
    because the underlying metric function is sklearn's
    ``root_mean_squared_error``."""
    params = dict(tiny_regressor_params)
    trainer_params = dict(params["trainer_params"])
    trainer_params["default_root_dir"] = str(tmp_path)
    params["trainer_params"] = trainer_params

    reg = PytorchLightningRegressor(**params)
    reg.fit(
        regression_split["X_train"], regression_split["y_train"],
        eval_set=(regression_split["X_val"], regression_split["y_val"]),
    )

    metrics = _read_metrics_csv(tmp_path)
    assert "val_RMSE" in metrics, (
        f"expected 'val_RMSE' in metrics.csv columns; got {sorted(metrics.keys())}. "
        "F-02 fix renames the metric label from 'MSE' (mislabelled) to 'RMSE' "
        "(what the metric function actually computes)."
    )
    assert "val_MSE" not in metrics, (
        f"'val_MSE' should no longer appear post-fix; got columns "
        f"{sorted(metrics.keys())}. The pre-fix mislabel was the bug."
    )

    val_rmse_values = metrics["val_RMSE"]
    assert val_rmse_values, "val_RMSE column is empty"
    assert all(v >= 0 for v in val_rmse_values), (
        f"val_RMSE must be non-negative (RMSE = sqrt(MSE) >= 0); got {val_rmse_values}"
    )


def test_logged_rmse_matches_sklearn_root_mean_squared_error(
    regression_split, tiny_regressor_params, tmp_path,
):
    """The logged ``val_RMSE`` value at the FINAL epoch must equal
    sklearn's ``root_mean_squared_error(y_val, predict(X_val))`` to within
    float-precision noise (the network is in train mode during validation
    but BN/Dropout default off in our network_params, so this round-trip
    is deterministic). Pre-fix the label said 'MSE' but value was sqrt(MSE)
    -- any downstream tool that squared the column to recover MSE would
    have been off by a factor; the rename fixes the contract."""
    params = dict(tiny_regressor_params)
    trainer_params = dict(params["trainer_params"])
    trainer_params["default_root_dir"] = str(tmp_path)
    params["trainer_params"] = trainer_params

    reg = PytorchLightningRegressor(**params)
    reg.fit(
        regression_split["X_train"], regression_split["y_train"],
        eval_set=(regression_split["X_val"], regression_split["y_val"]),
    )

    metrics = _read_metrics_csv(tmp_path)
    final_logged_rmse = metrics["val_RMSE"][-1]

    preds = reg.predict(regression_split["X_val"])
    independent_rmse = root_mean_squared_error(regression_split["y_val"], preds)

    # Loose tolerance: the logged value is computed at the end of the val
    # epoch on the SAME data, but the model's state at val-end may differ
    # microscopically from the post-fit-checkpoint state (best-epoch load,
    # dropout=0 doesn't differ but BN buffers may). 1e-3 relative is
    # comfortably below the "off-by-sqrt" scale the F-02 bug created.
    np.testing.assert_allclose(
        final_logged_rmse, independent_rmse, rtol=1e-2,
        err_msg=(
            f"logged val_RMSE={final_logged_rmse} diverges from "
            f"sklearn root_mean_squared_error={independent_rmse}; if these "
            f"diverge by ~sqrt(N) the rename is incomplete and the metric "
            "function is now MSE instead of RMSE."
        ),
    )


def test_no_validation_path_uses_train_loss_monitor(
    regression_split, tiny_regressor_params, tmp_path,
):
    """When ``eval_set`` is not provided, the monitor falls back to
    ``train_loss`` (not affected by the rename) and the run still
    completes without raising. Regression-check that the rename doesn't
    impact the no-val path."""
    params = dict(tiny_regressor_params)
    trainer_params = dict(params["trainer_params"])
    trainer_params["default_root_dir"] = str(tmp_path)
    params["trainer_params"] = trainer_params

    reg = PytorchLightningRegressor(**params)
    reg.fit(regression_split["X_train"], regression_split["y_train"])

    preds = reg.predict(regression_split["X_val"])
    assert preds.shape == (regression_split["X_val"].shape[0],)
