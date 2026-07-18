"""Regression: predict() after fit() should NOT emit
1. "No datamodule found from training. Creating temporary datamodule for prediction."
2. "Model was in training mode during prediction. Switching to eval mode."

Both were spurious - the fit-time datamodule wasn't stashed (#1) and eval()
wasn't called unconditionally (#2). Fixed by storing dm on self and calling
.eval() always (cheap idempotent op).
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

import numpy as np


def _make_regressor():
    """Build a minimal PytorchLightningRegressor with the now-required
    network_params + datamodule_class kwargs.

    Earlier versions of the estimator accepted only model_class / model_params /
    trainer_params / datamodule_params; the audit-wave constructor split out
    ``network_params`` (MLP topology) and ``datamodule_class`` so the data and
    network surfaces are explicit and clone-able. Tests using the old 4-arg
    form crash with ``TypeError: missing 2 required positional arguments``
    at construction time.
    """
    import torch
    from mlframe.training.neural import (
        MLPTorchModel,
        PytorchLightningRegressor,
        TorchDataModule,
    )

    return PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-3},
        network_params={"nlayers": 2},
        datamodule_class=TorchDataModule,
        datamodule_params={
            "read_fcn": None,
            "data_placement_device": None,
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        trainer_params={"max_epochs": 1, "logger": False, "accelerator": "cpu", "devices": 1},
    )


@pytest.mark.fast
def test_prediction_datamodule_stashed_after_fit():
    """After fit, the lightweight datamodule SHELL is retained on self
    so predict can reuse it (no spurious "No datamodule found" WARNING)
    -- AND the heavy train/val feature/label/sample_weight tensors are
    nulled INSIDE the datamodule so they don't get pickled into the
    save() bundle (~1.5 GB on prod-shape 4M x 323 float32 frame).
    """
    np.random.seed(0)
    X = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)

    est = _make_regressor()
    est.fit(X, y, eval_set=(X, y))

    # Shell stays (lightweight config + class refs, ~few KB).
    assert hasattr(est, "prediction_datamodule"), "fit() must stash the datamodule shell"
    assert (
        est.prediction_datamodule is not None
    ), "fit-end memory-safety must keep the lightweight datamodule shell so predict can reuse it; only the heavy tensors are nulled"
    # Heavy tensors inside the shell are nulled.
    _dm = est.prediction_datamodule
    for _attr in ("train_features", "train_labels", "train_sample_weight", "val_features", "val_labels"):
        if hasattr(_dm, _attr):
            assert getattr(_dm, _attr) is None, f"fit-end memory-safety must NULL {_attr} on the datamodule so save() doesn't pickle the train/val tensors"
    # The marker tells predict() that the null state is intentional.
    assert getattr(est, "_datamodule_tensors_dropped", False) is True


@pytest.mark.fast
def test_predict_does_not_warn_about_missing_datamodule(caplog):
    """predict-after-fit must NOT emit the "No datamodule found" WARNING:
    the fit-end memory-safety pass nulls the heavy tensors INSIDE the
    datamodule but keeps the shell, so predict() takes the reuse branch
    (no temporary-datamodule warning). The WARNING is reserved for the
    actual user-error path (estimator constructed without fit).
    """
    np.random.seed(0)
    X = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)

    est = _make_regressor()
    est.fit(X, y, eval_set=(X, y))

    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.base"):
        _ = est.predict(X)

    spurious_msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and ("No datamodule found" in r.getMessage() or "training mode during prediction" in r.getMessage())
    ]
    assert not spurious_msgs, f"Predict-after-fit should not emit these warnings. Got: {spurious_msgs}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "--no-cov", "-x", "-s", "--tb=short"]))
