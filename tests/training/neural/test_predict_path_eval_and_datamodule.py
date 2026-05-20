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
        MLPTorchModel, PytorchLightningRegressor, TorchDataModule,
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
    """After fit, self.prediction_datamodule is set so predict reuses it
    instead of falling through the "No datamodule found" warning branch."""
    np.random.seed(0)
    X = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)

    est = _make_regressor()
    est.fit(X, y, eval_set=(X, y))

    assert hasattr(est, "prediction_datamodule"), "fit() must stash prediction_datamodule on self"
    assert est.prediction_datamodule is not None


@pytest.mark.fast
def test_predict_does_not_warn_about_missing_datamodule(caplog):
    np.random.seed(0)
    X = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)

    est = _make_regressor()
    est.fit(X, y, eval_set=(X, y))

    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural.base"):
        _ = est.predict(X)

    spurious_msgs = [
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        and (
            "No datamodule found" in r.getMessage()
            or "training mode during prediction" in r.getMessage()
        )
    ]
    assert not spurious_msgs, (
        f"Predict-after-fit should not emit these warnings. Got: {spurious_msgs}"
    )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "--no-cov", "-x", "-s", "--tb=short"]))
