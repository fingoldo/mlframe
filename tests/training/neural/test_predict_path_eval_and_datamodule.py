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


@pytest.mark.fast
def test_prediction_datamodule_stashed_after_fit():
    """After fit, self.prediction_datamodule is set so predict reuses it
    instead of falling through the "No datamodule found" warning branch."""
    from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor

    np.random.seed(0)
    X = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)

    est = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"input_size": 4, "hidden_sizes": [4], "output_size": 1},
        trainer_params={"max_epochs": 1, "enable_progress_bar": False, "enable_checkpointing": False, "logger": False, "accelerator": "cpu"},
        datamodule_params={"batch_size": 16, "num_workers": 0},
    )
    est.fit(X, y, eval_set=(X, y))

    assert hasattr(est, "prediction_datamodule"), "fit() must stash prediction_datamodule on self"
    assert est.prediction_datamodule is not None


@pytest.mark.fast
def test_predict_does_not_warn_about_missing_datamodule(caplog):
    from mlframe.training.neural import MLPTorchModel, PytorchLightningRegressor

    np.random.seed(0)
    X = np.random.randn(64, 4).astype(np.float32)
    y = np.random.randn(64).astype(np.float32)

    est = PytorchLightningRegressor(
        model_class=MLPTorchModel,
        model_params={"input_size": 4, "hidden_sizes": [4], "output_size": 1},
        trainer_params={"max_epochs": 1, "enable_progress_bar": False, "enable_checkpointing": False, "logger": False, "accelerator": "cpu"},
        datamodule_params={"batch_size": 16, "num_workers": 0},
    )
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
