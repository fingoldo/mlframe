"""F-06 regression: ``random_state`` parameter on PytorchLightningEstimator
makes fits reproducible.

Pre-fix two ``fit()`` calls on the same data produced different
predictions because nothing seeded torch / numpy / DataLoader. Suite-
level callers usually owned reproducibility; direct-API users had no
canonical knob (sklearn convention is ``random_state``).

Post-fix:
  * Same ``random_state`` + same data -> bit-identical predictions
  * Different ``random_state`` + same data -> different predictions
    (the seed actually does something)
  * ``random_state=None`` (default) keeps the pre-fix non-deterministic
    behaviour: two fits give different predictions
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


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, noise=1.0, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.astype(np.float32), y.astype(np.float32), test_size=0.3, random_state=42,
    )
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te}


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=4, n_redundant=0,
        n_classes=2, random_state=42,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.astype(np.float32), y.astype(np.int64), test_size=0.3, random_state=42,
    )
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te}


def _regressor_params(random_state=None):
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-2},
        "network_params": {
            "nlayers": 1, "first_layer_num_neurons": 16,
            "dropout_prob": 0.1,  # nonzero so dropout RNG is exercised
            "inputs_dropout_prob": 0.0, "use_layernorm": False,
            "use_batchnorm": False, "activation_function": torch.nn.ReLU,
        },
        "datamodule_class": TorchDataModule,
        "datamodule_params": {
            "features_dtype": torch.float32,
            "labels_dtype": torch.float32,
            "dataloader_params": {"batch_size": 16, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 3, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        "random_state": random_state,
    }


def _classifier_params(random_state=None):
    p = _regressor_params(random_state=random_state)
    p["model_params"] = {"loss_fn": torch.nn.CrossEntropyLoss(), "learning_rate": 1e-2}
    p["datamodule_params"]["labels_dtype"] = torch.int64
    return p


def test_regressor_same_random_state_yields_identical_predictions(regression_data):
    """Two PytorchLightningRegressor fits with the same random_state on
    the same data MUST produce bit-identical predictions."""
    r1 = PytorchLightningRegressor(**_regressor_params(random_state=12345))
    r1.fit(regression_data["X_train"], regression_data["y_train"])
    p1 = r1.predict(regression_data["X_test"])

    r2 = PytorchLightningRegressor(**_regressor_params(random_state=12345))
    r2.fit(regression_data["X_train"], regression_data["y_train"])
    p2 = r2.predict(regression_data["X_test"])

    np.testing.assert_array_equal(p1, p2)


def test_classifier_same_random_state_yields_identical_predictions(binary_data):
    """Same property for the classifier path (different random ops:
    network init + dataloader shuffle + dropout; the label-encoding
    side is deterministic by construction)."""
    c1 = PytorchLightningClassifier(**_classifier_params(random_state=999))
    c1.fit(binary_data["X_train"], binary_data["y_train"])
    proba1 = c1.predict_proba(binary_data["X_test"])

    c2 = PytorchLightningClassifier(**_classifier_params(random_state=999))
    c2.fit(binary_data["X_train"], binary_data["y_train"])
    proba2 = c2.predict_proba(binary_data["X_test"])

    np.testing.assert_array_equal(proba1, proba2)


def test_regressor_different_random_state_yields_different_predictions(regression_data):
    """Two fits with DIFFERENT seeds must produce DIFFERENT predictions
    (otherwise the seed had no effect). Use a small max|diff| > 0 check;
    even a 1-epoch shallow MLP with different inits ends up at different
    points so any nonzero diff confirms the seed plumbing works."""
    r1 = PytorchLightningRegressor(**_regressor_params(random_state=1))
    r1.fit(regression_data["X_train"], regression_data["y_train"])
    p1 = r1.predict(regression_data["X_test"])

    r2 = PytorchLightningRegressor(**_regressor_params(random_state=2))
    r2.fit(regression_data["X_train"], regression_data["y_train"])
    p2 = r2.predict(regression_data["X_test"])

    diff = float(np.abs(p1 - p2).max())
    assert diff > 1e-4, (
        f"different random_state should produce different predictions; "
        f"got max|diff|={diff:.2e}"
    )


def test_random_state_none_preserves_legacy_nondeterministic_behaviour(regression_data):
    """When random_state is None (default), two fits should produce
    DIFFERENT predictions -- the pre-fix non-deterministic behaviour
    is preserved for callers who manage their own seed externally.
    This pins the contract so the default flip doesn't silently impose
    determinism on existing pipelines."""
    r1 = PytorchLightningRegressor(**_regressor_params(random_state=None))
    r1.fit(regression_data["X_train"], regression_data["y_train"])
    p1 = r1.predict(regression_data["X_test"])

    r2 = PytorchLightningRegressor(**_regressor_params(random_state=None))
    r2.fit(regression_data["X_train"], regression_data["y_train"])
    p2 = r2.predict(regression_data["X_test"])

    diff = float(np.abs(p1 - p2).max())
    assert diff > 1e-4, (
        f"random_state=None should preserve non-deterministic behaviour; "
        f"got max|diff|={diff:.2e} (suspiciously low -- something seeded "
        "the run without an explicit random_state)."
    )
