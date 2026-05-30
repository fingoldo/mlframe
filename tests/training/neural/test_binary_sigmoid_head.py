"""F-05 regression: PytorchLightningClassifier uses 1-output sigmoid +
BCEWithLogitsLoss for binary classification, not 2-output softmax + CE.

Architectural assertions:
  * For 2-class fits: ``_binary_sigmoid_head=True``, network output_dim=1,
    LightningModule's ``task_type=="binary"``, ``loss_fn`` is BCE.
  * For K>=3 class fits: ``_binary_sigmoid_head=False``, network
    output_dim=K, ``task_type`` stays None, ``loss_fn`` stays CE.
  * predict_proba shape stays (N, 2) for binary (sklearn-canonical),
    rows sum to 1, columns aligned with ``classes_``.
  * predict still returns ENTRIES of ``classes_`` (F-01 preserved).

Per the user's "ignore legacy" instruction, the prior 2-output softmax
binary path is gone -- no opt-in flag, no back-compat shim. Existing
checkpoints from the prior architecture cannot be loaded (state_dict
shape mismatch).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningClassifier,
    TorchDataModule,
)


def _params(loss_fn=None, random_state=0):
    return {
        "model_class": MLPTorchModel,
        "model_params": {
            "loss_fn": loss_fn if loss_fn is not None else torch.nn.CrossEntropyLoss(),
            "learning_rate": 1e-2,
        },
        "network_params": {
            "nlayers": 1, "first_layer_num_neurons": 16,
            "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
            "use_layernorm": False, "use_batchnorm": False,
            "activation_function": torch.nn.ReLU,
        },
        "datamodule_class": TorchDataModule,
        "datamodule_params": {
            "features_dtype": torch.float32,
            "labels_dtype": torch.int64,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 2, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        "random_state": random_state,
    }


def _last_linear_out_features(network) -> int:
    """Walk a Sequential network and return the out_features of the LAST nn.Linear."""
    last = None
    for module in network.modules():
        if isinstance(module, nn.Linear):
            last = module
    assert last is not None, "no nn.Linear found in network"
    return last.out_features


@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=160, n_features=5, n_informative=4, n_redundant=0,
        n_classes=2, random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.astype(np.float32), y.astype(np.int64), test_size=0.3, random_state=0,
    )
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


@pytest.fixture
def multiclass_data():
    X, y = make_classification(
        n_samples=180, n_features=6, n_informative=5, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.astype(np.float32), y.astype(np.int64), test_size=0.3, random_state=0,
    )
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def test_binary_fit_sets_sigmoid_head_attributes(binary_data):
    """Binary fit ⇒ ``_binary_sigmoid_head=True``, network out_dim=1,
    LightningModule's ``task_type=="binary"``, loss is BCE."""
    clf = PytorchLightningClassifier(**_params())
    clf.fit(binary_data["X_train"], binary_data["y_train"])

    assert clf._binary_sigmoid_head is True
    assert _last_linear_out_features(clf.network) == 1
    assert clf.model.task_type == "binary"
    assert isinstance(clf.model.loss_fn, nn.BCEWithLogitsLoss)


def test_multiclass_fit_keeps_softmax_head(multiclass_data):
    """3-class fit ⇒ ``_binary_sigmoid_head=False``, network out_dim=K,
    ``task_type`` remains None (multiclass softmax path), loss stays CE."""
    clf = PytorchLightningClassifier(**_params())
    clf.fit(multiclass_data["X_train"], multiclass_data["y_train"])

    assert clf._binary_sigmoid_head is False
    assert _last_linear_out_features(clf.network) == 3
    assert clf.model.task_type is None
    assert isinstance(clf.model.loss_fn, nn.CrossEntropyLoss)


def test_binary_predict_proba_shape_and_normalisation(binary_data):
    """predict_proba returns (N, 2), rows sum to 1.0, columns aligned
    with sorted ``classes_``."""
    clf = PytorchLightningClassifier(**_params())
    clf.fit(binary_data["X_train"], binary_data["y_train"])
    proba = clf.predict_proba(binary_data["X_test"])
    assert proba.shape == (binary_data["X_test"].shape[0], 2)
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5)
    # Column 0 + column 1 = 1; non-negative; column ordering matches sorted classes_.
    assert (proba >= 0).all()
    assert (proba <= 1).all()
    assert list(clf.classes_) == [0, 1]


def test_binary_predict_returns_class_labels(binary_data):
    """predict() still returns ENTRIES of classes_ (F-01 preserved
    through the F-05 refactor). For y in {0, 1} that's still {0, 1};
    for y in {10, 20} it would be {10, 20}."""
    # Use non-dense labels to make the assertion meaningful.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64) * 10 + 10  # 10 or 20
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    clf = PytorchLightningClassifier(**_params())
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_te)
    assert set(np.asarray(preds).tolist()).issubset({10, 20})


def test_binary_predict_proba_matches_sigmoid_p_eq_1_minus_p(binary_data):
    """Column 0 must EXACTLY equal 1 - column 1 (no floating-point drift
    from independent computations). This is a stricter check than the
    row-sum assertion."""
    clf = PytorchLightningClassifier(**_params())
    clf.fit(binary_data["X_train"], binary_data["y_train"])
    proba = clf.predict_proba(binary_data["X_test"])
    np.testing.assert_allclose(proba[:, 0], 1.0 - proba[:, 1], rtol=0, atol=1e-7)
