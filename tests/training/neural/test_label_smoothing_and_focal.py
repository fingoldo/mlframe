"""C5: label smoothing (multiclass only) + focal loss (binary, opt-in)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel, PytorchLightningClassifier, TorchDataModule,
)


def _params(label_smoothing=0.0, focal_loss_gamma=None, focal_loss_alpha=0.25,
            random_state=0):
    return {
        "model_class": MLPTorchModel,
        "model_params": {
            "loss_fn": nn.CrossEntropyLoss(), "learning_rate": 1e-2,
        },
        "network_params": {
            "nlayers": 1, "first_layer_num_neurons": 16,
            "dropout_prob": 0.0, "inputs_dropout_prob": 0.0,
            "use_layernorm": False, "use_batchnorm": False,
            "activation_function": nn.ReLU,
        },
        "datamodule_class": TorchDataModule,
        "datamodule_params": {
            "features_dtype": torch.float32, "labels_dtype": torch.int64,
            "dataloader_params": {"batch_size": 32, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 3, "enable_model_summary": False,
            "enable_progress_bar": False, "log_every_n_steps": 1,
            "devices": 1, "accelerator": "cpu", "logger": False,
        },
        "label_smoothing": label_smoothing,
        "focal_loss_gamma": focal_loss_gamma,
        "focal_loss_alpha": focal_loss_alpha,
        "random_state": random_state,
    }


@pytest.fixture
def imbalanced_binary():
    X, y = make_classification(
        n_samples=600, n_features=6, n_informative=5, n_redundant=0,
        n_classes=2, weights=[0.95, 0.05], random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.astype(np.float32), y.astype(np.int64),
        test_size=0.3, random_state=0, stratify=y,
    )
    return X_tr, X_te, y_tr, y_te


@pytest.fixture
def multiclass3():
    X, y = make_classification(
        n_samples=300, n_features=6, n_informative=5, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.astype(np.float32), y.astype(np.int64),
        test_size=0.3, random_state=0,
    )
    return X_tr, X_te, y_tr, y_te


def test_multiclass_label_smoothing_replaces_loss(multiclass3):
    """label_smoothing > 0 on multiclass injects a fresh CrossEntropyLoss
    with the requested epsilon."""
    X_tr, X_te, y_tr, _ = multiclass3
    clf = PytorchLightningClassifier(**_params(label_smoothing=0.1))
    clf.fit(X_tr, y_tr)
    # Inspect the LightningModule's loss to verify it carries the smoothing.
    loss = clf.model.loss_fn
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert float(loss.label_smoothing) == 0.1


def test_multiclass_no_label_smoothing_uses_user_loss(multiclass3):
    """label_smoothing=0 (default) leaves the caller's CrossEntropyLoss
    untouched (no smoothing injected)."""
    X_tr, _, y_tr, _ = multiclass3
    clf = PytorchLightningClassifier(**_params(label_smoothing=0.0))
    clf.fit(X_tr, y_tr)
    loss = clf.model.loss_fn
    assert isinstance(loss, nn.CrossEntropyLoss)
    assert float(loss.label_smoothing) == 0.0


def test_binary_label_smoothing_does_not_override_bce(imbalanced_binary):
    """label_smoothing on a BINARY classifier must NOT replace the BCE
    loss (Cattan 2024 shows LS regresses binary calibration; smoothing
    is gated to multiclass only)."""
    X_tr, _, y_tr, _ = imbalanced_binary
    clf = PytorchLightningClassifier(**_params(label_smoothing=0.1))
    clf.fit(X_tr, y_tr)
    # Binary path: loss must remain BCEWithLogitsLoss (NOT CrossEntropyLoss).
    loss = clf.model.loss_fn
    assert isinstance(loss, nn.BCEWithLogitsLoss)


def test_binary_focal_loss_completes_fit(imbalanced_binary):
    """focal_loss_gamma=2.0 on binary replaces BCE with sigmoid focal loss."""
    X_tr, X_te, y_tr, _ = imbalanced_binary
    clf = PytorchLightningClassifier(**_params(focal_loss_gamma=2.0))
    clf.fit(X_tr, y_tr)
    # Loss should be a callable (not a torch.nn module), produced by
    # _make_binary_focal_loss.
    loss = clf.model.loss_fn
    assert callable(loss)
    assert not isinstance(loss, nn.BCEWithLogitsLoss)
    preds = clf.predict(X_te)
    assert set(np.asarray(preds).tolist()).issubset({0, 1})


def test_binary_focal_loss_differs_from_bce(imbalanced_binary):
    """Focal-trained and BCE-trained models on the same imbalanced
    binary problem should produce DIFFERENT predictions (focal weights
    hard examples differently). If they're equal the focal kernel is
    silently degenerating to BCE."""
    X_tr, X_te, y_tr, _ = imbalanced_binary
    clf_bce = PytorchLightningClassifier(**_params(focal_loss_gamma=None))
    clf_bce.fit(X_tr, y_tr)
    preds_bce = clf_bce.predict_proba(X_te)

    clf_focal = PytorchLightningClassifier(**_params(focal_loss_gamma=2.0, focal_loss_alpha=0.25))
    clf_focal.fit(X_tr, y_tr)
    preds_focal = clf_focal.predict_proba(X_te)

    assert not np.allclose(preds_bce, preds_focal, atol=1e-3), (
        "Focal-trained predictions equal BCE-trained predictions; focal "
        "kernel may have silently degenerated to BCE."
    )


def test_focal_loss_alpha_default_does_not_crash(imbalanced_binary):
    """focal_loss_alpha=-1.0 disables the alpha weighting (per torchvision
    convention). Verify it doesn't crash."""
    X_tr, _, y_tr, _ = imbalanced_binary
    clf = PytorchLightningClassifier(**_params(focal_loss_gamma=2.0, focal_loss_alpha=-1.0))
    clf.fit(X_tr, y_tr)
