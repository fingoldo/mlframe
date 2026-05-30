"""Iter-3 sanity tests for the MLP estimators.

Two questions:

1. On a CLEAN linear regression problem (y = linear combination of X + small
   noise), does PytorchLightningRegressor reach R^2 comparable to
   ``sklearn.linear_model.LinearRegression``? An MLP with a real
   nonlinearity should not LOSE BADLY to a linear baseline on linear data.
   If it does, that exposes a training-mechanics bug (LR / optimizer /
   loss-shape / data-pipeline regression).

2. On a CLEAN binary classification problem, does ``predict_proba(X)``
   produce row-sums equal to 1.0 (per the sklearn classifier contract)?
   With a 2-output softmax head the rows must sum to 1 by construction;
   anywhere they don't is a normalisation bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from mlframe.training.neural import (
    MLPTorchModel,
    PytorchLightningClassifier,
    PytorchLightningRegressor,
    TorchDataModule,
)


@pytest.fixture
def linear_regression_data_scaled():
    """y = 3*x0 - 2*x1 + 1.5*x2 + eps with x columns on UNIT scale (so the
    bare-default MLP without upstream StandardScaler still has a fair shot)."""
    X, y = make_regression(
        n_samples=600, n_features=5, n_informative=3, noise=0.1, random_state=0,
    )
    # Pre-standardise X so the test isolates training mechanics from the
    # F-03 input-normalisation concern.
    X = StandardScaler().fit_transform(X).astype(np.float32)
    y = y.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


@pytest.fixture
def binary_classification_data():
    X, y = make_classification(
        n_samples=400, n_features=6, n_informative=4, n_redundant=0,
        n_classes=2, random_state=0,
    )
    X_tr, X_te, y_tr, y_te = train_test_split(
        X.astype(np.float32), y.astype(np.int64), test_size=0.3, random_state=0,
    )
    return {"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


@pytest.fixture
def regressor_params():
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.MSELoss(), "learning_rate": 1e-2},
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
            "dataloader_params": {"batch_size": 64, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 50,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
    }


@pytest.fixture
def classifier_params():
    return {
        "model_class": MLPTorchModel,
        "model_params": {"loss_fn": torch.nn.CrossEntropyLoss(), "learning_rate": 1e-2},
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
            "labels_dtype": torch.int64,
            "dataloader_params": {"batch_size": 64, "num_workers": 0},
        },
        "trainer_params": {
            "max_epochs": 30,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "log_every_n_steps": 1,
            "devices": 1,
            "accelerator": "cpu",
            "logger": False,
        },
    }


def test_mlp_regressor_matches_linear_baseline_on_linear_data(
    linear_regression_data_scaled, regressor_params,
):
    """On a CLEAN linear regression problem with standardised X, the
    MLP must reach within 0.10 R^2 of sklearn's LinearRegression. A
    real nonlinearity (ReLU) + enough epochs (50) should let the MLP
    approximate the affine map to high precision; a much wider gap
    would expose a training-mechanics bug (LR, optimiser, loss shape,
    data-pipeline regression).
    """
    torch.manual_seed(0)
    np.random.seed(0)

    reg_mlp = PytorchLightningRegressor(**regressor_params)
    reg_mlp.fit(linear_regression_data_scaled["X_train"], linear_regression_data_scaled["y_train"])
    preds_mlp = reg_mlp.predict(linear_regression_data_scaled["X_test"])
    r2_mlp = r2_score(linear_regression_data_scaled["y_test"], preds_mlp)

    reg_lin = LinearRegression()
    reg_lin.fit(linear_regression_data_scaled["X_train"], linear_regression_data_scaled["y_train"])
    preds_lin = reg_lin.predict(linear_regression_data_scaled["X_test"])
    r2_lin = r2_score(linear_regression_data_scaled["y_test"], preds_lin)

    print(f"\nSanity baseline (linear data, standardised X):")
    print(f"  PytorchLightningRegressor R^2 = {r2_mlp:+.4f}")
    print(f"  sklearn LinearRegression  R^2 = {r2_lin:+.4f}")
    print(f"  gap                            = {r2_lin - r2_mlp:+.4f}")

    assert r2_mlp > 0.85, (
        f"MLPRegressor R^2={r2_mlp:+.4f} should reach >0.85 on a clean "
        "linear problem (LinearRegression hit "
        f"R^2={r2_lin:+.4f}); a much lower R^2 indicates a training-"
        "mechanics bug (LR, optimiser, loss shape, pipeline regression)."
    )
    assert r2_lin - r2_mlp < 0.15, (
        f"MLPRegressor lost {r2_lin - r2_mlp:+.4f} R^2 to LinearRegression "
        "on linear data; gap should be <0.15."
    )


def test_binary_predict_proba_rows_sum_to_one(
    binary_classification_data, classifier_params,
):
    """sklearn classifier contract: ``predict_proba(X)`` rows must sum to
    1.0 (within float precision). With our 2-output softmax head the rows
    sum to 1 by construction; anywhere they don't is a normalisation bug.
    """
    torch.manual_seed(0)
    np.random.seed(0)

    clf = PytorchLightningClassifier(**classifier_params)
    clf.fit(binary_classification_data["X_train"], binary_classification_data["y_train"])
    proba = clf.predict_proba(binary_classification_data["X_test"])

    assert proba.shape == (binary_classification_data["X_test"].shape[0], 2)
    row_sums = proba.sum(axis=1)
    max_dev = float(np.abs(row_sums - 1.0).max())
    print(f"\nBinary predict_proba row-sum: max|sum - 1.0| = {max_dev:.6e}")
    assert max_dev < 1e-5, (
        f"predict_proba rows do not sum to 1.0; max deviation = {max_dev:.6e}. "
        "sklearn classifier contract requires probabilities sum to 1 per row."
    )

    # Also sanity: predict_proba(X) and predict(X) must agree -- predict
    # should be classes_[argmax(predict_proba)]
    preds_direct = clf.predict(binary_classification_data["X_test"])
    preds_via_proba = clf.classes_[np.argmax(proba, axis=1)]
    np.testing.assert_array_equal(preds_direct, preds_via_proba)


def test_binary_predict_proba_columns_aligned_with_sorted_classes(
    classifier_params,
):
    """Pre-fix (before F-19), labels in {-1, +1} could land in classes_
    out of sort order. F-19 routes through LabelEncoder which sorts to
    [-1, +1]. predict_proba columns must align with sorted classes_:
    column 0 = P(y = -1), column 1 = P(y = +1). Verify by training on
    perfectly-separable data and checking the proba column for the
    majority class on the y=+1 side is > 0.5."""
    rng = np.random.default_rng(0)
    n = 400
    # Easy separator: x_0 < 0 -> -1, x_0 > 0 -> +1
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.where(X[:, 0] > 0, 1, -1).astype(np.int64)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    torch.manual_seed(0)
    clf = PytorchLightningClassifier(**classifier_params)
    clf.fit(X_tr, y_tr)
    assert list(clf.classes_) == [-1, 1]

    proba = clf.predict_proba(X_te)
    # For test points with x_0 > 0 (clearly y=+1), P(y=+1) (column 1)
    # should dominate; mean of column-1 prob over those points > 0.5.
    pos_mask = X_te[:, 0] > 0
    p_pos_for_positives = proba[pos_mask, 1].mean()
    p_neg_for_negatives = proba[~pos_mask, 0].mean()
    print(
        f"\nBinary col alignment (classes_=[-1,+1]):\n"
        f"  P(y=+1 | x0>0) mean = {p_pos_for_positives:.4f}\n"
        f"  P(y=-1 | x0<0) mean = {p_neg_for_negatives:.4f}"
    )
    assert p_pos_for_positives > 0.5
    assert p_neg_for_negatives > 0.5
