"""Keras/TensorFlow compatibility wrappers.

TensorFlow is an optional dependency. This module imports cleanly without it,
but any instantiation of `build_keras_mlp` or `KerasCompatibleMLP` raises
ImportError directing the user either to install tensorflow+scikeras or to use
the PyTorch-Lightning-based `mlframe.training.neural.flat.generate_mlp`.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

_HAS_TF = importlib.util.find_spec("tensorflow") is not None

_INSTALL_MSG = (
    "Install tensorflow and scikeras for Keras compatibility; " "or use mlframe.training.neural.flat.generate_mlp for the PyTorch Lightning equivalent."
)


def build_keras_mlp(
    num_layers: int = 3,
    num_neurons: int = 100,
    activation: str = "relu",
    regularizer_l2: float = 0.001,
    dropout_rate: float = 0.3,
    loss: str = "mean_squared_logarithmic_error",
    input_dim: Optional[int] = None,
):
    """Build a Sequential Dense+BatchNorm+Dropout stack (regressor)."""
    if not _HAS_TF:
        raise ImportError(_INSTALL_MSG)

    try:
        from tensorflow.keras import regularizers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
    except ModuleNotFoundError:
        # TF 2.16+ split Keras back out into its own top-level package.
        from keras import regularizers
        from keras.models import Sequential
        from keras.layers import Dense, BatchNormalization, Dropout, Input

    reg = regularizers.l2(regularizer_l2)
    layers = []
    if input_dim is not None:
        layers.append(Input(shape=(input_dim,)))
    layers.append(
        Dense(
            num_neurons,
            kernel_initializer="normal",
            activation=activation,
            kernel_regularizer=reg,
            bias_regularizer=reg,
            activity_regularizer=reg,
        )
    )
    layers.append(BatchNormalization())
    layers.append(Dropout(dropout_rate))
    for _ in range(num_layers):
        layers.append(
            Dense(
                num_neurons,
                kernel_initializer="normal",
                activation=activation,
                kernel_regularizer=reg,
                bias_regularizer=reg,
                activity_regularizer=reg,
            )
        )
        layers.append(BatchNormalization())
        layers.append(Dropout(dropout_rate))
    layers.append(
        Dense(
            1,
            kernel_initializer="normal",
            activation="linear",
            kernel_regularizer=reg,
            bias_regularizer=reg,
            activity_regularizer=reg,
        )
    )

    model = Sequential(layers)
    model.compile(loss=loss, optimizer="adam")
    return model


class KerasCompatibleMLP(BaseEstimator, RegressorMixin):
    """sklearn wrapper over a Keras Sequential MLP."""

    def __init__(
        self,
        num_layers: int = 3,
        num_neurons: int = 100,
        activation: str = "relu",
        regularizer_l2: float = 0.001,
        dropout_rate: float = 0.3,
        loss: str = "mean_squared_logarithmic_error",
        epochs: int = 200,
        batch_size: int = 400,
        validation_split: float = 0.1,
        verbose: int = 0,
    ):
        # The _HAS_TF check belongs in fit(), not here: sklearn constructs estimators freely (clone) and must never raise just to instantiate.
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation
        self.regularizer_l2 = regularizer_l2
        self.dropout_rate = dropout_rate
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.verbose = verbose

    def __getstate__(self) -> dict:
        # A live Keras ``Sequential`` is not reliably dill/joblib-picklable (it
        # carries TF graph + optimizer state that the generic pickle path mangles).
        # Serialise it structurally via get_config() + get_weights() and rebuild in
        # __setstate__ so a fitted KerasCompatibleMLP round-trips with identical predictions.
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            state["_keras_config_"] = model.get_config()
            state["_keras_weights_"] = model.get_weights()
        return state

    def __setstate__(self, state: dict) -> None:
        config = state.pop("_keras_config_", None)
        weights = state.pop("_keras_weights_", None)
        self.__dict__.update(state)
        if config is None:
            self.model_ = None
            return
        try:
            from tensorflow.keras.models import Sequential
        except ModuleNotFoundError:
            from keras.models import Sequential
        self.model_ = Sequential.from_config(config)
        self.model_.set_weights(weights)

    def fit(self, X, y):
        if not _HAS_TF:
            raise ImportError(_INSTALL_MSG)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.model_ = build_keras_mlp(
            num_layers=self.num_layers,
            num_neurons=self.num_neurons,
            activation=self.activation,
            regularizer_l2=self.regularizer_l2,
            dropout_rate=self.dropout_rate,
            loss=self.loss,
            input_dim=X.shape[1],
        )
        self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        if getattr(self, "model_", None) is None:
            from sklearn.exceptions import NotFittedError
            raise NotFittedError("KerasCompatibleMLP has not been fitted yet.")
        assert self.model_ is not None
        X = np.asarray(X, dtype=np.float32)
        preds = self.model_.predict(X, verbose=0)
        return preds.ravel()


__all__ = ["build_keras_mlp", "KerasCompatibleMLP"]
