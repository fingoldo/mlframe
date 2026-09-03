"""TRAINING_NEURAL-5 regression test: build_keras_mlp(num_layers=N) must build exactly N hidden Dense
layers, not N+1.

Pre-fix, the function built one unconditional Dense block before the loop, then ran the loop
``num_layers`` more times -- N+1 hidden layers total for a parameter the docstring/name imply means N.
Requires a real TensorFlow install (skipped otherwise, per CI's importorskip convention for optional
heavy deps); the fix itself is exercised in whichever CI lane has tensorflow installed.
"""

from __future__ import annotations

import pytest

tensorflow = pytest.importorskip("tensorflow")

from mlframe.training.neural.keras_compat import build_keras_mlp


def _count_hidden_dense_layers(model) -> int:
    """Count Dense layers excluding the final 1-unit output layer (linear activation, units=1)."""
    from tensorflow.keras.layers import Dense

    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]
    # Last Dense layer is the output head (units=1); every other Dense layer is a hidden layer.
    return len(dense_layers) - 1


@pytest.mark.parametrize("num_layers", [1, 2, 3, 5])
def test_build_keras_mlp_hidden_layer_count_matches_num_layers(num_layers):
    """The number of hidden Dense layers built must equal the num_layers parameter exactly."""
    model = build_keras_mlp(num_layers=num_layers, num_neurons=8, input_dim=4)
    assert _count_hidden_dense_layers(model) == num_layers
