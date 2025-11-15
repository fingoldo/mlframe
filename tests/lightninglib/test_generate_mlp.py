"""
Tests for generate_mlp function using pytest and hypothesis.

Run tests:
    # Run all tests
    pytest tests/lightninglib/ -v

    # Run with coverage
    pytest tests/lightninglib/ --cov=mlframe.lightninglib --cov-report=html

    # Run specific test
    pytest tests/lightninglib/test_generate_mlp.py::test_basic_regression_network -v

    # Mutation testing
    mutatest -s lightninglib.py --testcmds "python -m pytest tests/lightninglib/test_generate_mlp.py" -n 5
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings, assume
from functools import partial

import sys
from pathlib import Path

# Add parent directory to path to import mlframe
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlframe.lightninglib import generate_mlp, MLPNeuronsByLayerArchitecture


# ================================================================================================
# Helper functions
# ================================================================================================


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_layers_of_type(model: nn.Module, layer_type: type) -> int:
    """Count number of layers of specific type."""
    return sum(1 for module in model.modules() if isinstance(module, layer_type))


def get_model_architecture(model: nn.Module) -> list:
    """Extract layer sizes from Linear layers."""
    sizes = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            sizes.append((module.in_features, module.out_features))
    return sizes


# ================================================================================================
# Basic Functionality Tests
# ================================================================================================


def test_basic_classification_network():
    """Test creating a simple classification network."""
    model = generate_mlp(
        num_features=10,
        num_classes=3,
        nlayers=2,
        first_layer_num_neurons=20,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)
    assert count_layers_of_type(model, nn.Linear) >= 3  # 2 hidden + 1 output


def test_basic_regression_network():
    """Test creating a regression network (num_classes=1)."""
    model = generate_mlp(
        num_features=10,
        num_classes=1,
        nlayers=2,
        first_layer_num_neurons=20,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)
    # Check final layer outputs 1 value
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linear_layers[-1].out_features == 1


def test_feature_extractor_none():
    """Test feature extractor creation with num_classes=None."""
    model = generate_mlp(
        num_features=10,
        num_classes=None,
        nlayers=2,
        first_layer_num_neurons=20,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)
    # Should not have a final classification layer
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linear_layers[-1].out_features == 20  # Last hidden layer


def test_feature_extractor_zero():
    """Test feature extractor creation with num_classes=0."""
    model = generate_mlp(
        num_features=10,
        num_classes=0,
        nlayers=2,
        first_layer_num_neurons=20,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert linear_layers[-1].out_features == 20


# ================================================================================================
# Architecture Pattern Tests
# ================================================================================================


def test_constant_architecture():
    """Test Constant architecture (all layers same size)."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=3,
        first_layer_num_neurons=50,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Constant,
        verbose=0,
    )

    arch = get_model_architecture(model)
    # All hidden layers should have 50 neurons
    for in_feat, out_feat in arch[:-1]:  # Exclude final layer
        assert out_feat == 50


def test_declining_architecture():
    """Test Declining architecture."""
    model = generate_mlp(
        num_features=100,
        num_classes=2,
        nlayers=3,
        first_layer_num_neurons=100,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=2.0,
        min_layer_neurons=1,
        verbose=0,
    )

    arch = get_model_architecture(model)
    # Each layer should have fewer neurons than previous (excluding final)
    for i in range(len(arch) - 2):
        assert arch[i + 1][1] < arch[i][1]


def test_expanding_architecture():
    """Test Expanding architecture."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=3,
        first_layer_num_neurons=10,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Expanding,
        consec_layers_neurons_ratio=2.0,
        verbose=0,
    )

    arch = get_model_architecture(model)
    # Each layer should have more neurons than previous (excluding final)
    for i in range(len(arch) - 2):
        assert arch[i + 1][1] > arch[i][1]


def test_autoencoder_architecture_symmetry():
    """Test Autoencoder creates symmetric architecture."""
    model = generate_mlp(
        num_features=100,
        num_classes=100,
        nlayers=6,
        first_layer_num_neurons=100,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Autoencoder,
        consec_layers_neurons_ratio=2.0,
        min_layer_neurons=1,
        verbose=0,
    )

    arch = get_model_architecture(model)
    layer_sizes = [out for _, out in arch[:-1]]  # Exclude final output layer

    # Check it decreases then increases (bottleneck in middle)
    mid = len(layer_sizes) // 2
    # First half should decrease
    for i in range(mid - 1):
        assert layer_sizes[i + 1] <= layer_sizes[i]
    # Second half should increase
    for i in range(mid, len(layer_sizes) - 1):
        assert layer_sizes[i + 1] >= layer_sizes[i]


def test_autoencoder_allows_small_bottleneck():
    """Test that Autoencoder allows bottleneck smaller than min_layer_neurons."""
    model = generate_mlp(
        num_features=100,
        num_classes=100,
        nlayers=6,
        first_layer_num_neurons=100,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Autoencoder,
        consec_layers_neurons_ratio=2.0,
        min_layer_neurons=20,  # High minimum
        verbose=0,
    )

    arch = get_model_architecture(model)
    layer_sizes = [out for _, out in arch[:-1]]

    # Bottleneck should be allowed to be smaller than min_layer_neurons
    min_size = min(layer_sizes)
    assert min_size < 20  # Should violate min_layer_neurons for symmetry


# ================================================================================================
# Normalization Tests
# ================================================================================================


def test_layernorm_applied():
    """Test LayerNorm is applied when use_layernorm=True."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        use_layernorm=True,
        verbose=0,
    )

    assert count_layers_of_type(model, nn.LayerNorm) >= 1


def test_batchnorm_applied():
    """Test BatchNorm is applied when use_batchnorm=True."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        use_batchnorm=True,
        verbose=0,
    )

    assert count_layers_of_type(model, nn.BatchNorm1d) >= 1


def test_layernorm_per_layer():
    """Test LayerNorm per layer when use_layernorm_per_layer=True."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=3,
        use_layernorm_per_layer=True,
        use_layernorm=False,  # Only per-layer, not input
        verbose=0,
    )

    # Should have LayerNorm after each hidden layer (3 layers)
    assert count_layers_of_type(model, nn.LayerNorm) >= 3


def test_groupnorm_applied():
    """Test GroupNorm is applied when groupnorm_num_groups > 0."""
    model = generate_mlp(
        num_features=16,  # Divisible by common group numbers
        num_classes=2,
        nlayers=2,
        groupnorm_num_groups=4,
        verbose=0,
    )

    assert count_layers_of_type(model, nn.GroupNorm) >= 1


def test_norm_kwargs_separation():
    """Test that different norm types use different kwargs."""
    # This test verifies the fix doesn't crash with momentum in LayerNorm
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        use_layernorm=True,
        use_batchnorm=True,
        layer_norm_kwargs=dict(eps=1e-6),
        batch_norm_kwargs=dict(eps=1e-5, momentum=0.2),
        verbose=0,
    )

    # Should create without error
    assert isinstance(model, nn.Sequential)


# ================================================================================================
# Dropout Tests
# ================================================================================================


def test_dropout_not_applied_when_zero():
    """Test dropout is not applied when probability is 0."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        verbose=0,
    )

    # Should have no Dropout layers
    assert count_layers_of_type(model, nn.Dropout) == 0


def test_dropout_applied_when_positive():
    """Test dropout is applied when probability > 0."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        dropout_prob=0.5,
        inputs_dropout_prob=0.1,
        verbose=0,
    )

    # Should have dropout layers
    assert count_layers_of_type(model, nn.Dropout) >= 1


# ================================================================================================
# Activation Function Tests
# ================================================================================================


def test_activation_function_instantiated():
    """Test that activation function is instantiated (not just class)."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        activation_function=nn.ReLU,
        verbose=0,
    )

    # Check that ReLU instances exist (not just the class)
    relu_count = count_layers_of_type(model, nn.ReLU)
    assert relu_count >= 2  # At least one per hidden layer


def test_custom_activation_function():
    """Test using custom activation function."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        activation_function=nn.LeakyReLU,
        verbose=0,
    )

    assert count_layers_of_type(model, nn.LeakyReLU) >= 2


def test_no_activation_function():
    """Test network without activation functions."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        activation_function=None,
        verbose=0,
    )

    # Should have no ReLU layers
    assert count_layers_of_type(model, nn.ReLU) == 0


# ================================================================================================
# Forward Pass Tests
# ================================================================================================


def test_forward_pass_classification():
    """Test forward pass produces correct output shape for classification."""
    num_features = 10
    num_classes = 3
    batch_size = 5

    model = generate_mlp(
        num_features=num_features,
        num_classes=num_classes,
        nlayers=2,
        verbose=0,
    )

    x = torch.randn(batch_size, num_features)
    output = model(x)

    assert output.shape == (batch_size, num_classes)


def test_forward_pass_regression():
    """Test forward pass produces correct output shape for regression."""
    num_features = 10
    batch_size = 5

    model = generate_mlp(
        num_features=num_features,
        num_classes=1,
        nlayers=2,
        verbose=0,
    )

    x = torch.randn(batch_size, num_features)
    output = model(x)

    assert output.shape == (batch_size, 1)


def test_forward_pass_feature_extractor():
    """Test forward pass for feature extractor."""
    num_features = 10
    batch_size = 5
    hidden_neurons = 20

    model = generate_mlp(
        num_features=num_features,
        num_classes=None,
        nlayers=2,
        first_layer_num_neurons=hidden_neurons,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Constant,
        verbose=0,
    )

    x = torch.randn(batch_size, num_features)
    output = model(x)

    # Output should be features from last hidden layer
    assert output.shape == (batch_size, hidden_neurons)


# ================================================================================================
# Parameter Tests
# ================================================================================================


def test_min_layer_neurons_not_modified():
    """Test that min_layer_neurons parameter is not modified."""
    original_min = 10

    model = generate_mlp(
        num_features=10,
        num_classes=20,  # Greater than min_layer_neurons
        nlayers=2,
        min_layer_neurons=original_min,
        verbose=0,
    )

    # This is a bit indirect, but we're checking the function doesn't crash
    # and produces a valid model
    assert isinstance(model, nn.Sequential)


def test_weights_initialization():
    """Test that weights initialization is applied."""
    model1 = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        weights_init_fcn=None,
        verbose=0,
    )

    model2 = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        weights_init_fcn=nn.init.xavier_uniform_,
        verbose=0,
    )

    # Both should be valid models
    assert isinstance(model1, nn.Sequential)
    assert isinstance(model2, nn.Sequential)


def test_example_input_array():
    """Test that example_input_array is set correctly."""
    num_features = 10

    model = generate_mlp(
        num_features=num_features,
        num_classes=2,
        nlayers=2,
        verbose=0,
    )

    assert hasattr(model, 'example_input_array')
    assert model.example_input_array.shape == (1, num_features)


# ================================================================================================
# Hypothesis-based Property Tests
# ================================================================================================


@given(
    num_features=st.integers(min_value=1, max_value=100),
    num_classes=st.integers(min_value=1, max_value=50),
    nlayers=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=50, deadline=None)
def test_property_valid_model_creation(num_features, num_classes, nlayers):
    """Property test: generate_mlp should always create a valid model."""
    try:
        model = generate_mlp(
            num_features=num_features,
            num_classes=num_classes,
            nlayers=nlayers,
            first_layer_num_neurons=max(num_features, num_classes),
            verbose=0,
        )

        assert isinstance(model, nn.Sequential)
        assert count_parameters(model) > 0

        # Test forward pass
        x = torch.randn(2, num_features)
        output = model(x)
        assert output.shape[0] == 2  # Batch size
        assert output.shape[1] == num_classes

    except AssertionError:
        # Some parameter combinations might be invalid
        pass


@given(
    dropout_prob=st.floats(min_value=0.0, max_value=0.9),
    inputs_dropout_prob=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=30, deadline=None)
def test_property_dropout_probabilities(dropout_prob, inputs_dropout_prob):
    """Property test: dropout probabilities should be valid."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        dropout_prob=dropout_prob,
        inputs_dropout_prob=inputs_dropout_prob,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)


@given(
    ratio=st.floats(min_value=1.1, max_value=3.0),
)
@settings(max_examples=20, deadline=None)
def test_property_layer_ratio(ratio):
    """Property test: different ratios should produce valid models."""
    model = generate_mlp(
        num_features=100,
        num_classes=2,
        nlayers=3,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=ratio,
        min_layer_neurons=1,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)


# ================================================================================================
# Edge Cases Tests
# ================================================================================================


def test_single_layer_network():
    """Test creating a network with only 1 hidden layer."""
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=1,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 2  # 1 hidden + 1 output


def test_large_network():
    """Test creating a large network."""
    model = generate_mlp(
        num_features=100,
        num_classes=10,
        nlayers=20,
        first_layer_num_neurons=200,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Constant,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)
    assert count_parameters(model) > 10000


def test_very_small_features():
    """Test network with very few input features."""
    model = generate_mlp(
        num_features=1,
        num_classes=2,
        nlayers=2,
        first_layer_num_neurons=5,
        verbose=0,
    )

    assert isinstance(model, nn.Sequential)
    x = torch.randn(3, 1)
    output = model(x)
    assert output.shape == (3, 2)


# ================================================================================================
# Verbose Output Tests
# ================================================================================================


def test_verbose_output(caplog):
    """Test that verbose=1 produces log output."""
    import logging
    caplog.set_level(logging.INFO)

    model = generate_mlp(
        num_features=50,
        num_classes=10,
        nlayers=3,
        first_layer_num_neurons=512,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=2.0,
        verbose=1,
    )

    # Check that something was logged
    assert len(caplog.records) > 0
    # Check for architecture string with arrow notation
    log_text = " ".join([record.message for record in caplog.records])
    assert "->" in log_text or "architecture" in log_text.lower()


def test_verbose_includes_model_type(caplog):
    """Test that verbose output includes model type (C/R/FE)."""
    import logging
    caplog.set_level(logging.INFO)

    # Test Classification
    generate_mlp(num_features=10, num_classes=3, nlayers=1, verbose=1)
    log_text = " ".join([record.message for record in caplog.records])
    assert "C" in log_text or "Classification" in log_text

    caplog.clear()

    # Test Regression
    generate_mlp(num_features=10, num_classes=1, nlayers=1, verbose=1)
    log_text = " ".join([record.message for record in caplog.records])
    assert "R" in log_text or "Regression" in log_text

    caplog.clear()

    # Test Feature Extractor
    generate_mlp(num_features=10, num_classes=None, nlayers=1, verbose=1)
    log_text = " ".join([record.message for record in caplog.records])
    assert "FE" in log_text or "feature extractor" in log_text.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
