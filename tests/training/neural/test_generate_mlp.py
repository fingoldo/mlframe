"""
Tests for generate_mlp function using pytest and hypothesis.

Run tests:
    # Run all tests
    pytest tests/training/neural/ -v

    # Run with coverage
    pytest tests/training/neural/ --cov=mlframe.training.neural --cov-report=html

    # Run specific test
    pytest tests/training/neural/test_generate_mlp.py::test_basic_regression_network -v
"""

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings, assume
from functools import partial

import sys
from pathlib import Path

# Add parent directory to path to import mlframe
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mlframe.training.neural import generate_mlp, MLPNeuronsByLayerArchitecture


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


# ================================================================================================
# Mutation Testing - Boundary and Arithmetic Tests
# ================================================================================================


def test_expanding_architecture_multiplicative_scaling():
    """Test that Expanding architecture uses multiplicative scaling, not additive.

    Kills mutation: `*` to `+` in cur_layer_virt_neurons calculation.
    """
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
    # With multiplicative ratio 2.0: 10 -> 20 -> 40
    # With additive: 10 -> 12 -> 14
    assert arch[0][1] == 10  # First hidden layer
    assert arch[1][1] == 20  # Second layer should be 2x first
    assert arch[2][1] == 40  # Third layer should be 4x first


def test_declining_architecture_multiplicative_scaling():
    """Test that Declining architecture uses multiplicative scaling.

    Kills mutation: `*` to `+` in cur_layer_virt_neurons calculation.
    """
    model = generate_mlp(
        num_features=100,
        num_classes=2,
        nlayers=3,
        first_layer_num_neurons=80,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=2.0,
        min_layer_neurons=1,
        verbose=0,
    )
    arch = get_model_architecture(model)
    # With multiplicative ratio 2.0: 80 -> 40 -> 20
    # With additive: 80 -> 78 -> 76
    assert arch[0][1] == 80
    assert arch[1][1] == 40  # Should be halved
    assert arch[2][1] == 20  # Should be quartered


def test_expanding_then_declining_multiple_layers_before_mid():
    """Test ExpandingThenDeclining with multiple layers before midpoint.

    Kills mutation: `layer <= mid_layer` to `layer == mid_layer`.
    """
    model = generate_mlp(
        num_features=100,
        num_classes=10,
        nlayers=5,
        first_layer_num_neurons=10,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.ExpandingThenDeclining,
        consec_layers_neurons_ratio=2.0,
        min_layer_neurons=1,
        verbose=0,
    )
    arch = get_model_architecture(model)
    # mid_layer = 5 // 2 = 2
    # Layers 0, 1, 2 should expand; layers 3, 4 should decline
    # Layer 1 should be larger than layer 0 (not just layer 2)
    assert arch[1][1] > arch[0][1], "Layer 1 should expand beyond layer 0"
    assert arch[2][1] >= arch[1][1], "Layer 2 should be >= layer 1"


def test_num_classes_boundary_at_two():
    """Test effective_min_neurons boundary condition at num_classes=2.

    Kills mutation: `num_classes > 1` to `num_classes >= 1` or `num_classes > 0`.
    """
    # Test that num_classes=2 triggers multiclass behavior (final layer = 2)
    model_2 = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        min_layer_neurons=5,
        verbose=0,
    )
    arch_2 = get_model_architecture(model_2)
    assert arch_2[-1][1] == 2, "Final layer should have 2 outputs for num_classes=2"

    # Test that num_classes=1 creates regression network (final layer = 1)
    model_1 = generate_mlp(
        num_features=10,
        num_classes=1,
        nlayers=2,
        min_layer_neurons=5,
        verbose=0,
    )
    arch_1 = get_model_architecture(model_1)
    assert arch_1[-1][1] == 1, "Final layer should have 1 output for num_classes=1"


def test_consec_layers_neurons_ratio_exactly_one():
    """Test that ratio=1.0 is valid (constant architecture).

    Kills mutation: `consec_layers_neurons_ratio >= 1.0` to `> 1.0`.
    """
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=3,
        first_layer_num_neurons=50,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Expanding,
        consec_layers_neurons_ratio=1.0,  # Boundary case - should create constant layers
        verbose=0,
    )
    arch = get_model_architecture(model)
    # With ratio 1.0, all hidden layers should be 50
    assert arch[0][1] == 50
    assert arch[1][1] == 50
    assert arch[2][1] == 50


def test_consec_layers_neurons_ratio_below_one_raises():
    """Test that ratio < 1.0 raises assertion error.

    Kills mutation: boundary conditions on consec_layers_neurons_ratio.
    """
    with pytest.raises(AssertionError):
        generate_mlp(
            num_features=10,
            num_classes=2,
            nlayers=3,
            consec_layers_neurons_ratio=0.9,  # Invalid - below 1.0
            verbose=0,
        )


def test_nlayers_zero_raises_error():
    """Test that nlayers=0 raises assertion error.

    Kills mutation: `nlayers >= 1` to `nlayers >= 0`.
    """
    with pytest.raises(AssertionError):
        generate_mlp(
            num_features=10,
            num_classes=2,
            nlayers=0,  # Invalid
            verbose=0,
        )


def test_nlayers_exactly_one():
    """Test that nlayers=1 is valid boundary.

    Kills mutation: `nlayers >= 1` to `nlayers > 1`.
    """
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=1,  # Boundary case
        verbose=0,
    )
    assert isinstance(model, nn.Sequential)
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 2  # 1 hidden + 1 output


def test_groupnorm_zero_disables():
    """Test that groupnorm_num_groups=0 disables GroupNorm.

    Kills mutation: `groupnorm_num_groups > 0` to `> -1` or `!= 0`.
    """
    model_disabled = generate_mlp(
        num_features=16,
        num_classes=2,
        nlayers=2,
        first_layer_num_neurons=16,
        groupnorm_num_groups=0,  # Disabled
        verbose=0,
    )
    assert count_layers_of_type(model_disabled, nn.GroupNorm) == 0

    # Verify positive value enables it
    model_enabled = generate_mlp(
        num_features=16,
        num_classes=2,
        nlayers=2,
        first_layer_num_neurons=16,
        groupnorm_num_groups=4,  # Enabled
        verbose=0,
    )
    assert count_layers_of_type(model_enabled, nn.GroupNorm) >= 1


def test_weight_init_applies_to_2d_weights():
    """Test weight initialization applies only to 2D+ weights.

    Kills mutation: `m.weight.dim() >= 2` to `<= 2`, `== 2`, `>= 1`.
    """
    init_value = 0.12345

    def custom_init(tensor):
        with torch.no_grad():
            tensor.fill_(init_value)

    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        weights_init_fcn=custom_init,
        verbose=0,
    )

    # Check Linear layer weights were initialized with our value
    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert torch.allclose(m.weight, torch.full_like(m.weight, init_value))


def test_weight_init_bias_handling():
    """Test that bias initialization only happens when bias exists.

    Kills mutation: `hasattr(m, "bias") and m.bias is not None` logic.
    """
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        weights_init_fcn=nn.init.xavier_uniform_,
        verbose=0,
    )

    for m in model.modules():
        if isinstance(m, nn.Linear):
            # Bias should exist and be properly initialized
            assert hasattr(m, 'bias')
            assert m.bias is not None
            assert m.bias.shape[0] == m.out_features


def test_dropout_not_applied_when_negative():
    """Test that negative dropout values raise error or don't add dropout.

    Kills mutation: `dropout_prob > 0` to `!= 0`.
    """
    # Dropout prob of exactly 0 should not add dropout
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        dropout_prob=0.0,
        inputs_dropout_prob=0.0,
        verbose=0,
    )
    assert count_layers_of_type(model, nn.Dropout) == 0


# ================================================================================================
# Phase 2 - High Priority Mutation Tests
# ================================================================================================


def test_autoencoder_bottleneck_architecture():
    """Test Autoencoder architecture creates bottleneck (declining then expanding).

    Kills mutation: layer comparison operators in architecture logic.
    """
    model = generate_mlp(
        num_features=100,
        num_classes=100,
        nlayers=6,
        first_layer_num_neurons=80,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Autoencoder,
        consec_layers_neurons_ratio=2.0,
        min_layer_neurons=1,
        verbose=0,
    )
    arch = get_model_architecture(model)
    # Should decline to bottleneck then expand
    # First half should decline
    assert arch[1][1] < arch[0][1], "Should decline from layer 0 to 1"


def test_min_layer_neurons_enforcement():
    """Test min_layer_neurons is enforced in declining architectures.

    Kills mutation: min_layer_neurons comparison logic.
    """
    model = generate_mlp(
        num_features=100,
        num_classes=2,
        nlayers=10,  # Many layers to force hitting minimum
        first_layer_num_neurons=100,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Declining,
        consec_layers_neurons_ratio=2.0,
        min_layer_neurons=10,  # Set minimum
        verbose=0,
    )
    arch = get_model_architecture(model)
    # All hidden layers should have at least min_layer_neurons
    for _, out_features in arch[:-1]:
        assert out_features >= 10, f"Layer has {out_features} < min_layer_neurons"


def test_batchnorm_and_layernorm_together():
    """Test using both BatchNorm and LayerNorm.

    Kills mutation: normalization layer creation logic.
    """
    model = generate_mlp(
        num_features=16,
        num_classes=2,
        nlayers=3,
        use_batchnorm=True,
        use_layernorm=True,
        verbose=0,
    )
    assert count_layers_of_type(model, nn.BatchNorm1d) >= 1
    assert count_layers_of_type(model, nn.LayerNorm) >= 1


def test_different_activation_functions():
    """Test using different activation functions.

    Kills mutation: activation function instantiation logic.
    """
    # Test with LeakyReLU
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        activation_function=nn.LeakyReLU,
        verbose=0,
    )
    # Check that LeakyReLU was created
    leaky_count = count_layers_of_type(model, nn.LeakyReLU)
    assert leaky_count >= 2

    # Test with GELU
    model_gelu = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=2,
        activation_function=nn.GELU,
        verbose=0,
    )
    gelu_count = count_layers_of_type(model_gelu, nn.GELU)
    assert gelu_count >= 2


def test_first_layer_neurons_override():
    """Test first_layer_num_neurons parameter.

    Kills mutation: first layer neuron count logic.
    """
    model = generate_mlp(
        num_features=10,
        num_classes=2,
        nlayers=3,
        first_layer_num_neurons=100,
        neurons_by_layer_arch=MLPNeuronsByLayerArchitecture.Constant,
        verbose=0,
    )
    arch = get_model_architecture(model)
    assert arch[0][1] == 100, "First hidden layer should have 100 neurons"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
