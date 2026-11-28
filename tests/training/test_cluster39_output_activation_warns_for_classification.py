"""Cluster 39 regression: a non-linear output_activation on a classification head is reported, not silently ignored (audit 2026-11-20, finding 39-F61)."""

import logging

from mlframe.training.neural._flat_generate_helpers import _generate_mlp_affine_composition_extrapolation_motivated as apply_activation


def test_classification_head_with_tanh_train_range_logs_a_warning(caplog):
    """num_classes=3 with tanh_train_range warns and adds no output layer; regression still raises when scale/center are missing."""
    layers = []
    with caplog.at_level(logging.WARNING, logger="mlframe.training.neural._flat_generate_helpers"):
        apply_activation("tanh_train_range", 3, None, None, layers)
    assert layers == [] and any("ignored for num_classes=3" in r.getMessage() for r in caplog.records)
