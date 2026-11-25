"""Cluster 39 regression: the assembler docstring names a real svd default (audit 2026-11-20, finding 39-F31).

The docstring cited ``FeatureHandlingConfig.svd_globals.default_dim``, which does not exist. It now names the ``svd_default_dim`` argument, which
defaults to 256 in the assembler.
"""

import inspect

from mlframe.training.feature_handling import assembler


def test_docstring_names_the_real_argument_not_a_config_field():
    """The module docstring points at svd_default_dim and says there is no svd_globals field (39-F31)."""
    doc = assembler.__doc__ or ""
    assert "svd_globals.default_dim" not in doc
    assert "svd_default_dim" in doc
    assert "no ``svd_globals`` config field" in doc


def test_assembler_default_is_256():
    """Control: the argument the docstring names defaults to 256 in the assembler (39-F31)."""
    sig = inspect.signature(assembler.assemble_for_model)
    assert sig.parameters["svd_default_dim"].default == 256
