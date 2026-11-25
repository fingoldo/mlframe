"""Cluster 40 regression: the recurrent modules carry no dead xxhash probe or empty TYPE_CHECKING block (audit 2026-11-20, finding 40-F32).

The probe was never read in these modules (the cache-key path has its own), and the TYPE_CHECKING block was empty. Both are removed, and the
modules still import.
"""

import importlib

import pytest


@pytest.mark.parametrize("module", ["mlframe.training.neural._recurrent_torch_model", "mlframe.training.neural.recurrent"])
def test_module_imports_without_the_dead_probe(module):
    """The module imports and carries no _HAS_XXHASH flag (40-F32)."""
    mod = importlib.import_module(module)
    assert not hasattr(mod, "_HAS_XXHASH")
