"""The four CS-3 promotions must stay reachable from their owning packages' public surfaces.

Each symbol below was previously only importable from a private module (or as a private name), forcing
``mlframe.benchmarking`` into a cross-package underscore import that
``test_no_underscore_imports_cross_package.py`` forbids.
"""

from __future__ import annotations

import importlib

import pytest

CASES = [
    ("mlframe.feature_selection", "extract_selected"),
    ("mlframe.feature_selection", "support_mask_from_selector"),
    ("mlframe.core.set_similarity", "kuncheva"),
    ("mlframe.training", "stratified_split"),
    ("mlframe.training", "stratified_split_3way"),
    ("mlframe.training.composite.discovery", "benjamini_hochberg_reject"),
    ("mlframe.training.composite.discovery", "benjamini_yekutieli_reject"),
    ("mlframe.training.composite.discovery", "bootstrap_gain_p_value"),
]


@pytest.mark.parametrize("module_name, symbol", CASES)
def test_promoted_symbol_is_public(module_name, symbol):
    """Each promoted helper must be reachable under a public name, so no caller needs a private-module import."""
    module = importlib.import_module(module_name)
    assert callable(getattr(module, symbol))


@pytest.mark.parametrize(
    "module_name, symbol",
    [c for c in CASES if c[0] in ("mlframe.feature_selection", "mlframe.training", "mlframe.core.set_similarity")],
)
def test_promoted_symbol_is_in_all(module_name, symbol):
    """Reachability is not enough: vulture only treats a re-export as used when it is listed in ``__all__``."""
    module = importlib.import_module(module_name)
    assert symbol in module.__all__
