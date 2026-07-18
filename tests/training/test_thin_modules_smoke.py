"""E-P1.4: smoke-import tests for thin-coverage modules.

Pure import-time checks - confirms the modules load on a stock dev env and
expose the public symbols other code reaches for. Failures here flag a
regression before any feature test touches them.
"""

from __future__ import annotations

import importlib

import pytest

THIN_MODULES = [
    "mlframe.training.quantile_postproc",
    "mlframe.training.configs",
    "mlframe.calibration.post",
]


@pytest.mark.fast
@pytest.mark.parametrize("module_name", THIN_MODULES)
def test_module_importable(module_name: str) -> None:
    """Module importable."""
    mod = importlib.import_module(module_name)
    assert mod is not None
    # Every loaded module exposes at least one public symbol.
    public = [n for n in dir(mod) if not n.startswith("_")]
    assert public, f"{module_name} exposes no public symbols"
