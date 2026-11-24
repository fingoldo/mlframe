"""Cluster 29 regression: the numba cache guard no longer defines the unused relative-import pattern (audit 2026-11-20, finding 29-F28).

Source-level pin: _RELATIVE_IMPORT is gone; the module still imports and its public guard remains.
"""

from pathlib import Path

from mlframe.utils import numba_cache_guard as ncg

_SRC = Path(ncg.__file__).resolve()


def test_unused_relative_import_pattern_is_removed():
    """The never-used _RELATIVE_IMPORT constant is deleted (29-F28)."""
    assert "_RELATIVE_IMPORT" not in _SRC.read_text(encoding="utf-8")


def test_module_still_imports():
    """Control: the module still loads and exposes its names (29-F28)."""
    assert ncg is not None
