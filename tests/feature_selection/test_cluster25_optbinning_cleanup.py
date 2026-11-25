"""Cluster 25 regression: the optbinning helper module drops its unused logger and stray module-level string (audit 2026-11-20, finding 25-F59).

Source-level pin: no logger is defined (nothing used it), and the misplaced docstring-like string after the imports is gone. The module still
imports.
"""

from pathlib import Path

from mlframe.feature_selection import optbinning as ob

_SRC = Path(ob.__file__).resolve()


def test_unused_logger_is_removed():
    """The module defines no logger, as nothing used one (25-F59)."""
    text = _SRC.read_text(encoding="utf-8")
    assert "logger = logging.getLogger" not in text
    assert "import logging" not in text


def test_stray_module_string_is_removed():
    """The misplaced categorical-FE string is no longer a module-level statement (25-F59)."""
    assert '"""Categorical feature engineering for ML.' not in _SRC.read_text(encoding="utf-8")
