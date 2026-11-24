"""Cluster 39 regressions: a module logger is named after its own module, and a dead `pass` branch is gone (audit 2026-11-20, findings 39-F11 and 39-F12).

The tuned-threshold report used to log under another module's name, so per-module log levels had no effect. The extractor base carried an
`if ...: pass` branch with a stale comment; it has been deleted.
"""

from pathlib import Path

from mlframe.training.core import _tuned_threshold_report as tr
from mlframe.training.extractors import _extractors_base as eb


def test_tuned_threshold_report_logger_uses_its_own_name():
    """The module logger is named after this module, not another one (39-F12)."""
    assert tr.logger.name == tr.__name__


def test_extractor_base_has_no_dead_clean_ram_pass():
    """The stale `if ...: pass  # clean_ram()` branch is removed (39-F11)."""
    src = Path(eb.__file__).read_text(encoding="utf-8")
    assert "pass  # clean_ram()" not in src
