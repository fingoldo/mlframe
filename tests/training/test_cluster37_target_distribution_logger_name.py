"""Cluster 37 regression: the target-distribution module logs under its own name (audit 2026-11-20, finding 37-F44).

The logger was named after the parent suite module, so per-module log levels set on this module had no effect. It now uses ``__name__``.
"""

from mlframe.training.core import _main_train_suite_target_distribution as tdist


def test_logger_is_named_after_this_module():
    """The module logger's name is the module's own dotted name (37-F44)."""
    assert tdist.logger.name == tdist.__name__
