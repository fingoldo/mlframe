"""Cluster 11 regression: a counted estimator module that fails to import with a non-ImportError is listed as missing (audit 2026-11-20, finding 11-F7).

Only ImportError was caught, so any other import failure aborted every arm. The failure is now logged at WARNING and the module's classes are
reported as not instrumented.
"""

import importlib
import logging

from mlframe.feature_selection._benchmarks.fs_hybrid import _arms as arms


def test_non_import_error_is_reported_not_raised(monkeypatch, caplog):
    """A module whose import raises RuntimeError is listed as missing with a WARNING (11-F7)."""
    real = importlib.import_module
    target = arms._COUNTED_ESTIMATORS[0][0]

    def _fake(name, *args, **kwargs):
        """Raise RuntimeError for the chosen module, delegate otherwise."""
        if name == target:
            raise RuntimeError("broken install")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _fake)
    with caplog.at_level(logging.WARNING):
        _classes, missing = arms._resolve_counted_classes()
    assert any(m.startswith(target + ".") for m in missing)
    assert any(r.levelno == logging.WARNING and target in r.getMessage() for r in caplog.records)
