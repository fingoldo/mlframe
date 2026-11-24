"""Cluster 11 regression: a broken catboost install is reported at WARNING, a missing one at INFO (audit 2026-11-20, finding 11-F8).

Both used to log at INFO, so a broken install looked the same as "not installed". The two cases now log at different levels, and both return False.
"""

import builtins
import logging

from mlframe.feature_selection._benchmarks.fs_hybrid import _arms_external as arms


def _patched_import(broken_exc):
    """An import hook that raises ``broken_exc`` for catboost and defers everything else."""
    real = builtins.__import__

    def _imp(name, *args, **kwargs):
        """Raise for catboost; delegate other imports."""
        if name == "catboost":
            raise broken_exc
        return real(name, *args, **kwargs)

    return _imp


def test_broken_install_logs_warning(monkeypatch, caplog):
    """A catboost import that raises a non-ImportError is a WARNING and returns False (11-F8)."""
    monkeypatch.setattr(builtins, "__import__", _patched_import(RuntimeError("broken wheel")))
    with caplog.at_level(logging.INFO):
        assert arms.catboost_available() is False
    assert any(r.levelno == logging.WARNING and "installed but fails to import" in r.getMessage() for r in caplog.records)


def test_missing_install_logs_info_not_warning(monkeypatch, caplog):
    """An ImportError (not installed) is INFO, not WARNING (11-F8)."""
    monkeypatch.setattr(builtins, "__import__", _patched_import(ImportError("No module named catboost")))
    with caplog.at_level(logging.INFO):
        assert arms.catboost_available() is False
    assert any(r.levelno == logging.INFO and "not installed" in r.getMessage() for r in caplog.records)
    assert not any(r.levelno == logging.WARNING and "catboost" in r.getMessage() for r in caplog.records)
