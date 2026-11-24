"""Cluster 28 regression: the crash-heartbeat env parser is tested for unset, zero and invalid values (audit 2026-11-20, finding 28-F75).

heartbeat_interval_from_env reads MLFRAME_CRASH_HEARTBEAT_S: unset gives the default, "0" disables (0.0), and an unparsable value falls back to the
default with a WARNING. A negative value is clamped to 0.
"""

import logging

import pytest

from mlframe.training import crash_diagnostics as cd

_VAR = "MLFRAME_CRASH_HEARTBEAT_S"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """Start each test with the variable unset."""
    monkeypatch.delenv(_VAR, raising=False)


def test_unset_gives_the_default(monkeypatch):
    """Without the variable the default is returned (28-F75)."""
    assert cd.heartbeat_interval_from_env(5.0) == 5.0


def test_zero_disables(monkeypatch):
    """A value of 0 gives 0.0, which disables the heartbeat (28-F75)."""
    monkeypatch.setenv(_VAR, "0")
    assert cd.heartbeat_interval_from_env(5.0) == 0.0


def test_invalid_falls_back_to_default_with_warning(monkeypatch, caplog):
    """An unparsable value returns the default and logs a WARNING (28-F75)."""
    monkeypatch.setenv(_VAR, "abc")
    with caplog.at_level(logging.WARNING):
        assert cd.heartbeat_interval_from_env(5.0) == 5.0
    assert any(r.levelno == logging.WARNING and "fell back to its default" in r.getMessage() for r in caplog.records)


def test_negative_is_clamped_to_zero(monkeypatch):
    """A negative value is clamped to 0.0 (28-F75)."""
    monkeypatch.setenv(_VAR, "-3")
    assert cd.heartbeat_interval_from_env(5.0) == 0.0
