"""Cluster 39 regression: MLFRAME_MEMORY_BUDGET_GB must be a positive finite number (audit 2026-11-20, finding 39-F52).

A value of 0, a negative value or a non-number used to be accepted (0 bytes reached downstream sizing). Such values are now ignored with a
WARNING and detection falls back to psutil/cgroup. A positive value is still honoured.
"""

import logging

import pytest

from mlframe.training.feature_handling import system as sysmod

_VAR = "MLFRAME_MEMORY_BUDGET_GB"


@pytest.mark.parametrize("bad", ["0", "-2", "nan", "inf", "abc"])
def test_non_positive_or_invalid_env_is_ignored(monkeypatch, caplog, bad):
    """A zero, negative, non-finite or unparsable budget falls back and logs a WARNING (39-F52)."""
    monkeypatch.setenv(_VAR, bad)
    monkeypatch.setattr(sysmod.psutil, "virtual_memory", lambda: type("VM", (), {"total": 16 * 10**9})())
    monkeypatch.setattr(sysmod, "_read_cgroup_memory_limit_bytes", lambda: None)
    with caplog.at_level(logging.WARNING):
        budget = sysmod.detect_memory_limit_bytes()
    assert budget > 0
    assert any("MLFRAME_MEMORY_BUDGET_GB" in r.getMessage() for r in caplog.records)


def test_positive_env_is_honoured(monkeypatch):
    """Control: a positive value is converted to bytes (39-F52)."""
    monkeypatch.setenv(_VAR, "4")
    assert sysmod.detect_memory_limit_bytes() == 4 * 10**9
