"""The kaleido burn threshold counts consecutive persistent-path failures, not every failure since process start."""

from __future__ import annotations

from mlframe.reporting.renderers import _kaleido as kal

def test_a_successful_persistent_write_resets_the_failure_run(monkeypatch):
    """One earlier failure must not burn the path on the next single failure once a persistent write has succeeded."""
    monkeypatch.setattr(kal, "_KALEIDO_PERSISTENT_FAIL_COUNT", 0)
    monkeypatch.setattr(kal, "_KALEIDO_PERSISTENT_BURNED", False)
    assert kal._record_kaleido_persistent_failure() is False  # first failure of the run
    kal._reset_kaleido_persistent_failures()  # a successful persistent write ends the run
    assert kal._record_kaleido_persistent_failure() is False  # the next failure starts a new run, below the threshold
    assert kal._is_kaleido_persistent_burned() is False

def test_threshold_still_burns_on_consecutive_failures(monkeypatch):
    """Two failures in a row still burn the path."""
    monkeypatch.setattr(kal, "_KALEIDO_PERSISTENT_FAIL_COUNT", 0)
    monkeypatch.setattr(kal, "_KALEIDO_PERSISTENT_BURNED", False)
    assert kal._record_kaleido_persistent_failure() is False
    assert kal._record_kaleido_persistent_failure() is True
    assert kal._is_kaleido_persistent_burned() is True
