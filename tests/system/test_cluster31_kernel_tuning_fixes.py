"""Cluster 31 regressions: a reused-pid lock is stale, and a sweep that persisted nothing is a failure (audit 2026-11-20, findings 31-F2, 31-F3)."""

import os
import types

import mlframe.system.kernel_tuning_cache._ensure as ensure
from mlframe.system.kernel_tuning_cache import cmd_refresh


def test_lock_recorded_by_another_process_incarnation_is_taken_over(tmp_path, monkeypatch):
    """A live pid whose recorded start token does not match is a reused pid: the lock is stale and tuning proceeds (31-F3)."""
    lock = tmp_path / "tuning.lock"
    holder_pid = os.getppid()  # a live process that is certainly not this test
    lock.write_text(f"{holder_pid}:0.000", encoding="utf-8")
    monkeypatch.setattr(ensure, "_lock_path", lambda: lock)
    with ensure.single_flight() as acquired:
        assert acquired is True


def test_live_holder_with_matching_token_still_blocks(tmp_path, monkeypatch):
    """The same process with its recorded token still holds the lock; a second run must not start (31-F3)."""
    lock = tmp_path / "tuning.lock"
    token = ensure._pid_start_token(os.getppid())
    lock.write_text(f"{os.getppid()}:{token}", encoding="utf-8")
    monkeypatch.setattr(ensure, "_lock_path", lambda: lock)
    if not token:
        return  # no psutil here: the token check is not applicable
    with ensure.single_flight() as acquired:
        assert acquired is False


def test_refresh_with_no_persisted_regions_returns_failure(monkeypatch):
    """An empty sweep used to exit 0 as a success (31-F2)."""
    import mlframe.system.kernel_tuning_cache as kt

    spec = types.SimpleNamespace(kernel_name="kern")
    monkeypatch.setattr(kt, "tune_spec", lambda s, force: 0)
    assert cmd_refresh({"kern": spec}, "kern") != 0
