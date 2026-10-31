"""The machine-wide tuning lock is not taken over while its owner is still writing it (CON2-1).

A process creates the lock and only then writes its pid. A second process that reads the empty file used to treat it as a dead
lock, unlink it and start a second sweep. A young empty lock now counts as live.
"""

import os
import time

import pytest

from mlframe.system.kernel_tuning_cache import _ensure


@pytest.fixture
def lock_dir(tmp_path, monkeypatch):
    """Point the tuning cache at an empty temp directory for the duration of the test."""
    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    return tmp_path


def test_young_empty_lock_is_treated_as_live_not_taken_over(lock_dir):
    """A lock another process is still filling in must not be unlinked: the caller gets False and no second run starts."""
    lock = lock_dir / ".ensure.lock"
    lock.write_text("", encoding="utf-8")
    with _ensure.single_flight() as acquired:
        assert acquired is False
    assert lock.exists(), "the young lock of a live process was removed"


def test_stale_empty_lock_is_taken_over_once_it_stops_changing(lock_dir):
    """A lock that has not changed for longer than the young window belongs to a crashed process and is reclaimed."""
    lock = lock_dir / ".ensure.lock"
    lock.write_text("", encoding="utf-8")
    old = time.time() - 3600
    os.utime(lock, (old, old))
    with _ensure.single_flight() as acquired:
        assert acquired is True


def test_lock_is_created_with_the_pid_already_in_it(lock_dir):
    """While this process holds the lock its file already names the pid, so no reader can see it empty."""
    with _ensure.single_flight() as acquired:
        assert acquired is True
        assert (lock_dir / ".ensure.lock").read_text(encoding="utf-8").strip() == str(os.getpid())
    assert not (lock_dir / ".ensure.lock").exists()
