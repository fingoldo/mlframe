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
        # The lock now records "pid:start-token" so a reused pid is not mistaken for the holder (audit 31-F3); the pid is the first field.
        assert (lock_dir / ".ensure.lock").read_text(encoding="utf-8").strip().split(":", 1)[0] == str(os.getpid())
    assert not (lock_dir / ".ensure.lock").exists()


def test_single_flight_same_process_second_thread_does_not_steal_live_lock(lock_dir):
    """While one thread holds the lock, a second thread of the same process gets False and the lock file stays."""
    import threading

    lock = lock_dir / ".ensure.lock"
    held = threading.Event()
    release = threading.Event()
    got = {}

    def _holder():
        """Take the lock and keep it until released."""
        with _ensure.single_flight() as acquired:
            got["holder"] = acquired
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=_holder)
    t.start()
    assert held.wait(timeout=5)
    try:
        with _ensure.single_flight() as second:
            got["second"] = second
        got["still_there"] = lock.exists()
    finally:
        release.set()
        t.join(timeout=5)
    assert got["holder"] is True
    assert got["second"] is False
    assert got["still_there"] is True
