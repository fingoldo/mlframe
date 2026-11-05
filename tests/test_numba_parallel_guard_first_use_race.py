"""Concurrent first calls to the guard's process-wide lock all receive the same RLock.

The lock used to come from ``lru_cache``, which does not serialise its first call. Two threads that reached it together could each build an
RLock and keep different ones, so two fan-outs held "the" lock at once. A slow constructor makes the window reliable to hit.
"""

from __future__ import annotations

import threading
import time

import mlframe._numba_parallel_guard as guard


def test_concurrent_first_callers_all_receive_one_lock(monkeypatch):
    """With a slow RLock constructor and many threads racing the first call, every thread gets the same object."""
    monkeypatch.setattr(guard, "_ENTRY_LOCK", None)
    real_rlock = threading.RLock

    def slow_rlock():
        """Build an RLock after a pause, widening the window between the check and the store."""
        time.sleep(0.05)
        return real_rlock()

    monkeypatch.setattr(threading, "RLock", slow_rlock)
    n_threads = 16
    barrier = threading.Barrier(n_threads)
    got = [None] * n_threads

    def worker(i):
        """Wait for every thread to be ready, then ask for the lock."""
        barrier.wait()
        got[i] = guard._entry_lock()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len({id(x) for x in got}) == 1
