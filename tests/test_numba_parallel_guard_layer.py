"""The parallel-kernel guard is a no-op under a thread-safe numba threading layer and the process-wide lock otherwise or when forced."""

import threading

from mlframe import _numba_parallel_guard as G


def test_forced_always_keeps_the_lock(monkeypatch):
    """MLFRAME_NUMBA_PARALLEL_GUARD=always returns the RLock whatever the layer."""
    monkeypatch.setenv("MLFRAME_NUMBA_PARALLEL_GUARD", "always")
    assert G._layer_allows_concurrent_entry() is False
    monkeypatch.setattr(G, "_concurrent_entry_ok", lambda: False)
    ctx = G.parallel_kernel_entry()
    assert type(ctx) is type(threading.RLock())
    with ctx, ctx:  # reentrant on the same thread
        pass


def test_tbb_layer_enters_freely(monkeypatch):
    """With a layer that tolerates concurrent entry the guard is a no-op context two threads can hold together."""
    monkeypatch.setattr(G, "_concurrent_entry_ok", lambda: True)
    inside = []
    barrier = threading.Barrier(2)

    def worker():
        """Hold the guard and wait for the other thread to be inside as well."""
        with G.parallel_kernel_entry():
            inside.append(1)
            barrier.wait(timeout=30)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in threads]
    [t.join(timeout=60) for t in threads]
    assert inside == [1, 1]


def test_the_probe_reports_a_boolean():
    """Compiling the probe kernel answers with a bool for the active layer on this host."""
    assert isinstance(G._layer_allows_concurrent_entry(), bool)
