"""Concurrent first callers of is_gpu_metrics_available run the device probe once and agree on the answer."""

import threading

import mlframe.metrics._gpu_metrics as gm


def test_probe_runs_once_when_eight_threads_race_the_first_call(monkeypatch):
    """A slow probe stands in for the cupy check; the lock must make the other seven callers reuse its result."""
    calls = []
    probe_may_finish = threading.Event()

    def slow_probe():
        """Record the call, pause, and publish the cached answer like the real probe does."""
        calls.append(1)
        probe_may_finish.wait(10)  # held open until every caller has asked, so the others queue behind the lock
        gm._GPU_AVAILABLE = True
        return True

    monkeypatch.setattr(gm, "_GPU_AVAILABLE", None)
    monkeypatch.setattr(gm, "_probe_gpu_metrics_available", slow_probe)
    results = []
    barrier = threading.Barrier(8)

    def worker():
        """Wait at the barrier, then ask for availability."""
        barrier.wait()
        results.append(gm.is_gpu_metrics_available())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    while len(results) < 1 and not calls:  # the first caller is inside the probe
        threading.Event().wait(0.001)
    probe_may_finish.set()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert results == [True] * 8
