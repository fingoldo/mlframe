"""The fresh-instance defaults cache is populated once even when many threads miss it at the same time.

Its miss path constructs a full MRMR(); without a lock every concurrently-unpickling joblib worker thread paid that construction.
"""

from __future__ import annotations

import threading
import time

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters.mrmr import _mrmr_class_config as cc


def test_fresh_instance_defaults_cache_populates_once_under_threads(monkeypatch):
    """Eight threads hitting a cold cache together construct the fresh instance exactly once, and all get the same dict."""
    monkeypatch.setattr(cc, "_FRESH_INSTANCE_DEFAULTS_CACHE", {})
    real_init = MRMR.__init__
    calls = {"n": 0}
    count_lock = threading.Lock()

    def slow_counting_init(self, *a, **kw):
        """Count constructions and widen the race window."""
        with count_lock:
            calls["n"] += 1
        time.sleep(0.05)
        real_init(self, *a, **kw)

    monkeypatch.setattr(MRMR, "__init__", slow_counting_init)
    barrier = threading.Barrier(8)
    results = []

    def worker():
        """Release all threads together, then resolve the defaults."""
        barrier.wait()
        results.append(MRMR._resolve_fresh_instance_defaults())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert calls["n"] == 1, f"fresh MRMR() constructed {calls['n']} times under a concurrent cold cache"
    assert all(r is results[0] for r in results)
