"""LightGBM dataset construction runs one thread at a time.

A production Jupyter kernel died with a Windows access violation plus a heap corruption (0xc0000374) while 16 rerank
threads were inside ``Booster.update`` and three more were inside a dataset construction. LightGBM's binning is not
safe to enter concurrently, so the shared-fold cache must let only one thread construct at a time; training stays
parallel, which is where the rerank's time actually goes.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from mlframe.training.composite.discovery import _lgb_shared_fold as sf


class _ConcurrencyProbe:
    """Stands in for ``lightgbm.Dataset``, recording how many threads are inside ``construct`` at once."""

    def __init__(self, data, label=None, params=None, free_raw_data=True):
        self.data = data
        self.label = label

    def construct(self):
        """Hold the "construction" open long enough for any other thread to overlap it."""
        with _STATE["lock"]:
            _STATE["live"] += 1
            _STATE["peak"] = max(_STATE["peak"], _STATE["live"])
        time.sleep(0.02)
        with _STATE["lock"]:
            _STATE["live"] -= 1
        return self

    def set_label(self, label):
        """Label swap, as the real dataset does between specs."""
        self.label = label


_STATE = {"lock": threading.Lock(), "live": 0, "peak": 0}


def _fake_train(params, ds, num_boost_round=0):
    """Stand-in booster: training is deliberately left parallel, so it only needs to take some time."""
    time.sleep(0.01)
    return object()


def _run_threads(n_threads: int, monkeypatch) -> int:
    """Drive ``fit_on_shared_fold`` from ``n_threads`` threads, each on its own matrix; returns the construct peak."""
    import lightgbm as lgb

    monkeypatch.setattr(lgb, "Dataset", _ConcurrencyProbe)
    monkeypatch.setattr(lgb, "train", _fake_train)
    _STATE["live"] = 0
    _STATE["peak"] = 0
    sf._CACHE.clear()
    params = sf.lgb_params(num_leaves=15, learning_rate=0.1, random_state=0, deterministic=False, num_threads=1)
    barrier = threading.Barrier(n_threads)

    def _worker(seed: int) -> None:
        x = np.zeros((32, 4), dtype=np.float32) + seed
        rows = np.arange(32)
        barrier.wait()  # every thread reaches construction together: without the lock they would overlap
        sf.fit_on_shared_fold(x, rows, np.zeros(32), params=params, n_estimators=1)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return int(_STATE["peak"])


def test_only_one_thread_constructs_at_a_time(monkeypatch):
    """The defect: constructions ran concurrently because the cache lock only guarded its dict."""
    assert _run_threads(8, monkeypatch) == 1


def test_the_probe_would_catch_concurrency_without_the_lock(monkeypatch):
    """The probe is only evidence if it can see overlap: with the lock neutralised it must report more than one."""
    import contextlib

    monkeypatch.setattr(sf, "_CONSTRUCT_LOCK", contextlib.nullcontext())
    assert _run_threads(8, monkeypatch) > 1
