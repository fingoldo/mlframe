"""LightGBM's native entry points run one host thread at a time, process-wide.

A production kernel died three times out of three with a heap corruption while composite discovery trained LightGBM
from 16 threads. Locally the threaded discovery crashed 3 of 9 fresh processes -- one thread in Booster.__init__, one in
update, one in predict -- and 0 of 12 with these entry points serialised.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from mlframe._lightgbm_thread_safety import GUARDED_METHODS, LGB_NATIVE_LOCK, patch_lightgbm_basic


def _fake_module():
    """Stand-ins for lightgbm.basic's classes that record how many threads are inside a guarded method at once."""
    state = {"live": 0, "peak": 0, "lock": threading.Lock()}

    def _probe(*_a, **_k):
        with state["lock"]:
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
        time.sleep(0.01)
        with state["lock"]:
            state["live"] -= 1

    class Dataset:
        construct = _probe

    class Booster:
        def __init__(self, *a, **k):
            _probe()

        update = _probe
        predict = _probe

    return SimpleNamespace(Dataset=Dataset, Booster=Booster), state


def _hammer(module, n_threads=8):
    barrier = threading.Barrier(n_threads)

    def work():
        barrier.wait()
        b = module.Booster()
        b.update()
        b.predict()
        module.Dataset().construct()

    threads = [threading.Thread(target=work) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_guarded_methods_never_overlap_across_threads():
    module, state = _fake_module()
    assert set(patch_lightgbm_basic(module)) == {f"{c}.{m}" for c, ms in GUARDED_METHODS.items() for m in ms}
    _hammer(module)
    assert state["peak"] == 1


def test_the_probe_sees_overlap_without_the_guard():
    """The probe is evidence only if it can see concurrency."""
    module, state = _fake_module()
    _hammer(module)
    assert state["peak"] > 1


def test_patching_twice_wraps_nothing_the_second_time():
    module, _ = _fake_module()
    patch_lightgbm_basic(module)
    assert patch_lightgbm_basic(module) == []


def test_the_lock_is_reentrant():
    """lgb.train builds its datasets from inside Booster.__init__: the same thread re-enters the lock."""
    with LGB_NATIVE_LOCK:
        # A non-reentrant lock would deadlock here; the timeout turns that into a failure instead of a hang.
        reacquired = LGB_NATIVE_LOCK.acquire(timeout=5)
        assert reacquired, "the same thread could not re-enter LGB_NATIVE_LOCK"
        LGB_NATIVE_LOCK.release()


def test_the_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv("MLFRAME_LGB_SERIALISE", "0")
    module, _ = _fake_module()
    assert patch_lightgbm_basic(module) == []


def test_the_installed_lightgbm_is_guarded():
    """Importing mlframe and lightgbm leaves the real native entry points wrapped."""
    pytest.importorskip("lightgbm")
    import lightgbm

    import mlframe  # noqa: F401 -- installs the import hook

    for cls_name, methods in GUARDED_METHODS.items():
        cls = getattr(lightgbm.basic, cls_name)
        assert methods
        for name in methods:
            assert getattr(cls.__dict__[name], "_mlframe_lgb_serialised", False), f"{cls_name}.{name} is not serialised"
