"""Regression test for the macOS concurrent-fit numba-abort fix (see ``_mrmr_class.py``'s
``_MACOS_NUMBA_PARALLEL_FIT_LOCK``).

CI reproduced (run 34612906516, shards 4/6/9, 2026-09-11): two threads calling ``MRMR.fit()``
concurrently triggered "Numba workqueue threading layer is terminating: Concurrent access has been
detected" followed by "Fatal Python error: Aborted" on macos-latest -- numba's default 'workqueue'
threading layer is not thread-safe for concurrent parallel=True kernel launch from multiple Python
threads. ``fit()``'s own docstring anticipates concurrent multi-threaded calls (multi-target
discovery, joblib-threading callers, web-service workers), so this is a genuine macOS stability gap,
not a test-only artifact.

Real darwin isn't available on this dev box, so this test simulates it by monkeypatching the
module's platform-derived lock to a real ``threading.Lock()`` (mirroring what ``sys.platform ==
"darwin"`` would compute at import time) and proves two concurrent ``fit()`` calls never run
``_fit_body`` simultaneously.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import _mrmr_class
from mlframe.feature_selection.filters.mrmr import MRMR


def _tiny_frame(seed: int, n: int = 80, p: int = 3):
    """Small synthetic frame, fast enough for a concurrency test."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(p)})
    y = pd.Series((X["f0"] + rng.normal(scale=0.1, size=n) > 0).astype(int), name="y")
    return X, y


def _fast_selector(seed: int) -> MRMR:
    """Fast selector."""
    return MRMR(max_runtime_mins=0.05, fe_max_steps=0, cv=2, random_state=seed, verbose=0)


def test_macos_lock_exists_and_is_a_lock():
    """The platform-gated lock matches ``sys.platform`` exactly: a real ``threading.Lock`` on darwin,
    ``None`` everywhere else -- not merely "one of the two", which would pass even if the platform
    gate itself were inverted or broken."""
    assert hasattr(_mrmr_class, "_MACOS_NUMBA_PARALLEL_FIT_LOCK"), "the platform-gated lock attribute is missing entirely"
    lock = _mrmr_class._MACOS_NUMBA_PARALLEL_FIT_LOCK
    is_darwin = sys.platform == "darwin"
    assert is_darwin == isinstance(lock, type(threading.Lock())), f"sys.platform={sys.platform!r} but lock={lock!r}"
    assert is_darwin or lock is None


def test_simulated_darwin_lock_serializes_concurrent_fit_bodies(monkeypatch):
    """With the darwin lock active, two threads' ``_fit_body`` calls never overlap in time."""
    sim_lock = threading.Lock()
    monkeypatch.setattr(_mrmr_class, "_MACOS_NUMBA_PARALLEL_FIT_LOCK", sim_lock)

    in_body = threading.Event()
    overlap_detected = threading.Event()
    real_fit_body = _mrmr_class.MRMR._fit_body

    def _spying_fit_body(self, *args, **kwargs):
        """Detect whether another thread is already inside ``_fit_body`` when this one enters."""
        if in_body.is_set():
            overlap_detected.set()
        in_body.set()
        try:
            time.sleep(0.05)  # widen the window so a real race would be caught
            return real_fit_body(self, *args, **kwargs)
        finally:
            in_body.clear()

    monkeypatch.setattr(_mrmr_class.MRMR, "_fit_body", _spying_fit_body)

    errors: list = []

    def worker(seed: int):
        """Fit one selector on its own frame."""
        try:
            X, y = _tiny_frame(seed)
            _fast_selector(seed).fit(X, y)
        except Exception as exc:  # pragma: no cover - surfaced via errors list
            errors.append((seed, repr(exc)))

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, f"concurrent fits raised: {errors}"
    assert not overlap_detected.is_set(), "two threads were inside _fit_body simultaneously under the simulated darwin lock"


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="this test monkeypatches away the REAL darwin lock to prove concurrent fits overlap without it -- "
    "running that on an actual macOS host reproduces the exact numba workqueue abort the lock exists to "
    "prevent (CI: run 34661105847, shard 4/10, Fatal Python error: Aborted from this test's own worker "
    "threads). The scenario it demonstrates (no lock -> real overlap) is only meaningful to prove on a "
    "platform where overlap is actually safe.",
)
def test_lock_is_none_off_darwin_so_real_concurrency_is_unaffected(monkeypatch):
    """Off darwin (the real value on this box), fit() takes the plain no-lock path -- concurrent
    fits are free to overlap, matching pre-fix behaviour on Linux/Windows."""
    monkeypatch.setattr(_mrmr_class, "_MACOS_NUMBA_PARALLEL_FIT_LOCK", None)

    overlap_detected = threading.Event()
    in_body = threading.Event()
    real_fit_body = _mrmr_class.MRMR._fit_body

    def _spying_fit_body(self, *args, **kwargs):
        """Detect concurrent entry."""
        if in_body.is_set():
            overlap_detected.set()
        in_body.set()
        try:
            time.sleep(0.05)
            return real_fit_body(self, *args, **kwargs)
        finally:
            in_body.clear()

    monkeypatch.setattr(_mrmr_class.MRMR, "_fit_body", _spying_fit_body)

    def worker(seed: int):
        """Fit one selector on its own frame."""
        X, y = _tiny_frame(seed)
        _fast_selector(seed).fit(X, y)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert overlap_detected.is_set(), "expected concurrent _fit_body entry with no lock active (sanity check that the spy/timing actually detects overlap)"
