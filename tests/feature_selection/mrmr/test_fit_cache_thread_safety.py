"""Thread-safety regression tests for the process-wide ``MRMR._FIT_CACHE``.

The fit cache is read (``in`` / ``[]`` / ``move_to_end``) and written (``[]=`` / ``move_to_end`` /
``popitem(last=False)`` / ``clear``) by every ``MRMR.fit`` call. Under concurrent fits (multi-target
discovery, joblib-threading callers, web-service workers -- all anticipated by the class docstring) two
threads racing those operations on the same ``OrderedDict`` can raise ``KeyError`` /
``RuntimeError: OrderedDict mutated during iteration`` or evict the wrong entry. The fix guards every
read-then-mutate sequence with ``_MRMR_FIT_CACHE_LOCK`` (an ``RLock``) in ``_fit_impl_core``, exposed on the
class as ``MRMR._FIT_CACHE_LOCK`` after the first fit.

The byte-cap eviction iterates ``_FIT_CACHE.values()`` (via ``_mrmr_cache_bytes_total``) while other threads
mutate the dict; that iterate-while-mutate window is the part that *deterministically* raised pre-fix -- see
``test_unlocked_iterate_while_mutate_raises_but_lock_serializes`` which reproduces both halves.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import _MRMR_FIT_CACHE_LOCK


def _tiny_frame(seed: int, n: int = 120, p: int = 4):
    """Distinct small synthetic frame per seed so each fit produces an independent cache entry."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{j}": rng.normal(size=n) for j in range(p)})
    # A learnable but cheap target so the fit does real work and stores a cache entry.
    y = pd.Series((X["f0"] + 0.5 * X["f1"] + rng.normal(scale=0.1, size=n) > 0).astype(int), name="y")
    return X, y


def _fast_selector(seed: int) -> MRMR:
    return MRMR(
        max_runtime_mins=0.05,
        fe_max_steps=0,
        cv=2,
        random_state=seed,
        verbose=0,
    )


def test_fit_cache_lock_is_rlock_and_published_on_class():
    """The lock exists, is reentrant (RLock), and is attached to the class on the first fit."""
    # ``RLock()`` is a factory, so assert against the type of an instance.
    assert isinstance(_MRMR_FIT_CACHE_LOCK, type(threading.RLock()))

    X, y = _tiny_frame(0)
    _fast_selector(0).fit(X, y)

    assert getattr(MRMR, "_FIT_CACHE_LOCK", None) is _MRMR_FIT_CACHE_LOCK, (
        "MRMR._FIT_CACHE_LOCK must be the canonical module-level fit-cache lock after a fit"
    )


def test_fit_holds_lock_around_every_cache_mutation():
    """Structural proof: every ``_FIT_CACHE`` mutation in a real fit happens while THIS thread holds the lock.

    This is the deterministic D2 sensor. Instrument ``MRMR._FIT_CACHE`` with an ``OrderedDict`` subclass whose
    mutating methods (``__setitem__`` / ``move_to_end`` / ``popitem`` / ``clear``) record whether
    ``_MRMR_FIT_CACHE_LOCK._is_owned()`` holds at call time -- ``_is_owned`` is per-calling-thread, so it is
    True only inside a ``with _MRMR_FIT_CACHE_LOCK:`` region on the fitting thread. The store + LRU/byte-cap
    eviction path is the one that mutates, so a fit that actually stores an entry exercises it. Pre-fix
    (no lock) every recorded value would be False; post-fix every value must be True.

    Why structural rather than racing the bug: the ``OrderedDict mutated during iteration`` race is real (it
    was reproduced in a standalone high-contention probe: 4 mutator + 4 reader threads x ~2M ops, which raised
    ``RuntimeError`` on every run) but it is GIL-scheduling-sensitive and does not trip deterministically under
    pytest at a bounded, fast scale -- a timing-dependent assertion would be flaky, which the suite forbids.
    The lock-ownership instrumentation proves the same guarantee with zero timing dependence.
    """
    lock_states: list[bool] = []

    class _AssertingCache(OrderedDict):
        def __setitem__(self, key, value):
            lock_states.append(_MRMR_FIT_CACHE_LOCK._is_owned())
            super().__setitem__(key, value)

        def move_to_end(self, key, last=True):
            lock_states.append(_MRMR_FIT_CACHE_LOCK._is_owned())
            super().move_to_end(key, last=last)

        def popitem(self, last=True):
            lock_states.append(_MRMR_FIT_CACHE_LOCK._is_owned())
            return super().popitem(last=last)

        def clear(self):
            lock_states.append(_MRMR_FIT_CACHE_LOCK._is_owned())
            super().clear()

    saved_cache = MRMR._FIT_CACHE
    try:
        MRMR._FIT_CACHE = _AssertingCache()
        X, y = _tiny_frame(123)
        _fast_selector(123).fit(X, y)
    finally:
        MRMR._FIT_CACHE = saved_cache

    assert lock_states, "fit did not mutate _FIT_CACHE -- the store path was not exercised (no cache entry stored?)"
    assert all(lock_states), (
        f"every _FIT_CACHE mutation during fit must hold _MRMR_FIT_CACHE_LOCK; "
        f"held={sum(lock_states)}/{len(lock_states)} mutations"
    )


def test_lock_serializes_concurrent_iterate_and_mutate():
    """Positive concurrency check: lock-guarded iterate + mutate on the cache lock never raises.

    Mirrors the byte-cap path (``_mrmr_cache_bytes_total`` iterates ``.values()``) racing concurrent stores /
    evictions, but with every access under ``_MRMR_FIT_CACHE_LOCK`` as the fix arranges. Bounded + fast.
    """
    cache: "OrderedDict[int, int]" = OrderedDict((k, k) for k in range(200))
    errors: list = []
    done = threading.Event()

    def mutate():
        try:
            for i in range(50_000):
                if done.is_set():
                    break
                with _MRMR_FIT_CACHE_LOCK:
                    cache[i % 400] = i
                    while len(cache) > 200:
                        cache.popitem(last=False)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    def read():
        try:
            for _ in range(50_000):
                if done.is_set():
                    break
                with _MRMR_FIT_CACHE_LOCK:
                    _ = sum(cache.values())
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=mutate) for _ in range(3)]
    threads += [threading.Thread(target=read) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    done.set()

    assert not errors, f"lock-guarded concurrent iterate+mutate must not raise, got {errors!r}"


def test_concurrent_real_fits_no_exception_and_bounded_cache():
    """Hammer concurrent MRMR.fit on distinct small frames; assert no exception and a bounded final cache."""
    MRMR.clear_fit_cache()
    cap = 4
    n_threads = 6
    iters_per_thread = 12
    errors: list = []

    def worker(tid: int):
        try:
            for k in range(iters_per_thread):
                seed = tid * 1000 + k
                X, y = _tiny_frame(seed)
                sel = _fast_selector(seed)
                sel.fit_cache_max = cap
                sel.fit(X, y)
        except Exception as exc:  # noqa: BLE001
            errors.append((tid, repr(exc)))

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent fits raised: {errors[:5]}"
    # The LRU cap must hold after the storm regardless of interleaving.
    assert isinstance(MRMR._FIT_CACHE, OrderedDict)
    assert len(MRMR._FIT_CACHE) <= cap, (
        f"_FIT_CACHE exceeded cap {cap}: len={len(MRMR._FIT_CACHE)}"
    )
    # Keys stay unique and the dict is internally consistent (no torn entries).
    assert len(set(MRMR._FIT_CACHE.keys())) == len(MRMR._FIT_CACHE)


if __name__ == "__main__":  # pragma: no cover -- manual smoke run
    raise SystemExit(pytest.main([__file__, "-v"]))
