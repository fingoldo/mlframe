"""Regression tests for the shared-state / lifecycle fixes in the 2026-10-04 concurrency audit (C5-01..C5-09)."""

from __future__ import annotations

import gc
import os
import threading
import warnings
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.filters import _gpu_resident_fe as grf
from mlframe.feature_selection.filters import _joblib_safe as js
from mlframe.feature_selection.filters import screen
from mlframe.feature_selection.filters.mrmr import _mrmr_class_fit_helpers as fit_helpers
from mlframe.utils.daemon_task import start_daemon_task
from mlframe.utils.warning_filters import silenced_once

_JOIN_S = 10.0


def _join(thread: threading.Thread) -> None:
    """Join a helper thread and fail loudly if it is still alive."""
    thread.join(timeout=_JOIN_S)
    assert not thread.is_alive()


def _solo_draws(seed: int) -> np.ndarray:
    """The two-draw sequence a seeded scope yields when nothing else touches the global RNG."""
    with screen._preserve_global_numpy_rng_state_unlocked(seed):
        return np.random.random(2)


def _interleaved_first_draw_then_second(scope, seed_a: int, seed_b: int) -> np.ndarray:
    """T1 enters ``scope(seed_a)`` and draws once; a second thread then enters ``scope(seed_b)`` if it can; T1 draws again.

    With an exclusive scope the second thread cannot get inside while T1 holds it, so the helper only waits for T2 to be inside when the scope is unguarded."""
    t1_first_done, t1_resume = threading.Event(), threading.Event()
    t2_inside, t2_release = threading.Event(), threading.Event()
    out: list[float] = []

    def _t1() -> None:
        """Seeded scope that pauses between its two draws."""
        with scope(seed_a):
            out.append(float(np.random.random()))
            t1_first_done.set()
            t1_resume.wait(timeout=_JOIN_S)
            out.append(float(np.random.random()))

    def _t2() -> None:
        """A competing seeded scope that reseeds the global RNG and stays inside until released."""
        with scope(seed_b):
            t2_inside.set()
            t2_release.wait(timeout=_JOIN_S)

    th1 = threading.Thread(target=_t1)
    th1.start()
    assert t1_first_done.wait(timeout=_JOIN_S)
    th2 = threading.Thread(target=_t2)
    th2.start()
    if scope is screen._preserve_global_numpy_rng_state_unlocked:
        assert t2_inside.wait(timeout=_JOIN_S)
    else:
        got = screen._GLOBAL_RNG_SCOPE_LOCK.acquire(blocking=False)
        if got:
            screen._GLOBAL_RNG_SCOPE_LOCK.release()
        assert not got, "a second thread must not be able to enter while T1 holds the scope"
        assert not t2_inside.is_set()
    t1_resume.set()
    _join(th1)
    t2_release.set()
    _join(th2)
    return np.asarray(out)


def test_seeded_scope_is_exclusive_and_streams_stay_reproducible() -> None:
    """C5-01: a concurrent seeded scope cannot reseed T1 mid-block; the unguarded variant demonstrably does."""
    solo = _solo_draws(11)
    guarded = _interleaved_first_draw_then_second(screen._preserve_global_numpy_rng_state, 11, 22)
    np.testing.assert_array_equal(guarded, solo)
    unguarded = _interleaved_first_draw_then_second(screen._preserve_global_numpy_rng_state_unlocked, 11, 22)
    assert not np.array_equal(unguarded, solo), "fixture must show the pre-fix interleaving corrupts the stream"


def test_seeded_scope_restores_caller_state() -> None:
    """C5-01: the scope still leaves the caller's global numpy stream where it found it."""
    np.random.seed(5)
    before = np.random.get_state()
    with screen._preserve_global_numpy_rng_state(99):
        np.random.random(3)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


def _old_take(host_codes):
    """The pre-fix lookup: id() only, shape/dtype compare, no identity check."""
    h = grf._RESIDENT_CODES_HANDOFF.get(id(host_codes))
    if h is None:
        return None
    dev, shape, dtype = h[:3]
    return dev if tuple(host_codes.shape) == shape and np.dtype(host_codes.dtype) == dtype else None


def test_resident_codes_not_served_to_a_recycled_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5-02: an entry stashed for array A is never returned for a different live array B that inherited A's id()."""
    monkeypatch.setattr(grf, "_RESIDENT_CODES_HANDOFF", OrderedDict())
    a = np.zeros((4, 3), dtype=np.int32)
    b = np.zeros((4, 3), dtype=np.int32)
    sentinel = object()
    grf._stash_resident_codes(a, sentinel)
    assert grf.take_resident_codes(a) is sentinel
    grf._RESIDENT_CODES_HANDOFF[id(b)] = grf._RESIDENT_CODES_HANDOFF.pop(id(a))  # b now sits where a recycled address would
    assert _old_take(b) is sentinel, "pre-fix logic serves A's device codes to B"
    assert grf.take_resident_codes(b) is None


def test_resident_codes_entry_dies_with_host_and_lookup_takes_the_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5-02: the handoff holds only a weakref to the host, and take_resident_codes reads under the registry lock."""
    monkeypatch.setattr(grf, "_RESIDENT_CODES_HANDOFF", OrderedDict())
    acquisitions: list[str] = []

    class _SpyLock:
        """Lock stand-in recording every acquisition."""

        def __enter__(self) -> None:
            """Record the acquisition."""
            acquisitions.append("enter")

        def __exit__(self, *exc: object) -> None:
            """Release (no-op)."""

    monkeypatch.setattr(grf, "_DEFERRED_HOST_FILL_LOCK", _SpyLock())
    a = np.zeros((2, 2), dtype=np.int32)
    grf._stash_resident_codes(a, object())
    acquisitions.clear()
    grf.take_resident_codes(a)
    assert acquisitions == ["enter"]
    ref = grf._RESIDENT_CODES_HANDOFF[id(a)][3]
    del a
    gc.collect()
    assert ref() is None


def _fresh_cache(monkeypatch: pytest.MonkeyPatch, max_entries: int) -> None:
    """Isolate the module's memmap cache and bound it to ``max_entries``."""
    monkeypatch.setattr(js, "_FIT_MEMMAP_CACHE", OrderedDict())
    monkeypatch.setattr(js, "_FIT_MEMMAP_RETIRED", [])
    monkeypatch.setattr(js, "_FIT_MEMMAP_CACHE_MAX_ENTRIES", max_entries)


def _cleanup_cache() -> None:
    """Unlink every file the isolated cache still owns."""
    for view, path in list(js._FIT_MEMMAP_CACHE.values()) + list(js._FIT_MEMMAP_RETIRED):
        js._release_fit_constant_entry(view, path)


def test_evicted_memmap_survives_while_a_caller_holds_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5-03: LRU eviction must not unlink a file a live Parallel call still references; it goes once the caller lets go."""
    _fresh_cache(monkeypatch, 2)
    try:
        held = js.fit_constant_memmap(np.full((50, 4), 1.0))
        held_path = held.filename
        js.fit_constant_memmap(np.full((50, 4), 2.0))
        js.fit_constant_memmap(np.full((50, 4), 3.0))  # evicts the LRU entry == `held`
        assert os.path.exists(held_path), "eviction unlinked a file a caller still holds"
        np.testing.assert_array_equal(np.asarray(held), 1.0)
        del held
        js.fit_constant_memmap(np.full((50, 4), 4.0))  # next eviction sweeps the retired entry
        assert not os.path.exists(held_path)
        assert js._FIT_MEMMAP_RETIRED == []
    finally:
        _cleanup_cache()


def test_eviction_unlinks_immediately_when_nothing_holds_the_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5-03: the 13GB-leak guard still works for unreferenced entries; forcing 'in use' reproduces the pre-fix unconditional unlink contrast."""
    _fresh_cache(monkeypatch, 1)
    try:
        path = js.fit_constant_memmap(np.full((50, 4), 1.0)).filename
        js.fit_constant_memmap(np.full((50, 4), 2.0))
        assert not os.path.exists(path)
    finally:
        _cleanup_cache()


def test_enter_active_fit_scope_rearms_under_the_counter_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5-05: the 0->1 re-arm runs while the counter lock is held, so a second fit cannot trip a breaker in the gap and have it cleared."""
    monkeypatch.setattr(fit_helpers, "_ACTIVE_FIT_COUNT", 0)
    held_during_rearm: list[bool] = []

    def _rearm() -> None:
        """Probe whether the counter lock is held at the moment the breakers are re-armed."""
        got = fit_helpers._ACTIVE_FIT_COUNT_LOCK.acquire(blocking=False)
        if got:
            fit_helpers._ACTIVE_FIT_COUNT_LOCK.release()
        held_during_rearm.append(not got)

    fake = SimpleNamespace(_rearm_gpu_circuit_breakers=_rearm)
    fit_helpers._MRMRFitHelpersMixin._enter_active_fit_scope(fake)  # type: ignore[arg-type]  # duck-typed self
    fit_helpers._MRMRFitHelpersMixin._enter_active_fit_scope(fake)  # type: ignore[arg-type]
    assert held_during_rearm == [True], "re-arm must run exactly once, on 0->1, inside the lock"
    fit_helpers._MRMRFitHelpersMixin._exit_active_fit_scope(fake)  # type: ignore[arg-type]
    fit_helpers._MRMRFitHelpersMixin._exit_active_fit_scope(fake)  # type: ignore[arg-type]
    assert fit_helpers._ACTIVE_FIT_COUNT == 0


def test_neural_mi_estimators_leave_callers_torch_stream_alone() -> None:
    """C5-07: a seeding estimator wrapper restores torch's global RNG, so the caller's later torch draws are unshifted."""
    torch = pytest.importorskip("torch")
    from mlframe.feature_selection.filters import _neural_mi as nm

    @nm._restores_torch_rng
    def _seeds_inside() -> float:
        """Stand-in for an estimator that reseeds torch for reproducibility."""
        torch.manual_seed(1234)
        return float(torch.rand(1))

    torch.manual_seed(7)
    expected_next = float(torch.rand(1))
    torch.manual_seed(7)
    _seeds_inside()
    assert float(torch.rand(1)) == expected_next


def test_silenced_once_does_not_leak_or_drop_filters_under_overlap() -> None:
    """C5-08: overlapping catch_warnings blocks leak a blanket 'ignore' (shown), while silenced_once installs one narrow filter and never restores."""

    class _Cat(UserWarning):
        """Unique category so the test cannot collide with other filters."""

    def _overlap(enter) -> None:
        """T1 and T2 enter the scope in sequence and T1 leaves first, the interleaving that corrupts snapshot/restore."""
        t1_in, t2_in, t1_out = threading.Event(), threading.Event(), threading.Event()

        def _t1() -> None:
            """Enter first, leave while T2 is still inside."""
            with enter():
                t1_in.set()
                assert t2_in.wait(timeout=_JOIN_S)
            t1_out.set()

        def _t2() -> None:
            """Enter after T1, leave after T1 has left."""
            with enter():
                t2_in.set()
                assert t1_out.wait(timeout=_JOIN_S)

        a = threading.Thread(target=_t1)
        a.start()
        assert t1_in.wait(timeout=_JOIN_S)
        b = threading.Thread(target=_t2)
        b.start()
        _join(a)
        _join(b)

    def _blanket_ignore_present() -> bool:
        """Whether a process-wide 'ignore everything' filter is installed."""
        return any(f[0] == "ignore" and f[2] is Warning and f[1] is None for f in warnings.filters)

    def _catch():
        """The pre-fix scope shape."""
        cm = warnings.catch_warnings()

        class _Wrapped:
            """Context manager applying simplefilter('ignore') inside catch_warnings."""

            def __enter__(self) -> None:
                """Enter and install the blanket ignore."""
                cm.__enter__()
                warnings.simplefilter("ignore")

            def __exit__(self, *exc: object) -> None:
                """Exit through catch_warnings."""
                cm.__exit__(*exc)

        return _Wrapped()

    with warnings.catch_warnings():
        warnings.resetwarnings()
        _overlap(lambda: silenced_once(_Cat, r"c508_mod"))
        assert not _blanket_ignore_present()
        n_filters = sum(1 for f in warnings.filters if f[2] is _Cat)
        assert n_filters == 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn_explicit("x", _Cat, "f.py", 1, module="c508_mod")
            warnings.warn_explicit("y", _Cat, "f.py", 1, module="other_mod")
        assert [str(w.message) for w in caught] == ["y"]
    with warnings.catch_warnings():
        warnings.resetwarnings()
        _overlap(_catch)
        assert _blanket_ignore_present(), "fixture must show the pre-fix overlap leaks a blanket ignore"


def test_daemon_task_settles_future_on_base_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5-09: a KeyboardInterrupt in the task still settles the future (the waiter does not hang to its timeout); the thread is a daemon."""
    seen = threading.Event()
    monkeypatch.setattr(threading, "excepthook", lambda args: seen.set())

    def _boom() -> None:
        """Raise a BaseException that is not an Exception."""
        raise KeyboardInterrupt("render interrupted")

    fut = start_daemon_task(_boom)
    with pytest.raises(KeyboardInterrupt):
        fut.result(timeout=_JOIN_S)
    assert seen.wait(timeout=_JOIN_S)
    for t in [t for t in threading.enumerate() if t.name == "mlframe-daemon-task"]:
        _join(t)


def test_daemon_task_thread_is_daemon_and_never_blocks_exit() -> None:
    """C5-04: the watchdog task thread is a daemon (an abandoned hung pool call cannot hold interpreter exit); results flow through the future."""
    release = threading.Event()
    started = threading.Event()

    def _hung() -> int:
        """Block until released, standing in for a wedged loky pool call."""
        started.set()
        release.wait(timeout=_JOIN_S)
        return 7

    fut = start_daemon_task(_hung)
    assert started.wait(timeout=_JOIN_S)
    workers = [t for t in threading.enumerate() if t.name == "mlframe-daemon-task"]
    assert workers and all(t.daemon for t in workers)
    release.set()
    assert fut.result(timeout=_JOIN_S) == 7
    for t in workers:
        _join(t)
