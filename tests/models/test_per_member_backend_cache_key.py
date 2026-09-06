"""The per-member backend memo must key on the env it reads, and must not memoise a transient failure.

`_per_member_use_numba` carried an `@lru_cache` keyed on `(elements_per_member, n_groups, ndim)` while its
body read `MLFRAME_PER_MEMBER_BACKEND` and `MLFRAME_PER_MEMBER_AUTOTUNE` and consulted the per-host
`KernelTuningCache` file -- none of which were in the key. The first call at a given shape froze whatever
the env said at that moment, and a later override was silently discarded.

The same body ended in a bare `except Exception` returning the element-count heuristic, so a momentary
lookup failure -- the tuning JSON being rewritten by a concurrent sweep, a Windows file lock from another
mlframe process, a transient nvidia-smi fault -- pinned that heuristic for the shape for the rest of the
process. The module's own docstring measures the two backends 5-18x apart across the 2-D regime, so a wrong
verdict there is not cosmetic.
"""

from __future__ import annotations

import pytest

from mlframe.models.ensembling import member_metrics as mm


@pytest.fixture(autouse=True)
def _clear_memo():
    """Each test starts from an empty memo.

    Written to tolerate the memo living on either function so the env-override test below reports a real
    assertion failure against the pre-fix shape rather than erroring in setup on a missing name.
    """
    for name in ("_per_member_backend_cached", "_per_member_use_numba"):
        fn = getattr(mm, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    yield
    for name in ("_per_member_backend_cached", "_per_member_use_numba"):
        fn = getattr(mm, name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()


def test_an_env_override_set_after_the_first_call_is_honoured(monkeypatch):
    """The env is read on every call, so changing it asks a different question rather than hitting a stale entry."""
    monkeypatch.setenv("MLFRAME_PER_MEMBER_BACKEND", "numba")
    assert mm._per_member_use_numba(50_000, 4, 2) is True

    monkeypatch.setenv("MLFRAME_PER_MEMBER_BACKEND", "numpy")
    assert mm._per_member_use_numba(50_000, 4, 2) is False, "the override was ignored; the shape-only memo answered instead"

    monkeypatch.setenv("MLFRAME_PER_MEMBER_BACKEND", "numba")
    assert mm._per_member_use_numba(50_000, 4, 2) is True


def test_the_env_is_part_of_the_cache_key(monkeypatch):
    """Two different env values at one shape must occupy two entries, not overwrite one answer."""
    monkeypatch.setenv("MLFRAME_PER_MEMBER_BACKEND", "numba")
    mm._per_member_use_numba(50_000, 4, 2)
    monkeypatch.setenv("MLFRAME_PER_MEMBER_BACKEND", "numpy")
    mm._per_member_use_numba(50_000, 4, 2)
    assert mm._per_member_backend_cached.cache_info().currsize >= 2


def test_a_transient_lookup_failure_is_not_memoised(monkeypatch):
    """`lru_cache` does not memoise a raised exception, which is what keeps the failure from sticking."""
    monkeypatch.delenv("MLFRAME_PER_MEMBER_BACKEND", raising=False)
    calls = {"n": 0}

    def _flaky(elements_per_member, n_groups, ndim, env, autotune):
        """Fail the first time, succeed afterwards."""
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("tuning cache locked by a concurrent sweep")
        return True

    monkeypatch.setattr(mm, "_per_member_backend_cached", __import__("functools").lru_cache(maxsize=256)(_flaky))

    # First call: the lookup fails, and the caller gets the element-count fallback rather than an exception.
    assert mm._per_member_use_numba(50_000, 4, 2) is (50_000 >= mm._PER_MEMBER_NUMBA_FLOOR_ELEMENTS)
    # Second call at the SAME shape: the failure was not cached, so the lookup is re-attempted and succeeds.
    assert mm._per_member_use_numba(50_000, 4, 2) is True, "the transient failure was memoised for this shape"
    assert calls["n"] == 2, f"the lookup was not re-attempted (called {calls['n']} times)"


def test_a_transient_failure_is_reported_above_debug(monkeypatch, caplog):
    """A wrong backend verdict is 5-18x of runtime, so it must leave more than a debug line."""
    import functools
    import logging

    monkeypatch.delenv("MLFRAME_PER_MEMBER_BACKEND", raising=False)

    def _always_fails(*_a, **_k):
        """Stand in for a tuning-cache lookup that cannot complete right now."""
        raise RuntimeError("nvidia-smi timed out")

    monkeypatch.setattr(mm, "_per_member_backend_cached", functools.lru_cache(maxsize=256)(_always_fails))
    with caplog.at_level(logging.WARNING, logger=mm.logger.name):
        mm._per_member_use_numba(50_000, 4, 2)
    assert any("transiently" in r.getMessage() for r in caplog.records), "a transient lookup failure was not reported above debug"


def test_the_element_floor_still_decides_the_fallback(monkeypatch):
    """The fallback is unchanged; only its permanence was the defect."""
    import functools

    monkeypatch.delenv("MLFRAME_PER_MEMBER_BACKEND", raising=False)

    def _always_fails(*_a, **_k):
        """Stand in for an unavailable tuning cache."""
        raise RuntimeError("unavailable")

    monkeypatch.setattr(mm, "_per_member_backend_cached", functools.lru_cache(maxsize=256)(_always_fails))
    below = mm._PER_MEMBER_NUMBA_FLOOR_ELEMENTS - 1
    above = mm._PER_MEMBER_NUMBA_FLOOR_ELEMENTS
    assert mm._per_member_use_numba(below, 4, 2) is False
    assert mm._per_member_use_numba(above, 4, 2) is True
