"""The kernel-tuning init retry budget must count consecutive failures, not failures forever.

`get_kernel_tuning_cache` retries a failing `KernelTuningCache()` construction three times rather than
latching on the first fault, because the realistic causes (a concurrently-rewritten tuning JSON, a Windows
file lock, a transient nvidia-smi fault) are momentary. But `_INIT_ATTEMPTS` was a monotone process-lifetime
counter reset only by `_reset_for_tests()`, so three UNRELATED faults hours apart exhausted a budget sized
for one persistent breakage -- and every one of the package's kernel-tuning dispatch sites then fell back to
hardcoded defaults for the rest of the process, which is exactly what the retry exists to prevent.
"""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters import _kernel_tuning as kt


@pytest.fixture(autouse=True)
def _clean_singleton():
    """Each test starts and ends with no cached singleton and an unspent budget."""
    kt._reset_for_tests()
    yield
    kt._reset_for_tests()


@pytest.fixture
def always_failing(monkeypatch):
    """Make every construction attempt raise the transient-looking error the retry is written for."""
    calls = []

    def _boom(*args, **kwargs):
        """Stand in for a concurrently-rewritten tuning file."""
        calls.append(1)
        raise ValueError("synthetic transient tuning-file fault")

    # The loader imports the class inside the function body, so the name to replace is the one on pyutilz.
    import pyutilz.performance.kernel_tuning.cache as ktc

    monkeypatch.setattr(ktc, "KernelTuningCache", _boom)
    return calls


class _FakeClock:
    """A monotonic clock the test advances by hand, so no real waiting is involved."""

    def __init__(self, start: float = 1000.0):
        """Start at an arbitrary non-zero point."""
        self.t = start

    def monotonic(self) -> float:
        """The current fake time."""
        return self.t


def _attempts_made(calls) -> int:
    """How many times the constructor actually ran."""
    return len(calls)


def test_three_failures_in_a_row_still_exhaust_the_budget(always_failing, monkeypatch):
    """The persistent-breakage case must keep giving up, or this fix would just remove the ceiling."""
    monkeypatch.setattr(kt, "time", _FakeClock())

    for _ in range(5):
        assert kt.get_kernel_tuning_cache() is None
    assert _attempts_made(always_failing) == kt._MAX_INIT_ATTEMPTS, (
        f"a persistently broken cache was constructed {_attempts_made(always_failing)} times; the budget of " f"{kt._MAX_INIT_ATTEMPTS} is not being enforced"
    )
    assert kt._CACHE_SINGLETON is False, "the terminal fallback was not latched after the budget was spent"


def test_failures_spread_beyond_the_cooldown_do_not_accumulate(always_failing, monkeypatch):
    """The bug: three momentary faults hours apart were treated as one persistent breakage."""
    clock = _FakeClock()
    monkeypatch.setattr(kt, "time", clock)

    for _ in range(5):
        clock.t += kt._INIT_ATTEMPT_COOLDOWN_S * 2  # an unrelated incident, well past the cooldown
        kt._CACHE_SINGLETON = None  # a new lookup after the previous one returned None
        assert kt.get_kernel_tuning_cache() is None

    assert _attempts_made(always_failing) == 5, (
        f"only {_attempts_made(always_failing)} of 5 well-separated faults were retried; the budget still " "accumulates across unrelated incidents"
    )
    assert kt._CACHE_SINGLETON is not False, "the fallback latched permanently despite the faults being far apart"


def test_a_reset_clears_the_failure_timestamp_too(always_failing, monkeypatch):
    """A leftover timestamp would make the next test's first failure look like a continuation."""
    monkeypatch.setattr(kt, "time", _FakeClock())
    kt.get_kernel_tuning_cache()
    assert kt._LAST_INIT_FAILURE_TS != 0.0, "the failure timestamp was never recorded"
    kt._reset_for_tests()
    assert kt._LAST_INIT_FAILURE_TS == 0.0, "_reset_for_tests left the failure timestamp behind"
