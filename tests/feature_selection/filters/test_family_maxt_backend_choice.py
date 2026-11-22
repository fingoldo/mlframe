"""The family null's device-or-CPU choice: the kernel-tuning cache decides when it has an entry, the work crossover otherwise.

The cache is replaced by a fake so each test pins one branch of ``_family_null_backend`` without a GPU sweep.
"""

from __future__ import annotations

import pytest

import pyutilz.performance.kernel_tuning.cache as ktc_module

import mlframe.feature_selection.filters._family_maxt_gate as gate


class _FakeCache:
    """Stands in for ``KernelTuningCache``: returns a scripted answer and records the lookup it was asked for."""

    def __init__(self, answer):
        self.answer = answer
        self.calls: list[dict] = []

    def get_or_tune(self, name, **kwargs):
        """Record the lookup key and dimensions, then return the scripted answer."""
        self.calls.append({"name": name, **kwargs})
        return self.answer


@pytest.fixture
def cache(monkeypatch):
    """Install a fake cache whose answer the test sets; return a setter."""
    holder: dict = {}

    class _Loader:
        """Replaces ``KernelTuningCache.load_or_create`` with one that hands back the fake."""

        @staticmethod
        def load_or_create():
            """Return the fake installed by the test."""
            return holder["cache"]

    def install(answer) -> _FakeCache:
        """Install a fake returning *answer*."""
        holder["cache"] = _FakeCache(answer)
        monkeypatch.setattr(ktc_module, "KernelTuningCache", _Loader)
        return holder["cache"]

    return install


def test_cache_dict_answer_is_used_for_the_choice(cache):
    """A cache entry with a ``kernel_choice`` decides the backend, whatever the work crossover says."""
    fake = cache({"kernel_choice": "device"})
    assert gate._family_null_backend(1000, 10, work=1) == "device"
    assert fake.calls[0]["name"] == "family_null_backend"
    assert fake.calls[0]["dims"] == {"n_rows": 1000, "n_cols": 10}
    assert fake.calls[0]["axes"] == ["n_rows", "n_cols"]
    assert fake.calls[0]["fallback"] == {"kernel_choice": "cpu"}
    assert fake.calls[0]["async_sweep"] is True


def test_cache_string_answer_is_returned_as_is(cache):
    """A bare string answer from the cache is the backend name."""
    cache("cpu")
    assert gate._family_null_backend(1000, 10, work=10**12) == "cpu"


def test_cache_dict_without_kernel_choice_falls_back_to_the_crossover(cache):
    """A cache entry that does not name a choice uses the work crossover, so the missing key is not read as a backend."""
    cache({"other": "x"})
    assert gate._family_null_backend(1000, 10, work=gate._FAMILY_DEVICE_MIN_WORK) == "device"
    assert gate._family_null_backend(1000, 10, work=gate._FAMILY_DEVICE_MIN_WORK - 1) == "cpu"


def test_empty_cache_answer_falls_back_to_the_crossover(cache):
    """No entry at all (a falsy answer) uses the work crossover."""
    cache(None)
    assert gate._family_null_backend(1000, 10, work=gate._FAMILY_DEVICE_MIN_WORK) == "device"
    assert gate._family_null_backend(1000, 10, work=0) == "cpu"


def test_crossover_is_inclusive_at_the_threshold(cache):
    """Exactly the threshold work goes to the device; one below stays on the CPU."""
    cache(None)
    assert gate._family_null_backend(5, 5, work=gate._FAMILY_DEVICE_MIN_WORK) == "device"
    assert gate._family_null_backend(5, 5, work=gate._FAMILY_DEVICE_MIN_WORK - 1) == "cpu"
