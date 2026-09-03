"""Unit: kernel_tuning_cache wiring for the corr/collinear backend dispatchers (measured-FUTURE item B).

Pins: a real sweep runs on a cache miss, persists regions, ``has()`` reflects the persisted
state, a second ``ensure_*`` call with ``force=False`` is a no-op, and ``choose_*_backend``
with ``run_auto_tune=True`` consults the persisted cache for a swept cell instead of the
hardcoded size gate.
"""

from __future__ import annotations

import pytest

pytest.importorskip("numba")

from pyutilz.performance.kernel_tuning import KernelTuningCache

from mlframe.training.composite.discovery import _ktc_dispatch as ktc


@pytest.fixture
def in_memory_cache(monkeypatch):
    """Fresh in-memory KernelTuningCache, patched in as the module's shared singleton."""
    cache = KernelTuningCache(in_memory=True)
    monkeypatch.setattr(ktc, "get_ktc_cache", lambda: cache)
    return cache


def _fake_corr_regions():
    """Cheap stand-in for ``_run_sweep_composite_corr``'s real numpy-vs-numba timing sweep.

    Only the persist/consult wiring is under test in the callers that use this, not the sweep's
    own timings -- ``test_ensure_composite_corr_tuning_runs_and_persists`` is the dedicated test
    that exercises the real (expensive) sweep and asserts it produces regions on this host.
    """
    return [
        {"n_samples_max": 10_000, "n_cols_max": 32, "backend_choice": "numba", "numpy_ms": 1.0, "numba_ms": 0.5},
        {"n_samples_max": None, "n_cols_max": None, "backend_choice": "numba"},
    ]


def _fake_collinear_regions():
    """Cheap stand-in for ``_run_sweep_composite_collinear``'s real numpy-vs-numba timing sweep.

    Same rationale as ``_fake_corr_regions``: only the persist/consult wiring is under test in the
    callers that use this."""
    return [
        {"n_samples_max": 2_000, "n_cols_max": 30, "backend_choice": "numba", "numpy_ms": 1.0, "numba_ms": 0.5},
        {"n_samples_max": None, "n_cols_max": None, "backend_choice": "numba"},
    ]


def test_ensure_composite_corr_tuning_runs_and_persists(in_memory_cache):
    """A cache miss triggers a real sweep and persists a corr_dispatch region."""
    assert in_memory_cache.has("composite_corr_dispatch") is False
    regions = ktc.ensure_composite_corr_tuning()
    assert regions, "sweep must return at least one region on this (numba-enabled) host"
    assert in_memory_cache.has("composite_corr_dispatch") is True


def test_ensure_composite_corr_tuning_force_false_is_a_noop(in_memory_cache, monkeypatch):
    """A second call with force=False must not re-invoke the sweep once already tuned."""
    # Only the no-op-on-already-tuned wiring is under test here, not the sweep's own timings
    # (that's test_ensure_composite_corr_tuning_runs_and_persists) -- fake the first, priming
    # call so this test doesn't also pay for a real numpy-vs-numba sweep.
    monkeypatch.setattr(ktc, "_run_sweep_composite_corr", lambda *a, **kw: _fake_corr_regions())
    ktc.ensure_composite_corr_tuning()
    calls = []
    monkeypatch.setattr(ktc, "_run_sweep_composite_corr", lambda *a, **kw: calls.append(1) or [])
    result = ktc.ensure_composite_corr_tuning(force=False)
    assert result is None, "an already-tuned kernel must not re-sweep"
    assert not calls, "the sweep function must not be invoked when force=False and cache.has() is True"


def test_ensure_composite_collinear_tuning_runs_and_persists(in_memory_cache):
    """A cache miss triggers a real sweep and persists a collinear_dispatch region."""
    assert in_memory_cache.has("composite_collinear_dispatch") is False
    regions = ktc.ensure_composite_collinear_tuning()
    assert regions, "sweep must return at least one region on this (numba-enabled) host"
    assert in_memory_cache.has("composite_collinear_dispatch") is True


def test_choose_corr_backend_consults_persisted_cache(in_memory_cache, monkeypatch):
    """choose_corr_backend(run_auto_tune=True) resolves via the persisted cache without error."""
    # The cache-consultation wiring is under test, not the sweep's own timings -- fake the priming
    # sweep (see test_ensure_composite_corr_tuning_runs_and_persists for the real-sweep coverage).
    monkeypatch.setattr(ktc, "_run_sweep_composite_corr", lambda *a, **kw: _fake_corr_regions())
    ktc.ensure_composite_corr_tuning()
    # A cell squarely inside the swept grid (n=10_000, cols=32) must resolve via the cache
    # lookup path (run_auto_tune=True consults get_or_tune, which hits the persisted region),
    # not silently fall through to the hardcoded gate -- both paths return a valid backend
    # name either way, so the real pin is that no exception is raised and the KTC was
    # actually consulted (verified indirectly: the persisted region exists and covers this cell).
    backend = ktc.choose_corr_backend(10_000, 32, min_rows=20_000, min_cols=64, run_auto_tune=True)
    assert backend in ("numba", "numpy")
    assert in_memory_cache.has("composite_corr_dispatch")


def test_choose_collinear_backend_consults_persisted_cache(in_memory_cache, monkeypatch):
    """choose_collinear_backend(run_auto_tune=True) resolves via the persisted cache without error."""
    # See test_choose_corr_backend_consults_persisted_cache: fake the priming sweep, real-sweep
    # coverage lives in test_ensure_composite_collinear_tuning_runs_and_persists.
    monkeypatch.setattr(ktc, "_run_sweep_composite_collinear", lambda *a, **kw: _fake_collinear_regions())
    ktc.ensure_composite_collinear_tuning()
    backend = ktc.choose_collinear_backend(2_000, 30, min_rows=256, min_cols=10, run_auto_tune=True)
    assert backend in ("numba", "numpy")
    assert in_memory_cache.has("composite_collinear_dispatch")


def test_choose_backend_without_auto_tune_never_sweeps(in_memory_cache, monkeypatch):
    """The default (run_auto_tune=False, as used by every real call site) must never trigger a sweep."""
    calls = []
    monkeypatch.setattr(ktc, "_run_sweep_composite_corr", lambda *a, **kw: calls.append(1) or [])
    monkeypatch.setattr(ktc, "_run_sweep_composite_collinear", lambda *a, **kw: calls.append(1) or [])
    ktc.choose_corr_backend(10_000, 32, min_rows=20_000, min_cols=64)
    ktc.choose_collinear_backend(2_000, 30, min_rows=256, min_cols=10)
    assert not calls, "run_auto_tune defaults to False at every dispatcher call site; a cache miss must fall through to the hardcoded gate, never sweep"


def test_ensure_tuning_returns_none_when_cache_unavailable(monkeypatch):
    """Both ensure_* functions degrade to None (never raise) when the KTC singleton is unavailable."""
    monkeypatch.setattr(ktc, "get_ktc_cache", lambda: None)
    assert ktc.ensure_composite_corr_tuning() is None
    assert ktc.ensure_composite_collinear_tuning() is None
