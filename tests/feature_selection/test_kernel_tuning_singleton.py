"""Regression test for the shared KernelTuningCache singleton.

A profile of fuzz combo c0143 attributed ~290ms / 6 calls to
``pyutilz.system.kernel_tuning_cache._build_provenance`` (which spawns
nvidia-smi via gpu_capability_summary) reached through
``discretize_2d_array``. Root cause: each call instantiated a fresh
``KernelTuningCache()``, defeating its in-instance ``_loaded`` cache and
re-running ``_load`` (and therefore ``_build_provenance``) every time. Two
more hot-path sites in ``filters/gpu.py`` had the same bug.

The fix routes all FS hot-path callers through one process-wide singleton
in ``filters/_kernel_tuning.py``. This test pins that contract: calling
``get_kernel_tuning_cache()`` N times must return the same instance (so
``_load`` runs at most once) and the singleton must be lock-protected so
concurrent threads don't double-init.
"""

from __future__ import annotations

import threading

import pytest


def test_get_kernel_tuning_cache_returns_singleton():
    from mlframe.feature_selection.filters import _kernel_tuning

    _kernel_tuning._reset_for_tests()
    a = _kernel_tuning.get_kernel_tuning_cache()
    b = _kernel_tuning.get_kernel_tuning_cache()
    c = _kernel_tuning.get_kernel_tuning_cache()
    # If pyutilz is absent on this host the singleton is None for ALL three
    # (the sentinel path also caches the miss).
    if a is None:
        assert b is None and c is None
        return
    assert a is b is c, "get_kernel_tuning_cache must return the same instance"


def test_singleton_thread_safe_init():
    """Concurrent get calls must return ONE instance and never double-init."""
    from mlframe.feature_selection.filters import _kernel_tuning

    _kernel_tuning._reset_for_tests()

    results = []
    barrier = threading.Barrier(8)

    def _worker():
        barrier.wait()
        results.append(_kernel_tuning.get_kernel_tuning_cache())

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8
    if results[0] is None:
        assert all(r is None for r in results)
        return
    # All eight returns must be identity-equal — exactly one instance was created.
    assert all(r is results[0] for r in results), "race produced multiple instances"


def test_dispatch_module_shares_filters_singleton():
    """dispatch.py's _get_cache must delegate to the same singleton so that
    benchmarks and the production filters share ONE KernelTuningCache instance.
    """
    pytest.importorskip("pyutilz.system.kernel_tuning_cache")

    from mlframe.feature_selection.filters import _kernel_tuning
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch as _disp

    _kernel_tuning._reset_for_tests()
    via_filters = _kernel_tuning.get_kernel_tuning_cache()
    if via_filters is None:
        pytest.skip("KernelTuningCache unavailable on this host")
    via_dispatch = _disp._get_cache()
    assert via_filters is via_dispatch, (
        "dispatch._get_cache must return the same instance as the shared singleton"
    )
