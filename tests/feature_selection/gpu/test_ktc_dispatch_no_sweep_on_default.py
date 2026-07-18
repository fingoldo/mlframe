"""Regression tests for the kernel_tuning_cache dispatch helpers in
``_benchmarks/kernel_tuning_cache/dispatch.py`` (P2 audit findings).

``lookup_joint_hist`` / ``lookup_mi_classif_backend`` / ``lookup_fe_mi_split_backend`` used to route
their DEFAULT (``run_auto_tune=False``, the only path anything in mlframe actually exercises today)
through ``cache.get_or_tune(...)`` with a no-op tuner (``lambda: []``). Since ``run_auto_tune`` is never
``True`` in production, every cache miss on every host went through get_or_tune's full env-check +
stale-check + guard + ``_run_tuner(...)`` path, which unconditionally logs a misleading INFO "sweep
starting" banner for a sweep that measures nothing. The fix routes the default path through a PURE
``cache.lookup()`` (mirroring ``dispatch.lookup_pairwise_corr_backend`` / the sibling
``training/composite/discovery/_ktc_dispatch._lookup_backend``'s deferred-sweep branch), keeping
``get_or_tune`` only for the explicit ``run_auto_tune=True`` (offline CLI) branch -- which now also
carries ``async_sweep=True`` so a future real tuner can never block a live fit synchronously.

Also pins the fix for an incidental bug found while implementing the above:
``lookup_pairwise_corr_backend`` called ``cache.lookup(kernel, dims={"p": p, "n": n})``, but
``KernelTuningCache.lookup`` takes ``**dims`` (individual kwargs), not a ``dims=`` parameter -- so the
region matcher saw one dim literally named ``"dims"`` (never present as a region constraint key) and
treated EVERY region as unconstrained, always returning the first region regardless of the actual (p, n).
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def _fresh_dispatch_cache(tmp_path, monkeypatch):
    """Isolated per-host cache dir + reset BOTH kernel_tuning_cache singletons (pyutilz's process-wide
    ``load_or_create()`` instance and mlframe's own ``filters._kernel_tuning`` singleton, which
    ``dispatch._get_cache()`` actually routes through) so this test's dispatch calls see a genuinely
    cold cache, unaffected by guards another test in the same session may have tripped."""
    from pyutilz.performance.kernel_tuning import cache as ktc
    from mlframe.feature_selection.filters import _kernel_tuning as _kt

    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    ktc.hw_fingerprint.cache_clear()
    ktc._DEFAULT_INSTANCE = None
    ktc._TUNED_THIS_PROCESS.clear()
    _kt._reset_for_tests()
    yield str(tmp_path)
    ktc.hw_fingerprint.cache_clear()
    ktc._DEFAULT_INSTANCE = None
    _kt._reset_for_tests()


def _no_sweep_disable(monkeypatch):
    """tests/conftest.py sets PYUTILZ_KERNEL_DISABLE_SWEEP=1 session-wide so cache-dependent dispatch
    tests never trigger a real sweep; that gate ALSO suppresses the misleading sweep-starting log this
    module pins (get_or_tune skips ``_run_tuner`` entirely when sweeps are disabled), so unset it to
    exercise the real production sweep-dispatch path."""
    monkeypatch.delenv("PYUTILZ_KERNEL_DISABLE_SWEEP", raising=False)


def test_lookup_joint_hist_default_never_logs_sweep_starting(_fresh_dispatch_cache, monkeypatch, caplog):
    """Lookup joint hist default never logs sweep starting."""
    _no_sweep_disable(monkeypatch)
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with caplog.at_level(logging.INFO):
        result = dispatch.lookup_joint_hist(n_samples=123_456, joint_size=25)  # run_auto_tune=False (default)

    sweep_logs = [r.message for r in caplog.records if "sweep starting" in r.message]
    assert not sweep_logs, f"default (run_auto_tune=False) dispatch must never log a sweep-starting banner: {sweep_logs}"
    assert isinstance(result, dict) and "kernel_variant" in result and "block_size" in result


def test_lookup_mi_classif_backend_default_never_logs_sweep_starting(_fresh_dispatch_cache, monkeypatch, caplog):
    """Lookup mi classif backend default never logs sweep starting."""
    _no_sweep_disable(monkeypatch)
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with caplog.at_level(logging.INFO):
        result = dispatch.lookup_mi_classif_backend(n_samples=50_000, k=10)  # run_auto_tune=False (default)

    sweep_logs = [r.message for r in caplog.records if "sweep starting" in r.message]
    assert not sweep_logs, f"default (run_auto_tune=False) dispatch must never log a sweep-starting banner: {sweep_logs}"
    assert result in ("njit", "cuda")


def test_lookup_fe_mi_split_backend_default_never_logs_sweep_starting(_fresh_dispatch_cache, monkeypatch, caplog):
    """Lookup fe mi split backend default never logs sweep starting."""
    _no_sweep_disable(monkeypatch)
    monkeypatch.delenv("MLFRAME_FE_MI_SPLIT", raising=False)
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    with caplog.at_level(logging.INFO):
        result = dispatch.lookup_fe_mi_split_backend(n_samples=300_000, k=10)  # run_auto_tune=False (default)

    sweep_logs = [r.message for r in caplog.records if "sweep starting" in r.message]
    assert not sweep_logs, f"default (run_auto_tune=False) dispatch must never log a sweep-starting banner: {sweep_logs}"
    assert result in ("single", "split")


def test_run_auto_tune_true_passes_async_sweep(_fresh_dispatch_cache, monkeypatch):
    """The remaining ``run_auto_tune=True`` (offline CLI) branch must pass ``async_sweep=True`` to
    ``get_or_tune``, matching ``TunerSpec.choose()``'s FIT-TIME contract: a real sweep must never block
    a live fit synchronously."""
    from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    monkeypatch.delenv("MLFRAME_FE_MI_SPLIT", raising=False)
    captured: dict = {}

    def _fake_get_or_tune(self, kernel_name, **kwargs):
        """Fake get or tune."""
        captured[kernel_name] = kwargs
        fb = kwargs.get("fallback")
        return fb() if callable(fb) else fb

    monkeypatch.setattr(KernelTuningCache, "get_or_tune", _fake_get_or_tune)

    dispatch.lookup_joint_hist(n_samples=1000, joint_size=10, run_auto_tune=True)
    dispatch.lookup_mi_classif_backend(n_samples=1000, k=5, run_auto_tune=True)
    dispatch.lookup_fe_mi_split_backend(n_samples=1000, k=5, run_auto_tune=True)

    for kernel_name in ("joint_hist_batched", "plugin_mi_classif_dispatch", "fe_mi_split_launch"):
        assert kernel_name in captured, f"{kernel_name} did not reach get_or_tune under run_auto_tune=True"
        assert (
            captured[kernel_name].get("async_sweep") is True
        ), f"{kernel_name}'s run_auto_tune=True branch must pass async_sweep=True so a real tuner never blocks a live fit"


def test_lookup_pairwise_corr_backend_honors_region_size_caps(_fresh_dispatch_cache):
    """Regression for the ``dims={"p": p, "n": n}`` vs ``**dims`` bug: a region's ``p_max``/``n_max``
    caps must actually gate which region is returned, not silently match every call regardless of size."""
    from mlframe.feature_selection.filters import _kernel_tuning as _kt
    from mlframe.feature_selection._benchmarks.kernel_tuning_cache import dispatch

    cache = _kt.get_kernel_tuning_cache()
    if cache is None:
        pytest.skip("KernelTuningCache unavailable on this host")
    cache.update(
        "fe_pairwise_complete_corr",
        axes=["p", "n"],
        regions=[
            {"p_max": 100, "n_max": 100_000, "backend_choice": "cupy"},
            {"backend_choice": "numpy"},  # catch-all
        ],
    )
    # small p, n -> within the first (constrained) region's caps.
    assert dispatch.lookup_pairwise_corr_backend(p=50, n=1000) == "cupy"
    # p above the first region's cap -> must fall through to the catch-all, NOT the first region again.
    assert dispatch.lookup_pairwise_corr_backend(p=500, n=1000) == "numpy"
