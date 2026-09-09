"""Regression tests for the macOS libomp-crash mitigation in lgb_shim.py.

LightGBM's Dataset construction opens an OpenMP parallel region whose thread-pool
initialisation crashes inside libomp.dylib itself on macOS CI (EXC_BAD_ACCESS in
__kmp_suspend_initialize_thread) -- confirmed a fault in libomp, not in LightGBM's
or mlframe's own code, and confirmed NOT specific to any one libomp version (both
22.1.8 and 23.1.0 reproduce it; see audits/ci_review_2026-09-08/_TRACKER.md, X5).
lgb_default_n_jobs() forces n_jobs=1 on darwin so LightGBM never opens that parallel
region; these tests pin that every LightGBM construction site in the package routes
through it, and that non-macOS platforms are unaffected.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("lightgbm")


def _reload_lgb_shim(monkeypatch: pytest.MonkeyPatch, platform: str, allow_multithread: str | None = None):
    """Reload lgb_shim.py with a patched sys.platform / env var, mirroring how the
    module's own module-level _MACOS_LGB_FORCE_SERIAL is computed once at import time.
    """
    import mlframe.training.lgb_shim as lgb_shim

    monkeypatch.setattr(lgb_shim._sys, "platform", platform)
    if allow_multithread is None:
        monkeypatch.delenv("MLFRAME_LGB_MACOS_ALLOW_MULTITHREAD", raising=False)
    else:
        monkeypatch.setenv("MLFRAME_LGB_MACOS_ALLOW_MULTITHREAD", allow_multithread)
    return importlib.reload(lgb_shim)


@pytest.fixture(autouse=True)
def _restore_lgb_shim():
    """Reload back to the real platform after each test so module-level state
    (_MACOS_LGB_FORCE_SERIAL, computed once at import) doesn't leak into other tests.
    """
    yield
    import mlframe.training.lgb_shim as lgb_shim

    importlib.reload(lgb_shim)


def test_lgb_default_n_jobs_forces_serial_on_darwin(monkeypatch):
    """On darwin, every n_jobs request (-1, None, or explicit) resolves to 1."""
    m = _reload_lgb_shim(monkeypatch, "darwin")
    assert m._MACOS_LGB_FORCE_SERIAL is True
    assert m.lgb_default_n_jobs(-1) == 1
    assert m.lgb_default_n_jobs(None) == 1
    assert m.lgb_default_n_jobs(4) == 1


def test_lgb_default_n_jobs_unaffected_on_linux_and_windows(monkeypatch):
    """Non-darwin platforms keep the original cpu_count()/explicit-value behaviour."""
    import os

    for plat in ("linux", "win32"):
        m = _reload_lgb_shim(monkeypatch, plat)
        assert m._MACOS_LGB_FORCE_SERIAL is False
        assert m.lgb_default_n_jobs(-1) == (os.cpu_count() or 1)
        assert m.lgb_default_n_jobs(None) == (os.cpu_count() or 1)
        assert m.lgb_default_n_jobs(4) == 4


def test_lgb_default_n_jobs_escape_hatch_env_var(monkeypatch):
    """MLFRAME_LGB_MACOS_ALLOW_MULTITHREAD=1 opts back into multithreaded LightGBM
    on macOS, for a host/libomp build that doesn't hit the fault.
    """
    m = _reload_lgb_shim(monkeypatch, "darwin", allow_multithread="1")
    assert m._MACOS_LGB_FORCE_SERIAL is False
    assert m.lgb_default_n_jobs(4) == 4


def test_lgb_shim_fit_path_uses_lgb_default_n_jobs_on_darwin(monkeypatch):
    """The dataset-reuse shim's own .fit() n_jobs pre-fill (the original call site
    this bug was found through) must route through lgb_default_n_jobs, not
    os.cpu_count() directly -- this is the exact regression the mitigation guards.
    """
    import numpy as np

    m = _reload_lgb_shim(monkeypatch, "darwin")
    rng = np.random.default_rng(0)
    X = rng.random((100, 4)).astype(np.float64)
    y = (X[:, 0] > 0.5).astype(int)
    model = m.LGBMClassifierWithDatasetReuse(n_estimators=5, verbose=-1)
    model.fit(X, y)
    assert model.n_jobs == 1


def test_helpers_training_configs_lgb_general_params_serial_on_darwin(monkeypatch):
    """The central LGB_GENERAL_PARAMS config (used by train_mlframe_models_suite,
    the exact repro path for the crash) must also resolve n_jobs=1 on darwin, not
    just the dataset-reuse shim's own fit() path.
    """
    m = _reload_lgb_shim(monkeypatch, "darwin")
    import mlframe.training._helpers_training_configs as h

    importlib.reload(h)
    try:
        assert h.lgb_default_n_jobs(-1) == 1
    finally:
        importlib.reload(m)
        importlib.reload(h)
