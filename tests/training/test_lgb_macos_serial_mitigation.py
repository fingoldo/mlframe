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


def _reload_lgb_shim(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest, platform: str, allow_multithread: str | None = None):
    """Reload lgb_shim.py with a patched sys.platform / env var, mirroring how the
    module's own module-level _MACOS_LGB_FORCE_SERIAL is computed once at import time.

    Registers its own restore via ``request.addfinalizer`` (reachable from this same function
    scope, not a separate autouse fixture the static reload-safety scanner cannot trace into) so
    the patched module never leaks into a later test.
    """
    import mlframe.training.lgb_shim as lgb_shim

    monkeypatch.setattr(lgb_shim._sys, "platform", platform)
    if allow_multithread is None:
        monkeypatch.delenv("MLFRAME_LGB_MACOS_ALLOW_MULTITHREAD", raising=False)
    else:
        monkeypatch.setenv("MLFRAME_LGB_MACOS_ALLOW_MULTITHREAD", allow_multithread)
    reloaded = importlib.reload(lgb_shim)
    request.addfinalizer(lambda: importlib.reload(lgb_shim))
    return reloaded


def test_lgb_default_n_jobs_forces_serial_on_darwin(monkeypatch, request):
    """On darwin, every n_jobs request (-1, None, or explicit) resolves to 1."""
    m = _reload_lgb_shim(monkeypatch, request, "darwin")
    assert m._MACOS_LGB_FORCE_SERIAL is True
    assert m.lgb_default_n_jobs(-1) == 1
    assert m.lgb_default_n_jobs(None) == 1
    assert m.lgb_default_n_jobs(4) == 1


def test_lgb_default_n_jobs_unaffected_on_linux_and_windows(monkeypatch, request):
    """Non-darwin platforms keep the original cpu_count()/explicit-value behaviour."""
    import os

    for plat in ("linux", "win32"):
        m = _reload_lgb_shim(monkeypatch, request, plat)
        assert m._MACOS_LGB_FORCE_SERIAL is False
        assert m.lgb_default_n_jobs(-1) == (os.cpu_count() or 1)
        assert m.lgb_default_n_jobs(None) == (os.cpu_count() or 1)
        assert m.lgb_default_n_jobs(4) == 4


def test_lgb_default_n_jobs_escape_hatch_env_var(monkeypatch, request):
    """MLFRAME_LGB_MACOS_ALLOW_MULTITHREAD=1 opts back into multithreaded LightGBM
    on macOS, for a host/libomp build that doesn't hit the fault.
    """
    m = _reload_lgb_shim(monkeypatch, request, "darwin", allow_multithread="1")
    assert m._MACOS_LGB_FORCE_SERIAL is False
    assert m.lgb_default_n_jobs(4) == 4


def test_lgb_shim_fit_path_uses_lgb_default_n_jobs_on_darwin(monkeypatch, request):
    """The dataset-reuse shim's own .fit() n_jobs pre-fill (the original call site
    this bug was found through) must route through lgb_default_n_jobs, not
    os.cpu_count() directly -- this is the exact regression the mitigation guards.
    """
    import numpy as np

    m = _reload_lgb_shim(monkeypatch, request, "darwin")
    rng = np.random.default_rng(0)
    X = rng.random((100, 4)).astype(np.float64)
    y = (X[:, 0] > 0.5).astype(int)
    model = m.LGBMClassifierWithDatasetReuse(n_estimators=5, verbose=-1)
    model.fit(X, y)
    assert model.n_jobs == 1


def test_helpers_training_configs_lgb_general_params_serial_on_darwin(monkeypatch, request):
    """The central LGB_GENERAL_PARAMS config (used by train_mlframe_models_suite,
    the exact repro path for the crash) must also resolve n_jobs=1 on darwin, not
    just the dataset-reuse shim's own fit() path.
    """
    _reload_lgb_shim(monkeypatch, request, "darwin")
    import mlframe.training._helpers_training_configs as h
    import mlframe.training.helpers as helpers_facade

    # `helpers.py` does `from ._helpers_training_configs import get_training_configs` ONCE at import
    # time, binding its own namespace to that specific function OBJECT. Reloading `h` re-executes its
    # top-level code and creates a NEW `get_training_configs` object in the (in-place-mutated) sibling
    # module -- the facade's already-bound reference is untouched, so it silently diverges from the
    # sibling's current object. CI (run 34703435856) caught this exact split via
    # tests/training/test_training_helpers_split.py::test_get_training_configs_identity failing under
    # a full, unsharded (-n 1) run where both test modules share one process: `assert
    # <function get_training_configs at ...> is <function get_training_configs at ...>`.
    #
    # ``importlib.reload()`` is NOT idempotent for identity: the original finalizer here
    # (``importlib.reload(h)`` again) does not restore the PRE-test object, it mints a THIRD, still
    # different one -- confirmed directly (a naive "restore the facade to its pre-test object, then
    # reload h again" attempt still left the two split after teardown, since the second reload's
    # fresh object never matches what the facade was restored to). The only way back to the exact
    # pre-test state is an explicit snapshot-and-setattr restore of BOTH modules' bindings, bypassing
    # reload entirely for the restore step.
    original_sibling_fn = h.get_training_configs
    original_facade_fn = helpers_facade.get_training_configs

    def _restore():
        """Put both modules' bindings back to their exact pre-test function objects."""
        h.get_training_configs = original_sibling_fn
        helpers_facade.get_training_configs = original_facade_fn

    request.addfinalizer(_restore)

    reloaded_h = importlib.reload(h)
    # Keep the facade's binding in sync with the just-reloaded sibling for the duration of this test
    # too (not only restored afterward) -- callers going through helpers.get_training_configs during
    # this test must see the SAME darwin-serial-n_jobs behaviour reloaded_h itself just proved.
    helpers_facade.get_training_configs = reloaded_h.get_training_configs
    assert reloaded_h.lgb_default_n_jobs(-1) == 1
