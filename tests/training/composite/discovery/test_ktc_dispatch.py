"""Regression tests for ``discovery._ktc_dispatch``'s KernelTuningCache-backed
backend lookup: with a pre-populated cache entry the dispatcher must use the
cached ``backend_choice``, and with no cache entry it must fall back cleanly to
the hardcoded size gate (never raise, never leave the choice undefined)."""
import pytest

from mlframe.training.composite.discovery._ktc_dispatch import (
    choose_collinear_backend,
    choose_corr_backend,
)


@pytest.fixture
def _fresh_ktc_cache_dir(tmp_path, monkeypatch):
    """Isolated per-host cache dir + a reset process-wide KTC singleton, so this
    test's lookups can never be short-circuited by a guard another test already
    tripped for the real user cache (same pattern as
    ``tests/feature_selection/fe/adaptive/test_fe_rung_schedule.py``)."""
    from pyutilz.performance.kernel_tuning import cache as ktc

    monkeypatch.setenv("PYUTILZ_KERNEL_CACHE_DIR", str(tmp_path))
    ktc.hw_fingerprint.cache_clear()
    ktc._DEFAULT_INSTANCE = None
    ktc._TUNED_THIS_PROCESS.clear()
    yield str(tmp_path)
    ktc.hw_fingerprint.cache_clear()
    ktc._DEFAULT_INSTANCE = None


def _clear_ktc_singleton(monkeypatch):
    """``_ktc_dispatch._get_cache`` memoises the singleton on the FS-side module
    (``mlframe.feature_selection.filters``); force it to re-resolve against the
    fresh in-process cache dir set by ``_fresh_ktc_cache_dir``."""
    from mlframe.feature_selection.filters import _kernel_tuning as fs_ktc

    monkeypatch.setattr(fs_ktc, "_CACHE_SINGLETON", None, raising=False)


def test_choose_corr_backend_uses_cached_entry(_fresh_ktc_cache_dir, monkeypatch):
    """With a persisted ``composite_corr_dispatch`` region covering the queried
    dims, the dispatcher must return the CACHED backend, even when it disagrees
    with the hardcoded size-gate fallback."""
    _clear_ktc_singleton(monkeypatch)
    monkeypatch.delenv("MLFRAME_COMPOSITE_CORR_BACKEND", raising=False)
    from mlframe.feature_selection.filters import get_kernel_tuning_cache

    cache = get_kernel_tuning_cache()
    assert cache is not None, "pyutilz KernelTuningCache unavailable -- cannot exercise the cache-hit path"

    # Hardcoded gate (see ``_corr_numba.py``) would pick "numpy" at these dims
    # (n_rows below any realistic min_rows); persist a region that says the
    # opposite ("numba") so a cache-hit is unambiguously distinguishable from
    # the fallback.
    cache.update(
        "composite_corr_dispatch",
        axes=["n_samples", "n_cols"],
        regions=[
            {"n_samples_max": None, "n_cols_max": None, "backend_choice": "numba"},
        ],
    )

    backend = choose_corr_backend(100, 4, min_rows=20_000, min_cols=64, run_auto_tune=False)
    assert backend == "numba", "cached region must win over the hardcoded numpy fallback"


def test_choose_corr_backend_falls_back_on_cache_miss(_fresh_ktc_cache_dir, monkeypatch):
    """With no persisted tuning for this kernel, the dispatcher must fall back to
    the hardcoded size gate cleanly (no exception, no sweep triggered since
    ``run_auto_tune`` defaults to False)."""
    _clear_ktc_singleton(monkeypatch)
    monkeypatch.delenv("MLFRAME_COMPOSITE_CORR_BACKEND", raising=False)

    # Below both thresholds -> hardcoded fallback is "numpy".
    backend_small = choose_corr_backend(100, 4, min_rows=20_000, min_cols=64, run_auto_tune=False)
    assert backend_small == "numpy"

    # Above both thresholds -> hardcoded fallback is "numba".
    backend_large = choose_corr_backend(50_000, 128, min_rows=20_000, min_cols=64, run_auto_tune=False)
    assert backend_large == "numba"


def test_choose_collinear_backend_uses_cached_entry(_fresh_ktc_cache_dir, monkeypatch):
    """A cached collinear_dispatch region wins over the hardcoded gate."""
    _clear_ktc_singleton(monkeypatch)
    monkeypatch.delenv("MLFRAME_COMPOSITE_COLLINEAR_BACKEND", raising=False)
    from mlframe.feature_selection.filters import get_kernel_tuning_cache

    cache = get_kernel_tuning_cache()
    assert cache is not None, "pyutilz KernelTuningCache unavailable -- cannot exercise the cache-hit path"

    cache.update(
        "composite_collinear_dispatch",
        axes=["n_samples", "n_cols"],
        regions=[
            {"n_samples_max": None, "n_cols_max": None, "backend_choice": "numpy"},
        ],
    )

    # Hardcoded gate would pick "numba" at these dims (above both min_rows/min_cols).
    backend = choose_collinear_backend(1_000, 50, min_rows=256, min_cols=10, run_auto_tune=False)
    assert backend == "numpy", "cached region must win over the hardcoded numba fallback"


def test_choose_collinear_backend_falls_back_on_cache_miss(_fresh_ktc_cache_dir, monkeypatch):
    """No persisted collinear_dispatch region -> falls back cleanly to the hardcoded size gate."""
    _clear_ktc_singleton(monkeypatch)
    monkeypatch.delenv("MLFRAME_COMPOSITE_COLLINEAR_BACKEND", raising=False)

    backend_small = choose_collinear_backend(100, 4, min_rows=256, min_cols=10, run_auto_tune=False)
    assert backend_small == "numpy"

    backend_large = choose_collinear_backend(1_000, 50, min_rows=256, min_cols=10, run_auto_tune=False)
    assert backend_large == "numba"


def test_env_override_wins_over_cached_entry(_fresh_ktc_cache_dir, monkeypatch):
    """The env-var force-override is documented as taking priority over the KTC
    lookup; a cached region that disagrees must never win."""
    _clear_ktc_singleton(monkeypatch)
    from mlframe.feature_selection.filters import get_kernel_tuning_cache

    cache = get_kernel_tuning_cache()
    assert cache is not None
    cache.update(
        "composite_corr_dispatch",
        axes=["n_samples", "n_cols"],
        regions=[{"n_samples_max": None, "n_cols_max": None, "backend_choice": "numba"}],
    )
    monkeypatch.setenv("MLFRAME_COMPOSITE_CORR_BACKEND", "numpy")
    backend = choose_corr_backend(100, 4, min_rows=20_000, min_cols=64, run_auto_tune=False)
    assert backend == "numpy"


def test_no_auto_tune_never_triggers_a_sweep(_fresh_ktc_cache_dir, monkeypatch):
    """``run_auto_tune`` defaults to False in every production call site
    (``_corr_numba.py`` / ``_collinear_numba.py``); a cache miss with the default
    must go through ``cache.lookup`` (silent None -> fallback), never
    ``cache.get_or_tune`` (which would run the real inline sweep)."""
    _clear_ktc_singleton(monkeypatch)
    monkeypatch.delenv("MLFRAME_COMPOSITE_CORR_BACKEND", raising=False)
    from mlframe.feature_selection.filters import get_kernel_tuning_cache

    cache = get_kernel_tuning_cache()
    assert cache is not None
    sweep_calls = {"n": 0}
    monkeypatch.setattr(cache, "get_or_tune", lambda *a, **kw: sweep_calls.__setitem__("n", sweep_calls["n"] + 1))

    backend = choose_corr_backend(100, 4, min_rows=20_000, min_cols=64, run_auto_tune=False)

    assert sweep_calls["n"] == 0, "run_auto_tune=False must never call get_or_tune (would run a real sweep)"
    assert backend == "numpy"
