"""
Root pytest fixtures shared across all test modules.
"""

import gc
import os
import warnings

# Best-effort silence for native-stderr chatter from Intel OpenMP. BLAS xerbla
# output ("** On entry to DLASCLS parameter number 4 had an illegal value") is
# printed from C via stderr and bypasses OPENBLAS_VERBOSE / MKL_VERBOSE flags;
# filter THOSE at the log boundary instead (PowerShell ``| Where-Object {
# $_ -notmatch '^ \*\* On entry to D[A-Z]+ parameter' } | Tee-Object``).
# KMP_WARNINGS does cleanly silence Intel OpenMP runtime chatter, so keep it.
os.environ.setdefault("KMP_WARNINGS", "off")

import pytest

# Auto-register synthetic-data fixtures from tests.training.synthetic so they're
# discoverable from tests outside tests/training (cross-package fixture sharing
# without per-test from-imports). Plugins listed here are loaded at conftest
# import time.
pytest_plugins = ["tests.training.synthetic"]

try:
    import psutil
except ImportError:
    psutil = None


# ---------------------------------------------------------------------------
# Fast-mode support
#
# Enable via `pytest --fast` or `MLFRAME_FAST=1 pytest`.
#
# Parametrized tests that cover many equivalent variants (10 scalers, every
# dim-reducer, every optimizer) can call `fast_subset(values)` when building
# their `@pytest.mark.parametrize` lists. In fast mode this returns a single
# representative value, so the whole code path is still exercised but runtime
# stays short. Outside fast mode it's an identity function.
#
# parametrize decorators run at import time (before fixtures), so fast mode is
# keyed off the env var, not a pytest fixture. The CLI flag just sets the env
# var in pytest_configure.
# ---------------------------------------------------------------------------

_FAST_ENV = "MLFRAME_FAST"


def is_fast_mode() -> bool:
    """Return True when tests should run in reduced-variant fast mode."""
    return os.environ.get(_FAST_ENV, "").strip() not in ("", "0", "false", "False")


# Canonical module-level fast-mode flag. Subdir conftests historically defined
# their own ``IS_FAST_MODE`` snapshots; they should re-use this one.
IS_FAST_MODE = is_fast_mode()


def fast_subset(values, *, representative=None, keep: int = 1):
    """In fast mode return a single representative; otherwise return values unchanged.

    `values` may be any iterable of parametrize arguments (including
    `pytest.param(...)` entries). `representative` picks a specific entry when
    the first element isn't the desired one; if None, the first `keep` entries
    are kept.
    """
    if not is_fast_mode():
        return list(values)
    values = list(values)
    if not values:
        return values
    if representative is not None:
        for v in values:
            candidate = v.values[0] if hasattr(v, "values") else v
            if candidate == representative:
                return [v]
        # fall through if not found
    return values[:keep]


def require_polars_ds():
    """Skip the calling test when ``polars-ds`` is not importable. Single source of truth so the 8x duplicated
    `pytest.importorskip("polars_ds")` calls in `tests/inference/`, `tests/training/test_pipeline.py`, and
    `tests/training/test_polarsds_pipeline_json_proxy.py` can collapse to one helper invocation. Keeps the
    importorskip reason consistent and centralises the gate when polars-ds gating changes."""
    return pytest.importorskip(
        "polars_ds",
        reason="polars-ds optional extra not installed; tests exercising the polars-native PreprocessingBackendConfig fastpath are skipped",
    )


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Fast mode: parametrized tests run one representative variant per group "
             f"(also enabled by {_FAST_ENV}=1).",
    )
    parser.addoption(
        "--run-fuzz",
        action="store_true",
        default=False,
        help="Include long-running fuzz-combo tests (test_fuzz_suite, "
             "test_fuzz_3way_suite, ...). They are deselected by default even "
             "without --fast because each runs ~150 combos through the suite.",
    )
    parser.addoption(
        "--run-biz-transformer",
        action="store_true",
        default=False,
        help="Include the feature_engineering/transformer/test_biz_val_*.py "
             "business-value tests. Each fits multiple boostings on real "
             "datasets (kin8nm, mammography, California Housing, ...) and "
             "takes minutes per case; deselected from the default run.",
    )


def pytest_configure(config):
    if config.getoption("--fast"):
        os.environ[_FAST_ENV] = "1"
    # Markers ``fast_only`` / ``slow_only`` / ``no_xdist`` are registered in pyproject.toml
    # ``[tool.pytest.ini_options].markers``; double-registering here invites silent description
    # drift (audit D2 P2 #18). The conftest only registers markers that are conftest-private
    # (defined and skipped here) or unique to the option-flag pattern that the marker docs in
    # pyproject would not naturally cover.
    config.addinivalue_line(
        "markers",
        "fuzz: long-running fuzz-combo test; deselected unless --run-fuzz is passed.",
    )
    config.addinivalue_line(
        "markers",
        "biz_transformer: feature_engineering/transformer biz_val test (real "
        "datasets, multi-boosting); deselected unless --run-biz-transformer "
        "is passed.",
    )


def pytest_collection_modifyitems(config, items):
    # Always: deselect ``fuzz``-marked items unless --run-fuzz is given. Applies
    # in BOTH fast and full modes - fuzz combos run hundreds of train_mlframe_models_suite
    # iterations and should never be in the standard CI loop. Opt in explicitly.
    if not config.getoption("--run-fuzz"):
        skip_fuzz = pytest.mark.skip(
            reason="skipped by default; pass --run-fuzz to include long fuzz-combo tests"
        )
        for item in items:
            if "fuzz" in item.keywords:
                item.add_marker(skip_fuzz)

    # Same opt-in pattern for the feature_engineering/transformer biz_val
    # tests: real-dataset fits across multiple boostings, several minutes
    # per test, not standard CI loop material.
    if not config.getoption("--run-biz-transformer"):
        skip_bt = pytest.mark.skip(
            reason="skipped by default; pass --run-biz-transformer to include "
                   "feature_engineering/transformer biz_val tests"
        )
        for item in items:
            if "biz_transformer" in item.keywords:
                item.add_marker(skip_bt)

    # Skip ``no_xdist``-marked items when xdist parallelism is active. Tests
    # that touch shared FS state (numba cache wipe), do heavy in-process
    # composite fits, or rely on a stable cwd will native-crash under load
    # even though they pass in isolation (observed 2026-05-20 on S: with
    # test_stacked_improves_holdout_mae_on_2level_synthetic and
    # test_iter118_year100k_cb_r2_iter102).
    _dist_opt = getattr(config.option, "dist", "no")
    if _dist_opt != "no":
        skip_xdist = pytest.mark.skip(
            reason="requires sequential execution; xdist parallelism active"
        )
        for item in items:
            if "no_xdist" in item.keywords:
                item.add_marker(skip_xdist)

    if not is_fast_mode():
        return
    skip_slow = pytest.mark.skip(reason="skipped in --fast mode")
    for item in items:
        if "slow_only" in item.keywords or "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def fast_mode() -> bool:
    """Session-scoped boolean exposing fast-mode state to tests."""
    return is_fast_mode()


# ===========================================================================
# thinc / pytest-randomly seed-overflow compat shim (2026-04-19)
# ===========================================================================
# Root cause: thinc (spacy/explosion.ai dep) ships a
# ``pytest_randomly.random_seeder`` entry point pointing at
# ``thinc.util.fix_random_seed``. That function calls
# ``numpy.random.seed(seed)`` WITHOUT the ``% 2**32`` clamp that
# pytest-randomly itself applies in its own ``_reseed``. pytest-randomly
# invokes every registered random_seeder entry point with
# ``seed = randomly_seed_option + crc32(nodeid_offset)`` — their sum
# easily exceeds 2**32. Result: random-order test runs cascade
# ``ValueError: Seed must be between 0 and 2**32 - 1`` from thinc,
# which fails the fixture setup and trips pytest's downstream
# ``previous item was not torn down properly`` assertion for every
# subsequent test in the session.
#
# Why module-level not session fixture (2026-04-19 round 11):
# pytest-randomly's ``pytest_runtest_setup`` hook fires BEFORE
# session-scope autouse fixtures activate on the first test of the
# session. By the time our fixture ran, thinc's bare seeder had
# already been invoked with the overflowing seed. Module-level
# patching in conftest.py runs at conftest import time — which is
# BEFORE any pytest hook fires — so by the time pytest-randomly
# resolves its entry points via ``e.load()``, ``thinc.util.fix_random_seed``
# is already our wrapper.
try:
    import thinc.util as _thinc_util  # noqa: E402
    _thinc_original_fix = _thinc_util.fix_random_seed

    def _thinc_clamped_fix_random_seed(seed: int = 0) -> None:
        try:
            return _thinc_original_fix(int(seed) % (2**32))
        except Exception as _seed_err:
            # thinc.util.fix_random_seed transitively calls cupy.random.seed,
            # which crashes hard on boxes where cupy is installed but curand
            # initialisation fails (e.g. driver/runtime mismatch, GPU busy,
            # or missing CUDA at runtime). Pre-fix this killed the entire
            # pytest session before a single test ran. Now we seed Python's
            # random + numpy directly so determinism still holds and skip
            # whatever thinc was trying to do with cupy.
            import random
            import numpy as np
            random.seed(int(seed) % (2**32))
            np.random.seed(int(seed) % (2**32))
            warnings.warn(
                f"thinc.util.fix_random_seed raised {_seed_err!r}; "
                f"fell back to seeding python+numpy directly. cupy random "
                f"state is unset for this session.",
                RuntimeWarning,
                stacklevel=2,
            )

    _thinc_util.fix_random_seed = _thinc_clamped_fix_random_seed

    # If pytest-randomly already populated its cache (e.g. another
    # conftest loaded before us), swap in the wrapper.
    try:
        import pytest_randomly as _pr  # noqa: E402
        if getattr(_pr, "entrypoint_reseeds", None):
            _pr.entrypoint_reseeds = [
                _thinc_clamped_fix_random_seed if r is _thinc_original_fix else r
                for r in _pr.entrypoint_reseeds
            ]
    except Exception:  # pragma: no cover
        pass
except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover
    warnings.warn(
        f"Skipping optional thinc pytest-randomly seed shim because thinc import failed: {exc}",
        RuntimeWarning,
    )
    pass


# pyutilz.system.tqdmu wraps tqdm.tqdm but does NOT honour any global "disable"
# env var, so MRMR's per-pair / "getting pairs MIs" / "Interactions order" bars
# spam the test log unconditionally regardless of any verbose flag we pass.
# Force disable=True by default in the test session; tests that explicitly want
# a visible bar still get one when they pass disable=False themselves.
#
# Same idempotency sentinel guards as the RFECV block above - see that comment
# for the rationale.
try:
    import pyutilz.system as _pus  # noqa: E402
    # Inner module holds the LIVE definition; pyutilz.system re-exports via
    # ``from .system import *`` so the two names are separate bindings to the
    # same function object until we rebind one of them.
    import pyutilz.system.system as _pus_inner  # noqa: E402
    if not getattr(_pus.tqdmu, "_mlframe_test_quieted", False):
        _orig_tqdmu = _pus.tqdmu

        def _quiet_tqdmu(*args, **kwargs):
            kwargs.setdefault("disable", True)
            return _orig_tqdmu(*args, **kwargs)

        _quiet_tqdmu._mlframe_test_quieted = True
        _pus.tqdmu = _quiet_tqdmu
        # Patch the INNER binding too. ``tqdmu_lazy_start`` (also in
        # pyutilz.system.system) does ``bar = tqdmu(iter([]), **kwargs)`` --
        # the bare ``tqdmu`` name resolves against the inner module's own
        # globals, NOT against pyutilz.system. Without this rebind the
        # ``pre_pipeline`` / ``mlframe model`` bars built via lazy_start
        # keep spamming the test log even after the outer rebind. Observed
        # 2026-05-20 on S: in fuzz_3way output. Once the inner tqdmu is
        # the quiet variant, lazy_start automatically inherits the
        # ``disable=True`` default - no separate wrapper needed.
        _pus_inner.tqdmu = _quiet_tqdmu
        # Re-export to any mlframe module that already imported the symbol
        # at top level. Each refresh is idempotent (we just point at the
        # already-patched _quiet_tqdmu).
        for _mod_path in (
            "mlframe.feature_selection.filters.mrmr",
            "mlframe.feature_selection.filters.feature_engineering",
            "mlframe.feature_selection.wrappers._rfecv",
            "mlframe.feature_selection.filters.screen",
            "mlframe.training.core._phase_train_one_target",
        ):
            try:
                import importlib
                _mod = importlib.import_module(_mod_path)
                if hasattr(_mod, "tqdmu"):
                    _mod.tqdmu = _quiet_tqdmu
            except Exception:
                pass
except (ImportError, OSError, RuntimeError):  # pragma: no cover
    pass


def _pytest_randomly_active() -> bool:
    """True iff pytest-randomly is loaded AND not disabled via ``-p no:randomly``.

    pytest-randomly seeds python/numpy/torch RNGs in its own ``pytest_runtest_setup``
    hook BEFORE autouse fixtures run; our reset would then trample that seed and
    collapse cross-test entropy to a single fixed value (defeating the whole point
    of randomly shuffling test ordering). When randomly is active, defer to it.
    """
    try:
        import pytest_randomly  # noqa: F401
    except ImportError:
        return False
    # ``-p no:randomly`` makes the plugin importable but not loaded into the
    # session; pytest_randomly sets ``random_seeder`` entry points only when
    # actively loaded. ``sys.modules`` import is not proof; check the pytest
    # plugin manager via the conftest pytestmark route is the strict signal,
    # but cheap heuristic: presence of the env var pytest-randomly sets when
    # it claims a session.
    import os
    return os.environ.get("PYTEST_RANDOMLY_SEED") is not None


@pytest.fixture(autouse=True)
def _reset_global_rng_state(request):
    """Reset python/numpy/torch global RNGs to seed=0 before each test for legacy tests that
    rely on the global RNG state being deterministic.

    Defers to pytest-randomly when active so the random-ordering plugin's per-test seeds
    aren't trampled (autouse fixtures fire AFTER pytest-randomly's setup hook). A test that
    truly needs the deterministic seed=0 reset under random-ordering should request the
    fixture explicitly or seed locally.
    """
    if _pytest_randomly_active():
        yield
        return
    import random as _random
    import numpy as _np
    _random.seed(0)
    _np.random.seed(0)
    try:
        import torch as _torch
        _torch.manual_seed(0)
    except ImportError:
        pass
    yield


@pytest.fixture(autouse=True)
def cleanup_memory(request):
    """Clean up memory after each test to prevent OOM issues.

    Fast path: when the test does NOT carry ``uses_matplotlib`` / ``uses_torch``
    markers, skip psutil sampling + pyarrow/matplotlib/torch cleanup and just
    gc.collect() once. The heavy cleanup paths only matter for tests that
    exercise those libraries; running them after every micro-test was costly.
    """
    import os

    has_heavy_marker = (
        request.node.get_closest_marker("uses_matplotlib") is not None
        or request.node.get_closest_marker("uses_torch") is not None
    )
    if not has_heavy_marker:
        yield
        gc.collect()
        return

    is_main_process = os.environ.get('PYTEST_CURRENT_TEST') is not None

    if psutil is not None:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        if is_main_process:
            print(f"\n[MEM] Before: {mem_before:.0f} MB")
    else:
        process = None
        mem_before = 0.0

    yield

    # pl.disable_string_cache() removed: per polars 1.x semantics it is a no-op
    # (the global cat StringCache is reset implicitly when no StringCache context
    # is active). Use pl.Enum built from train+val union for deterministic
    # category mapping across fixtures; see memory reference_polars_global_string_cache.

    # PyArrow memory pool: polars to pandas conversions allocate Categorical
    # arrays into pyarrow's default memory pool, which keeps used pages
    # cached across calls. Across fuzz combos with different cat-pool
    # values (e.g. seed=2024 c0009 unicode → c0060 empty), released
    # arrays' bytes can be re-served to a later combo's cat-dictionary
    # allocation, producing pd.Categorical whose ``categories`` array
    # silently contains leaked strings from the previous combo. Force
    # pyarrow to release pool pages so no allocation re-uses bytes from
    # a stale Categorical (seed=2024 c0060 flake).
    try:
        import pyarrow as _pa_cleanup
        try:
            _pa_cleanup.default_memory_pool().release_unused()
        except Exception:
            pass
    except ImportError:
        pass

    # 2026-04-29: matplotlib figure cleanup. Plots accumulate across the
    # fuzz suite (each report_model_perf builds 5+ figures), and the
    # Agg backend's allocator eventually trips ``MemoryError: bad
    # allocation`` deep in C++. ``plt.close('all')`` releases the
    # backend buffers so subsequent combos start clean. Surfaced
    # 3-way fuzz c0024 (29 mins into a 60-combo sweep).
    try:
        import matplotlib.pyplot as _plt
        _plt.close("all")
    except Exception:
        pass

    gc.collect()

    # Clear GPU memory. Distributed process-group teardown is the responsibility
    # of the multigpu tests themselves — tearing it down per test here would
    # race with tests that legitimately share a process group across fixtures.
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    except Exception:
        pass

    if psutil is not None and process is not None:
        mem_after = process.memory_info().rss / 1024 / 1024
        if is_main_process:
            print(f"[MEM] After: {mem_after:.0f} MB (delta: {mem_after - mem_before:+.0f} MB)")


@pytest.fixture(autouse=True)
def suppress_convergence_warnings():
    """Suppress convergence warnings during tests."""
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
        warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
        warnings.filterwarnings("ignore", message=".*Objective did not converge.*")
        yield
