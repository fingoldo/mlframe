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
# Stream test output line-by-line: by default Python buffers stdout under non-tty pipes (CI tail commands, IDE log panels, tee). Unbuffered output makes `pytest -s` actually-streaming so long-running tests show progress instead of opaque mid-test silence. Operators forcing buffered mode can pre-set the env var to 0.
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# Make CuBLAS pickable under ``torch.use_deterministic_algorithms(True)``. Lightning's
# ``Trainer(deterministic="warn")`` -- used by the recurrent training path -- requests
# deterministic algorithms but PyTorch ≥ CUDA 10.2 needs ``CUBLAS_WORKSPACE_CONFIG`` set
# BEFORE CuBLAS loads, otherwise every matmul fires a "Deterministic behavior was enabled
# but this operation is not deterministic" UserWarning per backward pass. Set the
# documented ``:4096:8`` workspace size (PyTorch CONTRIBUTING.md + NVIDIA cuBLAS docs).
# Pre-set by operator wins (the alternative ``:16:8`` slot trades workspace for memory).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Claim torch's CUDA (cuBLAS) context BEFORE any test imports mlframe (which pulls cupy + numba.cuda).
# On some GPUs (observed: RTX 500 Ada laptop, CUDA 12.8) letting cupy AND numba.cuda establish the
# device context first leaves torch's cublasSgemm raising CUBLAS_STATUS_INVALID_VALUE on EVERY matmul
# for the rest of the process -- a one-way corruption, so a later warm-up cannot recover it. Done at
# conftest import time (before collection imports mlframe) a tiny matmul claims torch's cublas handle
# first and keeps it healthy regardless of later cupy/numba init order. Guarded so CPU-only runs
# (CUDA_VISIBLE_DEVICES="") and no-torch / no-GPU hosts are unaffected; failures are non-fatal.
if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    try:
        import torch as _torch_warm
        if _torch_warm.cuda.is_available():
            _wm = _torch_warm.randn(8, 8, device="cuda")
            _ = _wm @ _wm
            _torch_warm.cuda.synchronize()
            del _wm
    except Exception:
        pass
# Headless tests MUST use the non-GUI Agg backend. The DSL render path renders each chart in a ThreadPoolExecutor
# worker (render_and_save); under an interactive backend (TkAgg on a desktop) Tk calls from a non-main thread hang
# ("main thread is not in main loop"), tripping the 60s render-thread timeout so EVERY chart is silently dropped --
# which made suite-render tests (e.g. test_ensembling_chart_artifacts) flip pass/fail with the ambient backend.
os.environ.setdefault("MPLBACKEND", "Agg")

# Isolate the per-host kernel_tuning_cache from the developer's real ``~/.pyutilz`` cache for the whole test session.
# Cache-dependent dispatch tests (per_member / dtw / cat-perm / batch_pair_mi backend choice) assert the un-tuned
# FALLBACK behaviour; on a dev box whose real cache was populated by a prior production sweep they would instead read
# the swept ``backend_choice`` (e.g. numpy-wins) and fail. Routing to a per-process tmp dir also stops the suite from
# polluting the real cache. ``load_or_create()`` binds the dir at first construction, so the companion
# ``_reset_kernel_tuning_singleton`` autouse fixture resets the singleton per test so a test's own override takes effect.
import tempfile as _tempfile  # noqa: E402
_kt_worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
os.environ["PYUTILZ_KERNEL_CACHE_DIR"] = os.path.join(_tempfile.gettempdir(), f"mlframe_test_kt_cache_{os.getpid()}_{_kt_worker}")
# Skip the on-miss auto-sweep during tests: with the cache isolated to an empty tmp dir, every cache-dependent dispatch
# would otherwise trigger a real (multi-second) kernel sweep on first call -- which both slows the suite and times out
# fallback-path tests. The caller's measurement-backed fallback is what those tests assert anyway. Dedicated sweep tests
# call ``run_*_sweep`` / ``ensure_*_tuning`` directly (unaffected by this gate).
os.environ.setdefault("PYUTILZ_KERNEL_DISABLE_SWEEP", "1")

# Auto-detect CUDA_PATH so cupy's ``_environment`` doesn't warn "CUDA path
# could not be detected" at import time. CuPy needs CUDA_PATH or CUDA_HOME
# to locate the toolkit DLLs / shared libs (libcudart, libcublas, libcudnn,
# nvrtc); without it the import succeeds but ``cuda.runtime.getDeviceCount()``
# fails -- every gpu-marker test then skips with "no CUDA" even though the
# device + drivers are present. Pre-set by operator always wins. Probe the
# documented install roots in order; first hit becomes ``CUDA_PATH``.
if not (os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")):
    import glob as _cuda_glob_probe
    _cuda_candidates = [
        # Windows installer default
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        # Common operator-relocated install drives (matches the testbed's
        # ``D:\CUDA\v12.9`` layout)
        r"D:\CUDA",
        r"E:\CUDA",
        # POSIX standard locations
        "/usr/local/cuda",
        "/opt/cuda",
    ]
    _cuda_found = None
    for _root in _cuda_candidates:
        if not os.path.isdir(_root):
            continue
        # Direct hit: the candidate IS a versioned CUDA install
        # (``D:\CUDA\v12.9`` already contains ``bin/`` + ``include/``).
        if os.path.isdir(os.path.join(_root, "bin")):
            _cuda_found = _root
            break
        # Two-level hit: ``C:\Program Files\...\CUDA\v12.9``. Pick the
        # highest-versioned subdir so multiple installs prefer the newest.
        _versions = sorted(_cuda_glob_probe.glob(os.path.join(_root, "v*")), reverse=True)
        for _ver in _versions:
            if os.path.isdir(os.path.join(_ver, "bin")):
                _cuda_found = _ver
                break
        if _cuda_found:
            break
    if _cuda_found:
        os.environ["CUDA_PATH"] = _cuda_found
        os.environ.setdefault("CUDA_HOME", _cuda_found)
    del _cuda_glob_probe, _cuda_candidates, _cuda_found
    try:
        del _root, _versions, _ver
    except NameError:
        pass

import pytest


# ---------------------------------------------------------------------------
# CatBoost: per-pytest-xdist-worker ``train_dir`` so the widget's background
# thread can't trip ``PermissionError`` on a shared ``catboost_info`` write.
#
# Mechanism: CatBoost's ``_get_train_dir`` reads ``params.get('train_dir',
# 'catboost_info')`` -- defaulting to a CWD-relative ``catboost_info``. Under
# ``pytest-xdist`` all workers share the project directory as CWD, so they
# fight over the same ``catboost_info/catboost_training.json`` file. The
# widget's background thread (started when ``plot=True`` reaches a
# ``plot_wrapper``) reads that file every 1 second; if a sibling worker
# holds the write lock the read raises ``PermissionError`` and the
# unhandled-thread-exception warning surfaces in the test summary.
#
# Surgical autouse fixture that monkeypatches ``catboost.core._get_train_dir``
# to default to a per-worker tmp directory when the caller didn't pass
# ``train_dir`` explicitly. Worker isolation = no shared file = no race. No
# made-up env vars (``CATBOOST_NO_WIDGET`` does not exist in CatBoost), no
# touching prod source.
#
# Scoped to ``function`` (not ``session``) so each test's tmp_path stays
# unique -- shared session-wide train_dir would still concentrate writes
# in one directory per worker, just shifted off the project root.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _catboost_per_worker_train_dir(tmp_path, monkeypatch):
    try:
        from catboost import core as _cb_core
    except ImportError:
        return  # CatBoost not installed -- nothing to isolate.
    _orig_get_train_dir = _cb_core._get_train_dir

    def _patched_get_train_dir(params):
        if not isinstance(params, dict) or "train_dir" in params:
            return _orig_get_train_dir(params)
        per_worker = tmp_path / "catboost_info"
        per_worker.mkdir(parents=True, exist_ok=True)
        return str(per_worker)

    monkeypatch.setattr(_cb_core, "_get_train_dir", _patched_get_train_dir)

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


def fast_n_estimators(full: int, *, fast: int = 40, floor: int = 1) -> int:
    """Shrink a boosting/ensemble iteration budget in fast mode, else return ``full``.

    The `--fast` / `fast_subset` machinery only trims parametrize *variants*; it
    does nothing for the model dimension, so business-value tests that fit real
    LGBM/XGB/CatBoost models with ``n_estimators``/``iterations`` in the 200-300
    range still pay full training cost under `--fast`. Wrap that budget in
    ``fast_n_estimators(300)`` to drop it (default 40) in fast mode while keeping
    the exact same code path exercised. Outside fast mode it is an identity.

    ``fast`` is clamped to ``[floor, full]`` so callers passing a budget already
    smaller than the fast target keep their (smaller) value.
    """
    if not is_fast_mode():
        return full
    return max(floor, min(fast, full))


def running_under_xdist() -> bool:
    """True when executing inside a pytest-xdist worker (i.e. the full ``-n`` parallel run)."""
    return bool(os.environ.get("PYTEST_XDIST_WORKER"))


def perf_time_budget(base_seconds: float, *, xdist_factor: float = 4.0) -> float:
    """Wall-clock time budgets are unreliable under ``-n`` parallel contention: a 2h full-suite run can starve any one
    worker for seconds, so a quiet-box budget that is correct standalone flakes under load. Multiply the budget when
    running under xdist so it still trips on a gross (order-of-magnitude) regression without flaking on transient
    scheduler stalls; standalone keeps the tight budget. Use for absolute ``elapsed <= budget`` assertions."""
    return base_seconds * xdist_factor if running_under_xdist() else base_seconds


def perf_speedup_floor(base_ratio: float, *, xdist_factor: float = 0.6) -> float:
    """Speedup-ratio floors compress under ``-n`` contention. A ratio is measured from two arms run back-to-back in the
    same process, so contention hits both and the ratio is more load-robust than an absolute time -- but small absolute
    times still add noise. Relax the floor under xdist so a real win still passes; standalone keeps the full ratio.
    Never drops below 1.0x: a speedup test must still prove the fast path is at least not slower under load."""
    if not running_under_xdist():
        return base_ratio
    return max(1.0, base_ratio * xdist_factor)


@pytest.fixture(autouse=True)
def _reset_kernel_tuning_singleton():
    """``KernelTuningCache.load_or_create()`` returns a process singleton that binds its cache dir at first
    construction, so a per-test ``PYUTILZ_KERNEL_CACHE_DIR`` override (or the session-default isolation set at the top
    of this file) is otherwise ignored once any earlier test has constructed it. Reset the module singleton around each
    test so the isolated dir actually takes effect and cache-dependent dispatch tests see a clean per-host cache."""
    try:
        from pyutilz.performance.kernel_tuning import cache as _ktc
    except Exception:
        _ktc = None
    if _ktc is not None:
        _ktc._DEFAULT_INSTANCE = None
    yield
    if _ktc is not None:
        try:
            _ktc._DEFAULT_INSTANCE = None
        except Exception:
            pass


def require_polars_ds():
    """Skip the calling test when ``polars-ds`` is not importable. Single source of truth so the 8x duplicated
    `pytest.importorskip("polars_ds")` calls in `tests/inference/`, `tests/training/test_pipeline.py`, and
    `tests/training/test_polarsds_pipeline_json_proxy.py` can collapse to one helper invocation. Keeps the
    importorskip reason consistent and centralises the gate when polars-ds gating changes."""
    return pytest.importorskip(
        "polars_ds",
        reason="polars-ds optional extra not installed; tests exercising the polars-native PreprocessingBackendConfig fastpath are skipped",
    )


def pytest_sessionfinish(session, exitstatus):
    """Release persistent GPU buffers (the friend-graph / MI ``_GPU_POOL`` + cupy memory pools) while the
    CUDA context is still alive, in a controlled order. Freeing them during interpreter atexit -- alongside
    torch's and numba.cuda's own CUDA teardown -- has heap-corrupted the worker (0xc0000374) on this
    multi-CUDA-library host (cupy-cu12 + numba-cuda-12.9 + torch-cu128). Best-effort; never raises."""
    try:
        from mlframe.feature_selection.filters import gpu as _gpu
        _gpu._GPU_POOL.free()
    except Exception:
        pass
    try:
        import cupy as _cp
        _cp.get_default_memory_pool().free_all_blocks()
        _cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


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
    # Markers ``slow_only`` / ``no_xdist`` are registered in pyproject.toml
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
    # B2#38 marker: opts a test OUT of the ``suppress_convergence_warnings`` autouse filter so
    # ``pytest.warns(ConvergenceWarning)`` can catch the warning instead of having it pre-filtered.
    config.addinivalue_line(
        "markers",
        "expects_convergence_warning: opt out of the ``suppress_convergence_warnings`` autouse filter; "
        "use when a test asserts the warning via ``pytest.warns(ConvergenceWarning)``.",
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
    # ``config.option.dist`` reads "no" on each xdist WORKER (only the controller sees the -n
    # value), so check the per-worker PYTEST_XDIST_WORKER env too -- otherwise no_xdist items
    # silently run (and native-crash) on workers.
    _dist_opt = getattr(config.option, "dist", "no")
    if _dist_opt != "no" or running_under_xdist():
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
#
# Upstream status (audit checkpoint 2026-05-26): no issue filed against thinc
# (explosion/thinc) for the missing ``% 2**32`` clamp in ``fix_random_seed``;
# this shim remains the in-repo workaround. Drop the shim once upstream's
# ``fix_random_seed`` accepts arbitrary 64-bit seeds without raising (verify
# by removing this block and running the suite with ``--randomly-seed=large``
# values; if no ``ValueError: Seed must be between 0 and 2**32 - 1`` cascades,
# the upstream fix has landed).
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
except ImportError:  # pragma: no cover
    # thinc is an optional dependency; its absence is the normal case here and warrants no warning.
    pass
except (OSError, RuntimeError) as exc:  # pragma: no cover
    # thinc is present but its import side-effects (cupy/CUDA init) failed: surface as a config notice, not a numeric RuntimeWarning.
    warnings.warn(
        f"Skipping optional thinc pytest-randomly seed shim because thinc import failed: {exc}",
        UserWarning,
    )


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
            "mlframe.feature_selection.wrappers.rfecv",
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
        # ``torch.manual_seed`` calls ``torch.cuda.manual_seed_all`` internally
        # when CUDA is built-in. A prior test that exhausted GPU memory or
        # crashed mid-kernel can leave the CUDA context corrupted -- the next
        # autouse-fixture reseed then raises ``CUDA error: an illegal memory
        # access was encountered`` and POISONS every downstream test's setup
        # phase. The fixture's job is deterministic CPU-side reseed; broaden
        # the catch so a corrupted CUDA context doesn't cascade into ERROR
        # for unrelated tests. The CPU seed (``_random.seed`` + ``_np.random.seed``
        # above) already fired.
        _torch.manual_seed(0)
    except (ImportError, RuntimeError) as _torch_seed_err:
        # ImportError: torch not installed (CI shards without neural extras).
        # RuntimeError: CUDA context corrupted by a prior test -- the CPU
        # seed branch already ran above; downstream CPU-only tests proceed.
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
    # [MEM] print is opt-in via MLFRAME_TEST_MEM_LOG=1; default off so CI scrollback isn't filled with one line per heavy-marker test.
    _mem_log_enabled = os.environ.get("MLFRAME_TEST_MEM_LOG", "").lower() in ("1", "true", "yes")

    if psutil is not None:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        if is_main_process and _mem_log_enabled:
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
        if is_main_process and _mem_log_enabled:
            print(f"[MEM] After: {mem_after:.0f} MB (delta: {mem_after - mem_before:+.0f} MB)")


@pytest.fixture(autouse=True)
def suppress_convergence_warnings(request):
    """Suppress sklearn ConvergenceWarning + the lbfgs / "Objective did not converge" pair during tests.

    B2#38 opt-out: tests that need to assert the warning via ``pytest.warns(ConvergenceWarning)`` must carry the
    ``@pytest.mark.expects_convergence_warning`` marker so this filter is bypassed. The marker is registered in
    ``pytest_configure`` below; absent the marker the catch-all filter applies (default behaviour).
    """
    from sklearn.exceptions import ConvergenceWarning

    if request.node.get_closest_marker("expects_convergence_warning") is not None:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
        warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
        warnings.filterwarnings("ignore", message=".*Objective did not converge.*")
        yield


@pytest.fixture(scope="session", autouse=True)
def _purge_stale_test_caches():
    """Wipe stale pytest / numba on-disk caches older than 7 days so they don't bleed across sessions.

    Empirically observed: stale ``.pytest_cache/v/cache/nodeids`` from a different pytest binary version makes the
    next session crash at collection ("KeyError: 'lastfailed'") or silently lose stepwise / lf state; stale
    ``.nbc/`` numba dirs from a pre-fastmath kernel rebuild make the next session run the wrong cached AOT module
    until manually wiped. Auto-purge anything older than 7 days at session start so the typical week-old artefact
    doesn't trip the next dev / CI session. Recent caches (under 7 days) stay - that's the active working set.

    Honour ``MLFRAME_KEEP_TEST_CACHES=1`` for the rare debug scenario where the user wants to keep stale caches in
    place (e.g. bisecting a cache-invalidation bug). Failures during purge are non-fatal: a missing dir / a
    permission-denied on a single file should not block the session."""
    if os.environ.get("MLFRAME_KEEP_TEST_CACHES", "").strip() in ("1", "true", "True"):
        yield
        return
    import time
    from pathlib import Path
    cutoff = time.time() - 7 * 24 * 3600
    repo_root = Path(__file__).resolve().parent.parent
    cache_dirs = (
        repo_root / ".pytest_cache",
        repo_root / ".nbc",
        repo_root / ".numba_cache",
        repo_root / "__pycache__" / "test_stragglers",
    )
    for cache_dir in cache_dirs:
        if not cache_dir.exists():
            continue
        try:
            mtime = cache_dir.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        try:
            import shutil
            shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass
    yield
