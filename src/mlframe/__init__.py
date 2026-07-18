"""mlframe -- production-grade tabular ML toolkit (classification, regression, ranking, quantile).

Public API convention: deep-import from subpackages, not from this top-level module.
The top-level ``mlframe`` namespace intentionally exports ONLY ``__version__`` and the
``mlframe.config`` enums; everything else is reached via its subpackage path:

    from mlframe.training import train_mlframe_models_suite
    from mlframe.training.composite import CompositeTargetEstimator
    from mlframe.feature_selection import MRMR, RFECV
    from mlframe.metrics.core import expected_calibration_error
    from mlframe.calibration.policy import pick_best_calibrator
    from mlframe.models.ensembling import score_ensemble
    from mlframe.inference.predict import predict_from_models
    from mlframe.evaluation.bootstrap import bootstrap_metric, delong_test

Reaching into double-underscore private submodules (``mlframe.training.core._*``,
``mlframe.feature_selection.wrappers._*``) is reserved for sibling-module use; tests
and out-of-tree consumers are expected to use the public subpackage surface. The
underscore convention is enforced by the meta-test ``tests/test_meta/
test_no_underscore_imports_cross_package.py``.

Why deep-import rather than top-level re-export: the codebase carries ~1k entry-point
symbols spread across 15 subpackages. Re-exporting every one through ``mlframe.*``
would (a) bloat ``mlframe`` import time by eagerly resolving every subpackage,
(b) introduce circular-import risk between subpackages that already coordinate via
``training/__init__.py``-style lazy ``__getattr__`` dispatch, and (c) duplicate the
authoritative ``__all__`` lists already maintained at the subpackage level. The CI
smoke step (``.github/workflows/ci.yml`` ``build.smoke``) confirms the deep-import
pattern works on a built wheel by importing ``mlframe.training``, ``mlframe.metrics.core``,
``mlframe.models.ensembling``, and ``mlframe.calibration.policy`` directly.
"""

from __future__ import annotations


def _autoconfigure_cuda_home() -> None:
    """Point CUDA_HOME/CUDA_PATH at the pip-installed nvidia NVVM when nothing else has.

    numba.cuda needs ``CUDA_HOME``/``CUDA_PATH`` to locate ``nvvm`` + ``libdevice``. On a host that
    has the ``nvidia-cuda-nvcc`` pip wheel (which bundles a working NVVM) but no system CUDA toolkit
    and no CUDA env var, numba silently reports ``cuda.is_available() == False`` and every GPU kernel
    falls back to CPU even though the GPU + driver + cupy all work. This sets the env var to the pip
    NVVM so the GPU path is used out of the box.

    Safety -- never override a real install: skips entirely if CUDA_HOME / CUDA_PATH (or any versioned
    ``CUDA_PATH_V*`` that the NVIDIA system installer sets) is already present, so a machine with a
    proper CUDA toolkit is left untouched. Cheap filesystem probe only (no numba/cupy import); never
    raises. Opt-out: ``MLFRAME_NO_CUDA_AUTOCONFIG=1``.
    """
    import os
    if os.environ.get("MLFRAME_NO_CUDA_AUTOCONFIG", "").strip() not in ("", "0", "false", "False"):
        return

    import pathlib

    def _dir_has_cudart(p: str) -> bool:
        """True if the CUDA runtime library (cudart) is present under p, in any of the layouts NVIDIA ships."""
        d = pathlib.Path(p)
        _patterns = ("bin/cudart64_*.dll", "lib64/libcudart*", "lib/libcudart*")
        return any(next(d.glob(_pat), None) is not None for _pat in _patterns)

    def _dir_has_nvvm(p: str) -> bool:
        """True if the NVVM codegen library is present under p, in any of the layouts NVIDIA ships."""
        d = pathlib.Path(p)
        _patterns = ("nvvm/bin/nvvm*.dll", "nvvm/lib64/libnvvm*")
        return any(next(d.glob(_pat), None) is not None for _pat in _patterns)

    def _find_complete_toolkit() -> "str | None":
        """Highest-versioned CUDA_PATH_V* system toolkit that has BOTH nvvm (codegen) and cudart (runtime,
        needed by numba's get_supported_ccs) -- a toolkit missing either cannot compile+run numba kernels."""
        cands = [v for k, v in os.environ.items() if k.startswith("CUDA_PATH_V") and v]
        for p in sorted(cands, reverse=True):
            if _dir_has_cudart(p) and _dir_has_nvvm(p):
                return p
        return None

    # Repair an INCOMPLETE CUDA_PATH/CUDA_HOME: a prior auto-config (or a pip-only setup) may have pointed
    # it at the nvidia-cuda-nvcc wheel, which bundles nvvm+libdevice but NOT cudart. numba's CUDA_HOME-
    # anchored loader then fails to find cudart -> get_supported_ccs() returns () -> every kernel raises
    # NvvmSupportError "No supported GPU compute capabilities found". If a complete system toolkit is
    # present, redirect to it so kernels actually compile (not just is_available()==True).
    _cur = os.environ.get("CUDA_HOME")
    if not _cur:
        _cur = os.environ.get("CUDA_PATH")
    if _cur and not _dir_has_cudart(_cur):
        _complete = _find_complete_toolkit()
        if _complete:
            os.environ["CUDA_HOME"] = _complete
            os.environ["CUDA_PATH"] = _complete
            import logging
            logging.getLogger(__name__).info(
                "mlframe: redirected CUDA_HOME/CUDA_PATH from an nvvm-only dir (%s, no cudart) to the "
                "complete CUDA toolkit %s so numba.cuda kernels compile; set MLFRAME_NO_CUDA_AUTOCONFIG=1 to skip.",
                _cur, _complete,
            )
        return

    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    if any(k.startswith("CUDA_PATH_V") for k in os.environ):
        return
    try:
        import sysconfig
        import pathlib

        roots = {sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")}
        for root in filter(None, roots):
            nvvm = pathlib.Path(root) / "nvidia" / "cuda_nvcc" / "nvvm"
            _dll_patterns = ("bin/nvvm*.dll", "lib64/libnvvm*")
            has_dll = any(next(nvvm.glob(_pat), None) is not None for _pat in _dll_patterns)
            has_libdevice = next(nvvm.glob("libdevice/libdevice*.bc"), None) is not None
            if has_dll and has_libdevice:
                cuda_nvcc = str(nvvm.parent)
                os.environ["CUDA_HOME"] = cuda_nvcc
                os.environ.setdefault("CUDA_PATH", cuda_nvcc)
                import logging

                logging.getLogger(__name__).info(
                    "mlframe: set CUDA_HOME/CUDA_PATH to the pip-installed nvidia NVVM (%s) so numba.cuda "
                    "GPU kernels are enabled; set MLFRAME_NO_CUDA_AUTOCONFIG=1 to skip.", cuda_nvcc,
                )
                return
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        logging.getLogger(__name__).debug("suppressed in __init__.py:117: %s", e)
        pass


gpu_disable_errors: list = []
"""Diagnostic collection: reasons ``_disable_broken_cupy`` poisoned ``sys.modules['cupy']``
(0 or 1 entries -- the guard runs once per process). Empty means cupy is unusable/absent
without needing the poison guard, or usable, or the guard never ran; a non-empty entry
lets a caller/report distinguish "cupy absent" from "cupy present but broken"."""


def _disable_broken_cupy() -> None:
    """Guard against the cupy softlink-recursion bug on broken CUDA installs.

    On a host where cupy imports cleanly but its NVRTC / cublas DLLs are
    mismatched or renamed (observed 2026-05-20 on S: after the user renamed
    cublas64_11.dll to unblock torch), the first cupy reduction kernel
    triggers an internal ``cupy_backends.cuda._softlink.SoftLink.__init__``
    that re-enters its own exception handler and overflows the stack with
    ``RecursionError: maximum recursion depth exceeded``. This propagates
    past most ``except Exception`` blocks too late to recover, and the
    error is non-deterministically interleaved by xdist workers.

    Strategy: probe once at mlframe import. If the probe fails for ANY
    reason (ImportError, RecursionError, CUDARuntimeError, ...), poison
    ``sys.modules['cupy']`` so every subsequent ``import cupy as cp`` in
    mlframe (22 call sites) raises ``ImportError`` and the existing
    ``except ImportError`` fallback paths route to CPU.

    Opt-out: set ``MLFRAME_KEEP_BROKEN_CUPY=1`` to skip this guard (for
    debugging cupy itself).
    """
    import os
    if os.environ.get("MLFRAME_KEEP_BROKEN_CUPY", "").strip() not in ("", "0", "false", "False"):
        return
    import sys
    try:
        import cupy as _cp
    except ImportError:
        # cupy not installed - clean state, nothing to disable.
        return
    except Exception as _imp_exc:
        # cupy IS installed but its native deps fail to load (e.g. a broken
        # CUDA toolkit where ``cublasLt64_*.dll`` raises an OSError/RuntimeError
        # at import via the pathfinder loader, NOT an ImportError). Treat this
        # exactly like the failed-probe path below: poison the import so every
        # later ``import cupy`` returns None instead of re-raising the DLL
        # error and taking down the whole process at ``import mlframe`` time.
        import logging
        import sys as _sys_imp
        logging.getLogger(__name__).warning(
            "mlframe: cupy import raised %s: %s. Disabling cupy for this " "process. Set MLFRAME_KEEP_BROKEN_CUPY=1 to skip.",
            type(_imp_exc).__name__,
            _imp_exc,
        )
        gpu_disable_errors.append(f"cupy import raised {type(_imp_exc).__name__}: {_imp_exc}")
        _sys_imp.modules["cupy"] = None  # type: ignore[assignment]
        return
    try:
        _ = _cp.asarray([1.0], dtype=_cp.float32).sum().item()
        # Probe succeeded: cupy is usable. Leave sys.modules alone.
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "mlframe: cupy is installed but unusable (NVRTC probe failed: "
            "%s: %s). Disabling cupy for this process to avoid the softlink "
            "RecursionError cascade. Set MLFRAME_KEEP_BROKEN_CUPY=1 to skip.",
            type(exc).__name__, exc,
        )
        gpu_disable_errors.append(f"cupy NVRTC probe failed: {type(exc).__name__}: {exc}")
        gpu_disable_errors.append(f"cupy NVRTC probe raised {type(exc).__name__}: {exc}")
        # Poison the import: ``sys.modules["cupy"] = None`` makes every subsequent ``import cupy``
        # raise ``ImportError`` immediately. CPython's import machinery treats a ``None`` entry as
        # a negative-cache marker -- documented behaviour since Python 2.x, still maintained in 3.12+
        # and 3.13. The data-model deprecation note about "non-module objects in sys.modules" applies
        # to arbitrary objects, not the explicit ``None`` sentinel; the CPython ``_find_and_load``
        # path checks for ``None`` BEFORE the module-type assertion (see ``Lib/importlib/_bootstrap.py``
        # in 3.12 / 3.13 source: ``if spec is _NEEDS_LOADING and name in sys.modules: ... if module is
        # None: raise ImportError(...)``). mlframe's 22 GPU call sites already handle ``ImportError`` as
        # a CPU fallback signal, so the ``None`` poisoning is the cheapest primitive available --
        # cheaper than installing a ``_SoftBan`` proxy module that would intercept attribute access
        # (the cost of which would be paid on every ``cupy.<thing>`` reference even where the caller
        # already wrapped in ``try / except ImportError``).
        sys.modules["cupy"] = None  # type: ignore[assignment]


def _patch_colorama_reinit_storm() -> None:
    """Make ``colorama.init()`` idempotent to stop a per-warning win32-API storm.

    Root cause (found via cProfile on a wellbore-100k GPU-strict MRMR fit, 2026-07-17):
    ``numba.cuda``'s dispatcher constructs a ``NumbaPerformanceWarning`` on EVERY kernel launch whose
    grid has fewer than 128 blocks (``numba/cuda/dispatcher.py``'s launch-configuration check) -- an
    expected, frequent occurrence across mlframe's many small-grid resident GPU kernels (one warp per
    reduction, one thread per tiny candidate batch, etc.), not a bug to "fix" by padding grids. Building
    that warning's message goes through ``numba.core.errors.HighlightColorScheme._markup``, which opens
    a **fresh** ``colorama.ColorShell()`` (``numba/core/errors.py``) on every call; ``ColorShell.__init__``
    unconditionally calls ``colorama.init()``, which unconditionally re-wraps ``sys.stdout``/``sys.stderr``
    and re-probes the Windows console API (``colorama.win32.winapi_test`` / ``_winapi_test``) -- colorama
    itself has NO guard against re-wrapping an already-wrapped stream. Measured: ~50,000 warning
    constructions in one fit, ~7.6s cumulative just in the colorama re-init path (confirmed via
    ``pstats.Stats.print_callers`` tracing ``winapi_test`` -> ``ansitowin32.__init__`` -> ``initialise.init``
    -> ``numba.core.errors.NumbaWarning.__init__``, all ~50k deep). Filtering the warning via
    ``warnings.simplefilter("ignore", NumbaPerformanceWarning)`` (the pattern already used in a few mlframe
    GPU modules) does NOT fix this: the expensive object is constructed as a plain Python expression
    BEFORE ``warnings.warn()`` is even called, so the filter can only suppress the eventual print, not the
    construction cost that already ran.

    Fix: wrap ``colorama.initialise.init`` so only the FIRST call with a given (autoreset, convert, strip,
    wrap) argument signature runs colorama's real logic; every later call with that same signature is a
    no-op. Terminal capabilities do not change mid-process, so re-probing them on every warning is pure
    waste -- the first call still runs colorama's real logic (correct colorized output is unaffected).
    Tried tracking ``id(sys.stdout)`` as the "already wrapped" signal first: rejected, because colorama's
    OWN ``init()`` reassigns ``sys.stdout`` to a FRESH wrapper object on every call (nesting a new
    ``StreamWrapper`` around whatever ``sys.stdout`` currently is) -- so the very act of running the
    real init changes the signal that decision was trying to read, and an id-based check placed either
    before or after the real call can never observe two consecutive calls as "the same". A plain per-
    signature call-once guard sidesteps that self-defeating dependency entirely.

    Patched in TWO places, both required: ``colorama.initialise.init`` itself (for any caller that does
    ``import colorama.initialise; colorama.initialise.init()`` / ``colorama.init()``), AND
    ``numba.core.errors.init`` -- ``numba/core/errors.py`` does ``from colorama import init, ...`` at
    MODULE LOAD TIME, which binds its OWN independent name to the ORIGINAL function object; rebinding
    ``colorama.initialise.init`` afterwards does NOT change what ``numba.core.errors.ColorShell.__init__``
    calls (confirmed via cProfile: patching only ``colorama.initialise.init`` showed zero change --
    ``numba.core.errors.init`` must be rebound directly, the actual call site). Best-effort: any failure
    (colorama/numba absent, a future version restructuring these modules) leaves behavior untouched.
    Opt-out: ``MLFRAME_KEEP_COLORAMA_REINIT=1``."""
    import logging
    import os
    if os.environ.get("MLFRAME_KEEP_COLORAMA_REINIT", "").strip() not in ("", "0", "false", "False"):
        return
    try:
        import colorama.initialise as _ci
    except Exception as e:
        logging.getLogger(__name__).debug("colorama unavailable, skipping reinit-storm patch: %s", e)
        return
    _orig_init = getattr(_ci, "init", None)
    if _orig_init is None or getattr(_orig_init, "_mlframe_idempotent", False):
        return

    import functools

    _seen_signatures: set = set()

    @functools.wraps(_orig_init)
    def _idempotent_init(autoreset=False, convert=None, strip=None, wrap=True):
        """Run colorama's real init() only on the first call for a given argument signature; later
        calls with the same signature are a no-op (terminal capabilities do not change mid-process)."""
        sig = (autoreset, convert, strip, wrap)
        if sig in _seen_signatures:
            return
        _orig_init(autoreset=autoreset, convert=convert, strip=strip, wrap=wrap)
        _seen_signatures.add(sig)

    _idempotent_init._mlframe_idempotent = True  # type: ignore[attr-defined]
    _ci.init = _idempotent_init
    try:
        import numba.core.errors as _ne
        if getattr(_ne, "init", None) is _orig_init:
            _ne.init = _idempotent_init
    except Exception as e:  # nosec B110 - best-effort: numba absent or restructured its colorama import
        logging.getLogger(__name__).debug("skipping numba colorama-init patch: numba absent or restructured (%s)", e)


_gpu_runtime_configured = False


def _ensure_gpu_runtime_configured() -> None:
    """Run the CUDA env autoconfig + broken-cupy guard + colorama-reinit patch once, on first GPU-path use.

    These routines mutate ``os.environ`` (CUDA_HOME/CUDA_PATH), import cupy + run a CUDA reduction kernel,
    and monkeypatch ``colorama.initialise.init``. Doing that at ``import mlframe`` time is hostile to the
    majority of users who never touch a GPU path: it probes the GPU and rewrites their environment on every
    import. All three are therefore deferred to the first time a GPU dispatcher asks ``is_cuda_available()``
    (see the wrapper installed below), so a plain ``import mlframe`` neither mutates the environment nor
    imports cupy nor touches colorama. The opt-out env vars (MLFRAME_NO_CUDA_AUTOCONFIG /
    MLFRAME_KEEP_BROKEN_CUPY / MLFRAME_KEEP_COLORAMA_REINIT) are still honoured inside each routine.
    """
    global _gpu_runtime_configured
    if _gpu_runtime_configured:
        return
    _gpu_runtime_configured = True
    _autoconfigure_cuda_home()
    _disable_broken_cupy()
    _patch_colorama_reinit_storm()


def _install_gpu_runtime_lazy_trigger() -> None:
    """Wrap ``pyutilz.system.gpu_dispatch.is_cuda_available`` so the GPU runtime is configured on first probe.

    Every mlframe GPU dispatcher routes its CUDA-availability check through this single pyutilz function, so
    wrapping it is the one deferral point that covers all GPU paths without importing cupy at ``import mlframe``
    time. The wrap is a pure reference swap -- it touches neither ``os.environ`` nor cupy until a GPU path
    actually calls it. If pyutilz is unavailable for any reason, GPU autoconfig simply stays deferred.
    """
    import logging
    try:
        from pyutilz.system import gpu_dispatch as _gd
    except Exception as e:
        logging.getLogger(__name__).debug("pyutilz.system.gpu_dispatch unavailable, GPU runtime autoconfig stays deferred: %s", e)
        return
    _orig = getattr(_gd, "is_cuda_available", None)
    if _orig is None or getattr(_orig, "_mlframe_gpu_runtime_wrapped", False):
        return

    import functools

    @functools.wraps(_orig)
    def _is_cuda_available(*args, **kwargs):
        """Configure the GPU runtime (once) then delegate to the real is_cuda_available()."""
        _ensure_gpu_runtime_configured()
        return _orig(*args, **kwargs)

    _is_cuda_available._mlframe_gpu_runtime_wrapped = True  # type: ignore[attr-defined]
    _gd.is_cuda_available = _is_cuda_available


_install_gpu_runtime_lazy_trigger()
del _install_gpu_runtime_lazy_trigger


from mlframe.version import __version__

from mlframe.config import (
    THOUSANDS_SEPARATOR,
    KERAS_MODEL_TYPES,
    LGBM_MODEL_TYPES,
    NGBOOST_MODEL_TYPES,
    XGBOOST_MODEL_TYPES,
    CATBOOST_MODEL_TYPES,
    HGBOOST_MODEL_TYPES,
    PYTORCH_MODEL_TYPES,
    TABNET_MODEL_TYPES,
    CategoricalsAssigning,
    CategoricalsHandling,
    MissingHandling,
    NumericsHandling,
    EarlyStopping,
    OutlierRemoval,
    FeatureSelection,
    HyperParameterTuning,
    SampleWeights,
    Resampling,
    TargetTransformer,
    ClassWeights,
)

__all__ = [
    "__version__",
    "THOUSANDS_SEPARATOR",
    "KERAS_MODEL_TYPES",
    "LGBM_MODEL_TYPES",
    "NGBOOST_MODEL_TYPES",
    "XGBOOST_MODEL_TYPES",
    "CATBOOST_MODEL_TYPES",
    "HGBOOST_MODEL_TYPES",
    "PYTORCH_MODEL_TYPES",
    "TABNET_MODEL_TYPES",
    "CategoricalsAssigning",
    "CategoricalsHandling",
    "MissingHandling",
    "NumericsHandling",
    "EarlyStopping",
    "OutlierRemoval",
    "FeatureSelection",
    "HyperParameterTuning",
    "SampleWeights",
    "Resampling",
    "TargetTransformer",
    "ClassWeights",
    # Lazily resolved headline training symbols (see ``__getattr__`` below).
    "train_mlframe_models_suite",
    "SimpleFeaturesAndTargetsExtractor",
    "SuiteResult",
    "FeaturesAndTargetsExtractor",
    "TargetTypes",
]


# Headline training symbols resolved lazily so ``import mlframe;
# mlframe.train_mlframe_models_suite`` works WITHOUT eagerly importing the heavy
# training stack at package load. Each entry points at the deepest module that
# defines the symbol so resolution pulls in the minimum surface. Mirrors the lazy
# ``__getattr__`` pattern in ``mlframe.training.__init__``.
_LAZY_TOPLEVEL = {
    "train_mlframe_models_suite": ("mlframe.training.core._main_train_suite", "train_mlframe_models_suite"),
    "SimpleFeaturesAndTargetsExtractor": ("mlframe.training.extractors._extractors_simple", "SimpleFeaturesAndTargetsExtractor"),
    "FeaturesAndTargetsExtractor": ("mlframe.training.extractors", "FeaturesAndTargetsExtractor"),
    "SuiteResult": ("mlframe.training.core._main_train_suite_encoding", "SuiteResult"),
    "TargetTypes": ("mlframe.training.configs", "TargetTypes"),
}

_lazy_toplevel_cache: dict = {}


def __getattr__(name: str):
    """Lazily resolve headline training symbols on first access."""
    if name in _LAZY_TOPLEVEL:
        if name not in _lazy_toplevel_cache:
            import importlib

            module_name, attr_name = _LAZY_TOPLEVEL[name]
            module = importlib.import_module(module_name)
            _lazy_toplevel_cache[name] = getattr(module, attr_name)
        return _lazy_toplevel_cache[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
