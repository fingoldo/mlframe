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
        d = pathlib.Path(p)
        return bool(list(d.glob("bin/cudart64_*.dll")) or list(d.glob("lib64/libcudart*")) or list(d.glob("lib/libcudart*")))

    def _dir_has_nvvm(p: str) -> bool:
        d = pathlib.Path(p)
        return bool(list(d.glob("nvvm/bin/nvvm*.dll")) or list(d.glob("nvvm/lib64/libnvvm*")))

    def _find_complete_toolkit() -> "str | None":
        # A toolkit that can actually COMPILE+RUN numba kernels needs BOTH nvvm (for codegen) and cudart
        # (numba's get_supported_ccs queries the runtime). The versioned CUDA_PATH_V* vars the NVIDIA
        # system installer sets point at full toolkits; pick the highest that has both libraries.
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
    _cur = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
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
            has_dll = bool(list(nvvm.glob("bin/nvvm*.dll")) or list(nvvm.glob("lib64/libnvvm*")))
            has_libdevice = bool(list(nvvm.glob("libdevice/libdevice*.bc")))
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
    except Exception:
        pass


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


_gpu_runtime_configured = False


def _ensure_gpu_runtime_configured() -> None:
    """Run the CUDA env autoconfig + broken-cupy guard once, on first GPU-path use.

    These two routines mutate ``os.environ`` (CUDA_HOME/CUDA_PATH) and import cupy + run a CUDA reduction
    kernel. Doing that at ``import mlframe`` time is hostile to the majority of users who never touch a GPU
    path: it probes the GPU and rewrites their environment on every import. Both are therefore deferred to the
    first time a GPU dispatcher asks ``is_cuda_available()`` (see the wrapper installed below), so a plain
    ``import mlframe`` neither mutates the environment nor imports cupy. The opt-out env vars
    (MLFRAME_NO_CUDA_AUTOCONFIG / MLFRAME_KEEP_BROKEN_CUPY) are still honoured inside each routine.
    """
    global _gpu_runtime_configured
    if _gpu_runtime_configured:
        return
    _gpu_runtime_configured = True
    _autoconfigure_cuda_home()
    _disable_broken_cupy()


def _install_gpu_runtime_lazy_trigger() -> None:
    """Wrap ``pyutilz.system.gpu_dispatch.is_cuda_available`` so the GPU runtime is configured on first probe.

    Every mlframe GPU dispatcher routes its CUDA-availability check through this single pyutilz function, so
    wrapping it is the one deferral point that covers all GPU paths without importing cupy at ``import mlframe``
    time. The wrap is a pure reference swap -- it touches neither ``os.environ`` nor cupy until a GPU path
    actually calls it. If pyutilz is unavailable for any reason, GPU autoconfig simply stays deferred.
    """
    try:
        from pyutilz.system import gpu_dispatch as _gd
    except Exception:
        return
    _orig = getattr(_gd, "is_cuda_available", None)
    if _orig is None or getattr(_orig, "_mlframe_gpu_runtime_wrapped", False):
        return

    import functools

    @functools.wraps(_orig)
    def _is_cuda_available(*args, **kwargs):
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
