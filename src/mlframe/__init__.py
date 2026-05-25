"""mlframe -- production-grade tabular ML toolkit (classification, regression, ranking, quantile).

Public API convention: deep-import from subpackages, not from this top-level module.
The top-level ``mlframe`` namespace intentionally exports ONLY ``__version__`` and the
``mlframe.config`` enums; everything else is reached via its subpackage path:

    from mlframe.training import train_mlframe_models_suite
    from mlframe.training import CompositeTargetEstimator   # via training/__init__.py
    from mlframe.feature_selection import MRMR, RFECV
    from mlframe.metrics.core import expected_calibration_error
    from mlframe.calibration.quality import pick_best_calibrator
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
``mlframe.models.ensembling``, and ``mlframe.calibration.quality`` directly.
"""

from __future__ import annotations


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


_disable_broken_cupy()
del _disable_broken_cupy


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
]
