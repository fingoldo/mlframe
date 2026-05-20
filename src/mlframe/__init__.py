"""mlframe -- production-grade tabular ML toolkit (classification, regression, ranking, quantile).

Public API surface is re-exported from the subpackages below. Import the
subpackage directly (``from mlframe.metrics import ECE``) rather than reaching
into private submodules.
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
    except BaseException as exc:
        import logging
        logging.getLogger(__name__).warning(
            "mlframe: cupy is installed but unusable (NVRTC probe failed: "
            "%s: %s). Disabling cupy for this process to avoid the softlink "
            "RecursionError cascade. Set MLFRAME_KEEP_BROKEN_CUPY=1 to skip.",
            type(exc).__name__, exc,
        )
        # Poison the import: subsequent `import cupy` raises ImportError,
        # which mlframe's GPU code paths already handle as a fallback signal.
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
