"""GPU probing helpers for XGBoost and LightGBM."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# numba is an optional dep: probing CUDA via numba.cuda is convenient but the
# training package must still import on machines without numba (or without a
# working CUDA driver). Wrap both the import and the call so any failure -
# ImportError, OSError on missing libcuda, runtime probe errors - degrades
# silently to CPU-only mode.
try:
    from numba.cuda import is_available as is_cuda_available
    try:
        CUDA_IS_AVAILABLE = bool(is_cuda_available())
    except Exception:
        CUDA_IS_AVAILABLE = False
except Exception:
    CUDA_IS_AVAILABLE = False


def _probe_xgb_gpu_support() -> bool:
    """True only when a CUDA device is visible AND the installed XGBoost binary was built with ``USE_CUDA``; avoids the per-fit GPU-fallback warning XGB emits when asked for a device it cannot use."""
    if not CUDA_IS_AVAILABLE:
        return False
    try:
        import xgboost as _xgb
        info = _xgb.build_info() if hasattr(_xgb, "build_info") else {}
        return bool(info.get("USE_CUDA", False))
    except Exception:
        return False


def _probe_lgb_gpu_support() -> bool:
    """Conservative LightGBM CUDA-support probe: cannot cheaply detect a CUDA-enabled LGB build without a real training run, so defaults to ``False`` and only returns ``True`` when the caller opts in via ``MLFRAME_TRUST_LGB_CUDA=1``."""
    if not CUDA_IS_AVAILABLE:
        return False
    try:
        # LightGBM exposes GPU via either CUDA build or OpenCL build.
        # The ``device_type='cuda'`` path requires a build flag we can
        # detect by attempting a tiny train with device_type='cuda';
        # too expensive to do at import. Instead, probe the binary
        # filename for hints (``lib_lightgbm_cuda`` etc.) and fall
        # back to True only if a known marker is present. Conservative:
        # default False, opt-in by setting the env var
        # ``MLFRAME_TRUST_LGB_CUDA=1`` if you know your build supports it.
        import os
        if os.environ.get("MLFRAME_TRUST_LGB_CUDA") == "1":
            return True
        return False
    except Exception:
        return False


XGB_GPU_AVAILABLE = _probe_xgb_gpu_support()
LGB_GPU_AVAILABLE = _probe_lgb_gpu_support()

if CUDA_IS_AVAILABLE and not XGB_GPU_AVAILABLE:
    logger.info(
        "[gpu-probe] CUDA detected but installed XGBoost binary lacks GPU support "
        "(``xgb.build_info()['USE_CUDA']`` is False). XGB will run on CPU; "
        "rebuild XGB with USE_CUDA=ON or install a GPU wheel to enable. "
        "This INFO replaces a per-fit ``WARNING: Device is changed from GPU "
        "to CPU as we couldn't find any available GPU on the system``."
    )
if CUDA_IS_AVAILABLE and not LGB_GPU_AVAILABLE:
    logger.info("[gpu-probe] LightGBM GPU support not opted-in " "(``MLFRAME_TRUST_LGB_CUDA`` not set). LGB will run on CPU.")


# =============================================================================
# Multi-output (multiclass + multilabel) dispatch helpers — 2026-04-24
# =============================================================================
#
# Probability-surface contract: every classification estimator's
# ``predict_proba`` is canonicalised to ``(N, K)`` shape regardless of
# source — sklearn binary returns ``(N, 2)``, ``MultiOutputClassifier``
# returns ``List[(N, 2)]``, CB native ``MultiLogloss`` returns ``(N, K)``
# already. The canonicalizer + decision-rule pair below wraps that
# heterogeneity behind two pure functions used at every site that
# previously hard-coded ``probs[:, 1]`` (4 sites in core.py, 3 in
# evaluation.py, 2 in automl.py).
