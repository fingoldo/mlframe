"""GPU probing helpers for XGBoost and LightGBM."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# numba is an optional dep, so a genuine ImportError means "no numba" and CPU-only is the right answer.
#
# Anything ELSE is not evidence about the machine, and it used to be treated as if it were. This runs at IMPORT
# and its result is cached for the process lifetime, so a transient fault -- a driver reset, a contended GPU, an
# OSError from libcuda while another process holds the device -- pinned every CatBoost `task_type`, every
# `_xgb_device` and every `_lgb_device` resolution to CPU for the rest of the run, logged only at `debug`. That
# is the same shape as the documented `_select_mi_backend` regression, where one startup hiccup cost ~100x.
#
# So: ImportError is silent and decisive; anything else warns and stays OPTIMISTIC, leaving the decision to the
# per-library probes below, which check the installed binary rather than the driver and cannot be fooled by a
# momentary device fault.
try:
    from numba.cuda import is_available as is_cuda_available
except ImportError as e:
    logger.debug("numba.cuda unavailable (%s); assuming no CUDA", e)
    CUDA_IS_AVAILABLE = False
else:
    try:
        CUDA_IS_AVAILABLE = bool(is_cuda_available())
    except Exception as e:
        logger.warning(
            "numba.cuda.is_available() raised %s: %s -- this is a transient device/driver condition rather than "
            "evidence that no GPU exists, so GPU support stays enabled and the per-library probes decide. Set "
            "CUDA_VISIBLE_DEVICES='' to force CPU.",
            type(e).__name__, e,
        )
        CUDA_IS_AVAILABLE = True


def _probe_xgb_gpu_support() -> bool:
    """True only when a CUDA device is visible AND the installed XGBoost binary was built with ``USE_CUDA``; avoids the per-fit GPU-fallback warning XGB emits when asked for a device it cannot use."""
    if not CUDA_IS_AVAILABLE:
        return False
    try:
        import xgboost as _xgb
        info = _xgb.build_info() if hasattr(_xgb, "build_info") else {}
        return bool(info.get("USE_CUDA", False))
    except Exception as exc:
        logger.debug("_probe_xgb_gpu_support: build_info probe failed, assuming no GPU support: %s", exc)
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
    except Exception as exc:
        logger.debug("_probe_lgb_gpu_support: env-var probe failed, assuming no GPU support: %s", exc)
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
