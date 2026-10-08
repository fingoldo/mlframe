"""Leaf numeric helpers of the usability-aware selection: finite scrub, float64 view, safe absolute correlation and the GPU-path switch (carved from _usability_aware_selection; re-exported there)."""

from __future__ import annotations

import logging

from typing import Any
import numpy as np

logger = logging.getLogger(__name__)


def _scrub(v: np.ndarray, dtype: Any = np.float64) -> np.ndarray:
    """Cast ``v`` to ``dtype`` and replace every non-finite entry (NaN/+inf/-inf) with 0.0, returning a new array."""
    # ``np.where(isfinite, a, 0)`` is bit-identical to ``nan_to_num(nan=0, posinf=0, neginf=0)`` (isfinite
    # is False for exactly nan/+inf/-inf) but ~2.8x faster (no per-call isposinf/isneginf/_getmaxmin
    # machinery): 764us -> 269us on a 100k float32 column. _scrub is called ~17k+/retention fit on full-n
    # columns, so this is a direct cut to the pool-build cost. Verified bit-identical over float32/float64 +
    # nan/inf fuzz.
    # A cast to a narrower dtype (e.g. float64 -> float32) can overflow to +-inf on an extreme
    # input value -- numpy warns "overflow encountered in cast" even though that's exactly the
    # non-finite case this function's whole job is to zero out right below. Harmless, suppressed
    # locally rather than at every caller.
    with np.errstate(over="ignore"):
        a = np.asarray(v, dtype=dtype)
    return np.where(np.isfinite(a), a, 0)


def _f64(v: np.ndarray) -> np.ndarray:
    """Upcast a stored (possibly float32) candidate column to float64 for MI / correlation /
    recipe-edge computation where the heavy-tail precision matters (transient; not stored)."""
    return np.asarray(v, dtype=np.float64)


def _abscorr(u: np.ndarray, v: np.ndarray) -> float:
    """Absolute Pearson correlation ``|corr(u, v)|`` in float64, used as the diversity / near-duplicate gate. Returns
    0.0 if either input is empty or near-constant (std < 1e-12), or if the raw correlation is non-finite."""
    # GATED GPU PATH (MLFRAME_FE_GPU_USABILITY, default OFF). The cupy twin is float64 + the SAME
    # std<1e-12 guard, but a cupy reduction can reassociate the last bits vs numpy -> a |corr| drift
    # that, on the ULP-sensitive clean-form demotion, could flip a pin. So it is ENABLED only on a host
    # where the gate-on pytest verified the SAME selection; on ANY cupy/device error we fall through to
    # the exact numpy path (the fit is never broken by a GPU problem).
    if _GPU_USABILITY():
        try:
            from ._usability_gpu import gpu_abscorr
            return gpu_abscorr(u, v)
        except Exception as e:  # nosec B110 - optional/best-effort path, rationale documented
            logger.debug("_abscorr: GPU path failed, falling back to the exact CPU path: %s", e)
    u = _f64(u); v = _f64(v)  # precision for the heavy-tail correlation
    if u.size == 0 or float(np.std(u)) < 1e-12 or float(np.std(v)) < 1e-12:
        return 0.0
    r = np.corrcoef(u, v)[0, 1]
    return abs(float(r)) if np.isfinite(r) else 0.0


def _GPU_USABILITY() -> bool:
    """Whether the gated cupy usability-scoring path is active (``MLFRAME_FE_GPU_USABILITY`` + live
    cupy + global GPU not disabled). Default OFF; the CPU path is the proven, selection-exact default.
    Imported lazily so a no-cupy host never touches the GPU module."""
    try:
        from ._usability_gpu import fe_gpu_usability_enabled
        return fe_gpu_usability_enabled()
    except Exception as e:
        logger.debug("_GPU_USABILITY: fe_gpu_usability_enabled() check failed, staying on the CPU path: %s", e)
        return False
