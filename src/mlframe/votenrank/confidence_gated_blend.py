"""Confidence-gated override blend: fold a high-precision auxiliary model into an ML ensemble only where it's confident.

A small hand-built or rule-based auxiliary model is often low-recall but very precise in the narrow region
where it fires. Blending it at a fixed weight everywhere dilutes the main ensemble's predictions across the
whole input space for a signal that's only trustworthy in a slice of it. This "confidence-gated override
blend" (used by an 8th-place Ubiquant team, ``0.92 * ensemble + 0.08 * handbuilt_model`` only once the
handbuilt model's own confidence cleared a threshold) instead applies the auxiliary model's blend weight ONLY
where its own confidence signal exceeds a threshold, falling back to a low/zero weight elsewhere.

Backend dispatch (numpy / njit / njit_parallel / cupy) is measured and routed through
``pyutilz.performance.kernel_tuning.cache.KernelTuningCache`` -- see ``_confidence_gated_blend_ktc_dispatch.py``
for the measured numbers and rationale (no backend dominates uniformly; cupy only wins GPU-resident).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    from numba import njit, prange

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a hard dependency in practice, guarded defensively.
    _NUMBA_AVAILABLE = False

# Below this size, dispatch/compile-check overhead dominates any backend difference; skip the ladder per the
# project's "skip the whole ladder for kernels called <100x/fit or already <1% of wall" rule for tiny inputs.
_DISPATCH_MIN_N = 2_000


def _blend_numpy(ensemble: np.ndarray, aux: np.ndarray, conf: np.ndarray, threshold: float, gated_w: float, default_w: float) -> np.ndarray:
    weight = np.where(conf >= threshold, gated_w, default_w)
    return np.asarray((1.0 - weight) * ensemble + weight * aux, dtype=np.float64)


if _NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True)
    def _blend_njit(ensemble: np.ndarray, aux: np.ndarray, conf: np.ndarray, threshold: float, gated_w: float, default_w: float) -> np.ndarray:
        n = ensemble.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            w = gated_w if conf[i] >= threshold else default_w
            out[i] = (1.0 - w) * ensemble[i] + w * aux[i]
        return out

    @njit(cache=True, fastmath=True, parallel=True)
    def _blend_njit_parallel(ensemble: np.ndarray, aux: np.ndarray, conf: np.ndarray, threshold: float, gated_w: float, default_w: float) -> np.ndarray:
        n = ensemble.shape[0]
        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            w = gated_w if conf[i] >= threshold else default_w
            out[i] = (1.0 - w) * ensemble[i] + w * aux[i]
        return out

else:  # pragma: no cover - defensive fallback when numba is unavailable.
    _blend_njit = _blend_numpy
    _blend_njit_parallel = _blend_numpy


def _blend_cupy(ensemble: np.ndarray, aux: np.ndarray, conf: np.ndarray, threshold: float, gated_w: float, default_w: float) -> np.ndarray:
    import cupy as cp

    e = cp.asarray(ensemble)
    a = cp.asarray(aux)
    c = cp.asarray(conf)
    weight = cp.where(c >= threshold, gated_w, default_w)
    out = (1.0 - weight) * e + weight * a
    return np.asarray(cp.asnumpy(out), dtype=np.float64)


def confidence_gated_blend(
    ensemble_pred: np.ndarray,
    auxiliary_pred: np.ndarray,
    auxiliary_confidence: np.ndarray,
    confidence_threshold: float,
    gated_weight: float,
    default_weight: float = 0.0,
    force_backend: Optional[str] = None,
) -> np.ndarray:
    """Blend ``auxiliary_pred`` into ``ensemble_pred`` at ``gated_weight`` only where confident, else ``default_weight``.

    Parameters
    ----------
    ensemble_pred, auxiliary_pred
        Same-length prediction arrays (probabilities or scores) from the main ensemble and the auxiliary
        (typically rule-based/handbuilt) model.
    auxiliary_confidence
        Per-row confidence signal for the auxiliary model (e.g. its own predicted probability, or a distance-
        from-decision-boundary score) -- rows with ``auxiliary_confidence >= confidence_threshold`` get the
        auxiliary model blended in at ``gated_weight``; other rows get ``default_weight`` (typically ``0.0``,
        i.e. pure ensemble).
    confidence_threshold
        The gating cutoff on ``auxiliary_confidence``.
    gated_weight
        Auxiliary model's blend weight where confident (``0 <= gated_weight <= 1``).
    default_weight
        Auxiliary model's blend weight elsewhere (``0 <= default_weight <= 1``); the ensemble always gets the
        complementary weight ``1 - w`` at each row.
    force_backend
        Override the measured/dispatched backend (``"numpy"``/``"njit"``/``"njit_parallel"``/``"cupy"``);
        also settable via the ``MLFRAME_CONFIDENCE_BLEND_BACKEND`` env var (checked first).

    Returns
    -------
    np.ndarray
        The blended prediction, same shape as ``ensemble_pred``.
    """
    if not (0.0 <= gated_weight <= 1.0) or not (0.0 <= default_weight <= 1.0):
        raise ValueError("confidence_gated_blend: gated_weight and default_weight must be in [0, 1]")
    ensemble_pred = np.asarray(ensemble_pred, dtype=np.float64)
    auxiliary_pred = np.asarray(auxiliary_pred, dtype=np.float64)
    auxiliary_confidence = np.asarray(auxiliary_confidence, dtype=np.float64)
    if not (ensemble_pred.shape == auxiliary_pred.shape == auxiliary_confidence.shape):
        raise ValueError("confidence_gated_blend: ensemble_pred, auxiliary_pred, auxiliary_confidence must share the same shape")

    n = ensemble_pred.shape[0]
    if n < _DISPATCH_MIN_N:
        return _blend_numpy(ensemble_pred, auxiliary_pred, auxiliary_confidence, confidence_threshold, gated_weight, default_weight)

    env_backend = os.environ.get("MLFRAME_CONFIDENCE_BLEND_BACKEND", "").strip().lower()
    backend = force_backend or (env_backend if env_backend in ("numpy", "njit", "njit_parallel", "cupy") else None)
    if backend is None:
        from mlframe.votenrank._confidence_gated_blend_ktc_dispatch import choose_confidence_blend_backend

        fallback = "njit_parallel" if _NUMBA_AVAILABLE else "numpy"
        backend = choose_confidence_blend_backend(n, fallback=fallback)

    if backend == "njit" and _NUMBA_AVAILABLE:
        return np.asarray(_blend_njit(ensemble_pred, auxiliary_pred, auxiliary_confidence, confidence_threshold, gated_weight, default_weight))
    if backend == "njit_parallel" and _NUMBA_AVAILABLE:
        return np.asarray(_blend_njit_parallel(ensemble_pred, auxiliary_pred, auxiliary_confidence, confidence_threshold, gated_weight, default_weight))
    if backend == "cupy":
        try:
            return _blend_cupy(ensemble_pred, auxiliary_pred, auxiliary_confidence, confidence_threshold, gated_weight, default_weight)
        except Exception:
            pass  # GPU unavailable/failed -> fall through to numpy.
    return _blend_numpy(ensemble_pred, auxiliary_pred, auxiliary_confidence, confidence_threshold, gated_weight, default_weight)


__all__ = ["confidence_gated_blend"]
