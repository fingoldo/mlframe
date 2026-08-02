"""Measurement-backed backend selection for ``confidence_gated_blend`` via ``kernel_tuning_cache``.

Measured on this dev box, 2026-07-10 (n=1,000,000 elements, host-array input):

    numpy         20.7 ms  (np.where + two elementwise multiplies -- 3 intermediate array allocations)
    njit           4.4 ms  (single fused pass, no intermediate arrays)
    njit_parallel  3.6 ms  (prange over the fused pass)
    cupy resident  0.8 ms  (data already on GPU)
    cupy e2e       8.5 ms  (host input: H2D transfer dominates -- slower than njit_parallel)

No backend dominates uniformly: cupy wins only when the caller already has GPU-resident arrays (skips
transfer), otherwise njit_parallel wins for host input at large n while plain njit/numpy are competitive at
small n once dispatch overhead is accounted for. Routed through
``pyutilz.performance.kernel_tuning.cache.KernelTuningCache`` per the project's "measure, don't hardcode a
threshold" rule, with a real (not placeholder) tuner that measures all available backends at the ACTUAL call
shape on first use, non-blocking via ``async_sweep=True``.

Env override: ``MLFRAME_CONFIDENCE_BLEND_BACKEND=numpy|njit|njit_parallel|cupy``.
"""
from __future__ import annotations

from mlframe._ktc_dispatch_shared import forced_backend, measure_backend, get_ktc_cache

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

_ENV_KEY = "MLFRAME_CONFIDENCE_BLEND_BACKEND"
_VALID_BACKENDS = ("numpy", "njit", "njit_parallel", "cupy")
_KERNEL_NAME = "votenrank_confidence_gated_blend"


def _make_tuner(n: int) -> Callable[[], list]:
    """Zero-arg tuner measuring every available backend at size ``n`` for HOST-array input (the common
    caller shape); GPU-resident-input callers should force ``cupy`` directly since the measured e2e cost here
    includes a transfer that a resident caller wouldn't pay."""

    def _tuner() -> list:
        """Measure every available backend at size n and return the region dict for the fastest one."""
        from mlframe.votenrank.confidence_gated_blend import _blend_njit, _blend_njit_parallel, _blend_numpy

        rng = np.random.default_rng(0)
        ensemble = rng.uniform(0, 1, n)
        aux = rng.uniform(0, 1, n)
        conf = rng.uniform(0, 1, n)
        threshold, gated_w, default_w = 0.6, 0.5, 0.0

        timings: dict = {}
        timings["numpy"] = measure_backend(lambda: _blend_numpy(ensemble, aux, conf, threshold, gated_w, default_w))
        try:
            _blend_njit(ensemble, aux, conf, threshold, gated_w, default_w)  # compile before timing
            timings["njit"] = measure_backend(lambda: _blend_njit(ensemble, aux, conf, threshold, gated_w, default_w))
            _blend_njit_parallel(ensemble, aux, conf, threshold, gated_w, default_w)
            timings["njit_parallel"] = measure_backend(lambda: _blend_njit_parallel(ensemble, aux, conf, threshold, gated_w, default_w))
        except Exception as exc:
            logger.debug("confidence_gated_blend tuner: numba unavailable/failed (%s)", exc)
        try:
            import cupy as cp

            def _gpu_call() -> object:
                """Run one GPU-resident confidence-gated blend pass and return the host result."""
                e = cp.asarray(ensemble)
                a = cp.asarray(aux)
                c = cp.asarray(conf)
                weight = cp.where(c >= threshold, gated_w, default_w)
                out = (1.0 - weight) * e + weight * a
                result = cp.asnumpy(out)
                return result

            timings["cupy"] = measure_backend(_gpu_call)
        except Exception as exc:
            logger.debug("confidence_gated_blend tuner: cupy unavailable/failed (%s)", exc)

        if not timings:
            return [{"backend_choice": "numpy", "n_samples_max": n}]
        winner = min(timings, key=lambda k_: timings[k_])
        return [{"backend_choice": winner, "n_samples_max": n, **{f"wall_ms_{k_}": v for k_, v in timings.items()}}]

    return _tuner


def choose_confidence_blend_backend(n: int, *, fallback: str) -> str:
    """Pick ``"numpy"`` / ``"njit"`` / ``"njit_parallel"`` / ``"cupy"`` for ``confidence_gated_blend`` at size ``n``.

    Order: env-var force-override -> KTC measured region -> caller-supplied size-threshold fallback (used
    immediately on cache miss; the async sweep measures in the background and persists for next time).
    """
    forced = forced_backend(_ENV_KEY, _VALID_BACKENDS)
    if forced is not None:
        return forced

    cache = get_ktc_cache()
    if cache is None or cache is False:
        return fallback
    try:
        result = cache.get_or_tune(
            _KERNEL_NAME,
            dims={"n_samples": n},
            tuner=_make_tuner(n),
            axes=["n_samples"],
            fallback={"backend_choice": fallback},
            async_sweep=True,
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        if bc in _VALID_BACKENDS:
            return bc
    except Exception as exc:  # any cache hiccup -> hardcoded fallback, never raise.
        logger.debug("confidence_gated_blend KTC lookup failed: %s", exc)
    return fallback


__all__ = ["choose_confidence_blend_backend"]
