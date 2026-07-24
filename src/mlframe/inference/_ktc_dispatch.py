"""Measurement-backed backend selection for ``apply_logical_constraints`` via ``kernel_tuning_cache``.

Per the project's "integrate with kernel_tuning_cache, do NOT hardcode" rule: the choice between
njit-single-thread / njit-parallel / cupy-GPU genuinely depends on shape AND hardware (measured on this dev
box, 2026-07-03, n_labels=10-20, 3-5 rules):

    n=1,000:     njit_single wins  (0.02ms  vs njit_par 0.09ms vs gpu 5.19ms)
    n=100,000:   njit_parallel wins (8.38ms vs gpu 9.73ms      vs njit_single 47.6ms)
    n=1,000,000: cupy wins          (12.29ms vs njit_par 45.2ms vs njit_single ~290ms)

GPU launch overhead dominates at small n (many tiny per-rule kernel launches), but wins decisively at large n
once the row count amortises it. No backend dominates uniformly, so the choice routes through the tuning
cache exactly as ``calibration.ensembling`` does.

Env override: ``MLFRAME_LOGICAL_CONSTRAINTS_BACKEND=njit_single|njit_parallel|cupy``.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_ENV_KEY = "MLFRAME_LOGICAL_CONSTRAINTS_BACKEND"
_VALID_BACKENDS = ("njit_single", "njit_parallel", "cupy")
_KERNEL_NAME = "inference_apply_logical_constraints"


def _forced_backend() -> str | None:
    """Return the backend name forced via the env-var override, or ``None`` if unset/invalid."""
    val = os.environ.get(_ENV_KEY, "").strip().lower()
    return val if val in _VALID_BACKENDS else None


def _get_cache() -> Any:
    """Return the shared kernel-tuning-cache singleton, or ``None`` if unavailable."""
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache
    except Exception as exc:
        logger.debug("_get_cache: import failed, tuning cache unavailable: %s", exc)
        return None
    try:
        return get_kernel_tuning_cache()
    except Exception as exc:  # pragma: no cover - defensive; singleton already guards.
        logger.debug("_get_cache: singleton construction failed: %s", exc)
        return None


def _measure_backend(fn: Callable[[], object], n_iters: int = 3) -> float:
    """Warm ``fn`` once then return its best wall-clock time in milliseconds over ``n_iters`` runs."""
    fn()
    best = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _make_tuner(n: int, n_labels: int, n_rules: int) -> Callable[[], list[dict]]:
    """Zero-arg tuner measuring every available backend at shape ``(n, n_labels, n_rules)``."""

    def _tuner() -> list[dict]:
        """Measure njit-single, njit-parallel, and cupy backends at this shape and return the winning tuning rule."""
        from mlframe.inference.logical_constraints import _NUMBA_AVAILABLE, _apply_njit, _apply_njit_parallel

        rng = np.random.default_rng(0)
        preds = rng.uniform(0, 1, size=(n, n_labels))
        rules = [(i, i + 1) for i in range(0, min(n_rules * 2, n_labels - 1), 2)] or [(0, min(1, n_labels - 1))]
        rules_arr = np.asarray(rules, dtype=np.int64)

        timings: dict[str, float] = {}
        if _NUMBA_AVAILABLE:
            timings["njit_single"] = _measure_backend(lambda: _apply_njit(preds.copy(), rules_arr))
            timings["njit_parallel"] = _measure_backend(lambda: _apply_njit_parallel(preds.copy(), rules_arr))
        try:
            import cupy as cp

            preds_gpu = cp.asarray(preds)
            rules_gpu = cp.asarray(rules_arr)

            def _gpu_call() -> object:
                """Apply the rule set on the GPU-resident arrays and synchronize before returning the result."""
                out = preds_gpu.copy()
                for r in range(rules_gpu.shape[0]):
                    c, p = int(rules_arr[r, 0]), int(rules_arr[r, 1])
                    violates = out[:, c] > out[:, p]
                    tmp = out[violates, c].copy()
                    out[violates, c] = out[violates, p]
                    out[violates, p] = tmp
                cp.cuda.Stream.null.synchronize()
                return out

            timings["cupy"] = _measure_backend(_gpu_call)
        except Exception as exc:
            logger.debug("apply_logical_constraints tuner: cupy unavailable/failed (%s)", exc)

        if not timings:
            return [{"backend_choice": "njit_single", "n_samples_max": n, "n_labels_max": n_labels}]
        winner = min(timings, key=lambda k_: timings[k_])
        return [
            {
                "backend_choice": winner,
                "n_samples_max": n,
                "n_labels_max": n_labels,
                **{f"wall_ms_{k_}": v for k_, v in timings.items()},
            }
        ]

    return _tuner


def choose_logical_constraints_backend(n: int, n_labels: int, n_rules: int, *, fallback: str) -> str:
    """Pick ``"njit_single"`` / ``"njit_parallel"`` / ``"cupy"`` for ``apply_logical_constraints``.

    Order: env-var force-override -> KTC measured region -> caller-supplied size-threshold fallback (used
    immediately on cache miss; the async sweep measures in the background and persists for next time).
    """
    forced = _forced_backend()
    if forced is not None:
        return forced

    cache = _get_cache()
    if cache is None or cache is False:
        return fallback
    try:
        result = cache.get_or_tune(
            _KERNEL_NAME,
            dims={"n_samples": n, "n_labels": n_labels},
            tuner=_make_tuner(n, n_labels, n_rules),
            axes=["n_samples", "n_labels"],
            fallback={"backend_choice": fallback},
            async_sweep=True,
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        if bc in _VALID_BACKENDS:
            return bc
    except Exception as exc:
        logger.debug("apply_logical_constraints KTC lookup failed: %s", exc)
    return fallback


__all__ = ["choose_logical_constraints_backend"]
