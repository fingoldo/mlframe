"""Measurement-backed backend selection for ``odds_ratio_combine`` via ``kernel_tuning_cache``.

Per the project's "integrate with kernel_tuning_cache, do NOT hardcode" rule: the choice between
njit-single-thread / njit-parallel / cupy-GPU for ``odds_ratio_combine`` genuinely depends on both input
shape AND host hardware (measured on this dev box, 2026-07-03):

    n=1,000    k=10: njit_single wins  (0.02ms  vs njit_par 0.09ms vs gpu_resident 0.39ms)
    n=100,000  k=20: gpu wins          (5.23ms  vs njit_par 8.38ms vs njit_single 48.3ms)
    n=1,000,000 k=5: njit_par wins     (25.9ms  vs gpu 28.2ms      vs njit_single ~166ms)

None of the three backends dominates uniformly, and the crossovers depend on the host's CPU core count / GPU
model — exactly the scenario ``pyutilz.performance.kernel_tuning.cache.KernelTuningCache`` exists for. This
module routes the choice through that cache, with a real (not placeholder) tuner that measures all
available backends at the ACTUAL call shape on first use. The sweep runs via ``async_sweep=True`` so it
NEVER blocks the caller's hot path: the first call at a new shape gets the size-threshold fallback
immediately while the sweep measures in a background thread and persists the winner for subsequent calls
(this process and every future process on this host).

Env override: ``MLFRAME_ODDS_COMBINE_BACKEND=njit_single|njit_parallel|cupy``.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)

_ENV_KEY = "MLFRAME_ODDS_COMBINE_BACKEND"
_VALID_BACKENDS = ("njit_single", "njit_parallel", "cupy")
_KERNEL_NAME = "calibration_odds_ratio_combine"


def _forced_backend() -> str | None:
    """Return the backend forced via the ``MLFRAME_ODDS_COMBINE_BACKEND`` env var, or ``None`` if unset/invalid."""
    val = os.environ.get(_ENV_KEY, "").strip().lower()
    return val if val in _VALID_BACKENDS else None


def _get_cache() -> Any:
    """Return the shared ``KernelTuningCache`` singleton, or ``None`` if pyutilz/FS is unavailable."""
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache
    except Exception:  # pyutilz / FS package unavailable -> hardcoded fallback.
        return None
    try:
        return get_kernel_tuning_cache()
    except Exception:  # pragma: no cover - defensive; singleton already guards.
        return None


def _measure_backend(fn: Callable[[], object], n_iters: int = 3) -> float:
    """Min-of-N wall time in ms, after one warm-up call."""
    fn()
    best = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _make_tuner(n: int, k: int) -> Callable[[], list[dict]]:
    """Build a zero-arg tuner that measures every available backend at shape ``(n, k)`` and returns the
    region dict for the winner. Closes over ``n, k`` so ``get_or_tune`` can call it with no arguments."""

    def _tuner() -> list[dict]:
        """Measure every available backend at shape ``(n, k)`` and return the region dict for the fastest one."""
        from mlframe.calibration.ensembling import _NUMBA_AVAILABLE, _odds_combine_njit, _odds_combine_njit_parallel

        rng = np.random.default_rng(0)
        p = rng.uniform(0.01, 0.99, size=(n, k))
        timings: dict[str, float] = {}
        if _NUMBA_AVAILABLE:
            timings["njit_single"] = _measure_backend(lambda: _odds_combine_njit(p, 1e-7))
            timings["njit_parallel"] = _measure_backend(lambda: _odds_combine_njit_parallel(p, 1e-7))
        try:
            import cupy as cp

            p_gpu = cp.asarray(p)

            def _gpu_call() -> object:
                """Run one GPU-resident odds-ratio-combine pass and synchronize before returning."""
                p_c = cp.clip(p_gpu, 1e-7, 1.0 - 1e-7)
                logits = cp.log(p_c / (1.0 - p_c))
                combined_logit = logits.sum(axis=1)
                r = 1.0 / (1.0 + cp.exp(-combined_logit))
                cp.cuda.Stream.null.synchronize()
                return r

            timings["cupy"] = _measure_backend(_gpu_call)
        except Exception as exc:
            logger.debug("odds_ratio_combine tuner: cupy unavailable/failed (%s)", exc)

        if not timings:
            return [{"backend_choice": "njit_single", "n_samples_max": n, "n_members_max": k}]
        winner = min(timings, key=lambda k_: timings[k_])
        # Region bounded at exactly the measured (n, k): only queries at this-or-smaller shape reuse this
        # measurement; a genuinely different (larger) shape misses and triggers its own tune, building up a
        # per-shape-bucket table over time. Coarser than a full multi-point grid sweep (see
        # ``joint_hist_batched``), but a real per-call measurement beats a hardcoded guess.
        return [{"backend_choice": winner, "n_samples_max": n, "n_members_max": k, **{f"wall_ms_{k_}": v for k_, v in timings.items()}}]

    return _tuner


def choose_odds_combine_backend(n: int, k: int, *, fallback: str) -> str:
    """Pick ``"njit_single"`` / ``"njit_parallel"`` / ``"cupy"`` for ``odds_ratio_combine`` at shape ``(n, k)``.

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
            dims={"n_samples": n, "n_members": k},
            tuner=_make_tuner(n, k),
            axes=["n_samples", "n_members"],
            fallback={"backend_choice": fallback},
            async_sweep=True,
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        if bc in _VALID_BACKENDS:
            return bc
    except Exception as exc:  # any cache hiccup -> hardcoded fallback, never raise.
        logger.debug("odds_ratio_combine KTC lookup failed: %s", exc)
    return fallback


__all__ = ["choose_odds_combine_backend"]
