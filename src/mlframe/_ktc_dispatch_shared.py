"""Shared helpers for the per-domain ``_ktc_dispatch.py`` modules (calibration, inference,
votenrank, training/composite/discovery): small utilities independently duplicated across those
modules, consolidated here so a fix can't silently drift out of sync across copies. Each module
keeps its own ``_ENV_KEY``/``_VALID_BACKENDS``/``_KERNEL_NAME`` domain config and dispatch table.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)


def forced_backend(env_key: str, valid_backends: Sequence[str]) -> Optional[str]:
    """Return the backend forced via the ``env_key`` env var, or ``None`` if unset/invalid."""
    val = os.environ.get(env_key, "").strip().lower()
    return val if val in valid_backends else None


def get_ktc_cache() -> Any:
    """Return the shared ``KernelTuningCache`` singleton, or ``None`` if pyutilz/FS is unavailable."""
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache
    except Exception as exc:  # pyutilz / FS package unavailable -> hardcoded fallback.
        logger.debug("get_ktc_cache: import failed, tuning cache unavailable: %s", exc)
        return None
    try:
        return get_kernel_tuning_cache()
    except Exception as exc:  # pragma: no cover - defensive; singleton already guards.
        logger.debug("get_ktc_cache: get_kernel_tuning_cache() failed: %s", exc)
        return None


def measure_backend(fn: Callable[[], object], n_iters: int = 3, setup: Optional[Callable[[], None]] = None) -> float:
    """Warm ``fn`` once then return its best wall-clock time in milliseconds over ``n_iters`` runs.

    ``setup`` runs before every call and is NOT timed. It exists for kernels that mutate their input buffer:
    re-running such a kernel on its own output measures an already-converged no-op, while giving each
    iteration a fresh copy inside the timed region charges the backend for an allocation its production call
    site never makes. Restoring the buffer in ``setup`` is the only form that measures the production shape.
    """
    if setup is not None:
        setup()
    fn()
    best = float("inf")
    for _ in range(n_iters):
        if setup is not None:
            setup()
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best
