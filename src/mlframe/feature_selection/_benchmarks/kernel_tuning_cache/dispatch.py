"""Runtime dispatcher for the mlframe joint-hist kernels.

Thin wrapper that delegates storage + lookup to the generic
``pyutilz.system.kernel_tuning_cache.KernelTuningCache``. This module
keeps the mlframe-specific entry point ``lookup_joint_hist`` so
``filters/gpu.py:mi_direct_gpu_batched`` doesn't need to know about the
generic backing storage; it also owns the hand-tuned fallbacks used
when the cache is missing AND auto-tune hasn't been triggered yet.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


# Hand-tuned fallbacks used before / instead of the sweep cache. These
# match the constants in ``filters/gpu.py:mi_direct_gpu_batched`` so the
# dispatcher's "first-process / no cache yet" decision is identical to
# the hand-coded one.
_FALLBACK_JOINT_HIST = {"kernel_variant": "shared", "block_size": 512}
_FALLBACK_JOINT_HIST_GLOBAL = {"kernel_variant": "global", "block_size": 1024}
_SHARED_HIST_MAX_JOINT_FALLBACK = 4096


_CACHE_SINGLETON: Optional[object] = None  # KernelTuningCache instance
_LOAD_LOCK = threading.Lock()


def _get_cache():
    """Lazy import + singleton init of the pyutilz cache."""
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is not None:
        return _CACHE_SINGLETON
    with _LOAD_LOCK:
        if _CACHE_SINGLETON is None:
            try:
                from pyutilz.system.kernel_tuning_cache import KernelTuningCache
                _CACHE_SINGLETON = KernelTuningCache()
            except ImportError:
                logger.debug(
                    "pyutilz.system.kernel_tuning_cache unavailable; "
                    "using hand-tuned fallbacks"
                )
                _CACHE_SINGLETON = False  # sentinel: cache absent
    return _CACHE_SINGLETON


def lookup_joint_hist(n_samples: int, joint_size: int,
                       *, run_auto_tune: bool = False) -> dict:
    """Return ``{"kernel_variant", "block_size"}`` for the given size pair.

    Hits the pyutilz ``KernelTuningCache`` for ``joint_hist_batched``.
    On cache miss + ``run_auto_tune=True`` triggers a one-time sweep
    (~30s) via :mod:`auto_tune`. Returns the hand-tuned fallback if
    pyutilz is unavailable or the kernel hasn't been tuned yet.
    """
    cache = _get_cache()
    if cache is False or cache is None:
        # pyutilz missing entirely -> source-code fallback.
        return _fallback_for_joint_size(joint_size)

    choice = cache.lookup("joint_hist_batched", n_samples=n_samples, joint_size=joint_size)
    if choice is not None:
        return {
            "kernel_variant": choice["kernel_variant"],
            "block_size": choice["block_size"],
        }

    # Cache miss; optionally auto-tune.
    if run_auto_tune:
        try:
            from . import auto_tune as _auto_tune
            _auto_tune.ensure_joint_hist_tuning(force=False)
            choice = cache.lookup(
                "joint_hist_batched", n_samples=n_samples, joint_size=joint_size,
            )
            if choice is not None:
                return {
                    "kernel_variant": choice["kernel_variant"],
                    "block_size": choice["block_size"],
                }
        except Exception as e:
            logger.debug("auto_tune sweep failed: %s", e)

    return _fallback_for_joint_size(joint_size)


def _fallback_for_joint_size(joint_size: int) -> dict:
    """Hand-tuned source-code defaults used when the cache is absent."""
    if joint_size > _SHARED_HIST_MAX_JOINT_FALLBACK:
        return dict(_FALLBACK_JOINT_HIST_GLOBAL)
    return dict(_FALLBACK_JOINT_HIST)


def reset_cache() -> None:
    """Drop the in-memory cache singleton; next lookup re-loads from disk
    (or re-runs auto-tune if forced). For tests + driver-update hooks."""
    global _CACHE_SINGLETON
    with _LOAD_LOCK:
        if _CACHE_SINGLETON not in (None, False):
            try:
                _CACHE_SINGLETON.reset()
            except Exception:
                pass
        _CACHE_SINGLETON = None


__all__ = ["lookup_joint_hist", "reset_cache"]
