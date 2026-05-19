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

# Online-learning counter: every Nth ``lookup_joint_hist`` call we
# OPTIONALLY re-measure the chosen kernel + one alternative and update
# the cache if the alternative is faster. Off by default (gated on
# ``$MLFRAME_KTC_ONLINE_LEARN``) because it adds a one-call overhead
# of a couple milliseconds; users who want continuous self-tuning opt in.
_LEARN_COUNTER = 0
_LEARN_EVERY = 1000  # 0.1% sampling rate at default


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
        result = {
            "kernel_variant": choice["kernel_variant"],
            "block_size": choice["block_size"],
        }
        _maybe_online_relearn(n_samples, joint_size, result)
        return result

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


def _maybe_online_relearn(n_samples: int, joint_size: int, current_choice: dict) -> None:
    """Optional online relearn: every ``_LEARN_EVERY`` calls and when the
    env-var is enabled, time the current choice + 1 alternative and update
    the cache if the alternative wins. Adds ~5-10 ms per re-measure call
    (only 0.1% of total at default cadence), zero overhead at the other
    99.9% of calls.

    Gated behind ``MLFRAME_KTC_ONLINE_LEARN=1`` so production fits never
    pay the re-measure cost unless explicitly opted in.
    """
    import os
    if os.environ.get("MLFRAME_KTC_ONLINE_LEARN", "").strip().lower() in (
        "", "0", "false", "no", "off",
    ):
        return
    global _LEARN_COUNTER
    _LEARN_COUNTER += 1
    if _LEARN_COUNTER % _LEARN_EVERY != 0:
        return
    try:
        from . import auto_tune as _at
        # Re-measure ONE point (current size) with both variants + both
        # block_size candidates; pick winner; update cache if it differs
        # from current. Limited to a single (n_samples, joint_size) tuple
        # so the relearn cost stays bounded.
        regions = _at._run_sweep_joint_hist(n_iters=3)  # noqa: SLF001 - intentional
        if not regions:
            return
        cache = _get_cache()
        if cache and cache is not False:
            cache.update("joint_hist_batched",
                         axes=["n_samples", "joint_size"], regions=regions)
            logger.info(
                "online relearn: n=%d joint=%d cache updated (counter=%d)",
                n_samples, joint_size, _LEARN_COUNTER,
            )
    except Exception as exc:
        logger.debug("online relearn failed: %s", exc)


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
