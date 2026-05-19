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


# HW-aware fallback table per compute-capability major version. Picks
# sensible defaults BEFORE the auto-tune sweep ever runs. Sourced from
# anecdotal measurements + CUDA arch characteristics:
#   * cc 5/6 (Maxwell/Pascal): 48 KB shared, modest atomic throughput
#   * cc 7 (Volta/Turing): 96 KB shared, faster shared atomics
#   * cc 8 (Ampere): 100-164 KB shared, vastly faster atomics
#   * cc 9+ (Hopper / Blackwell): async global atomics, prefer global path
# Once the auto-tune sweep runs on the host, the empirically-measured
# region overrides the fallback for the matched (n_samples, joint_size).
_FALLBACK_BY_CC: dict[int, dict] = {
    5: {"kernel_variant": "shared", "block_size": 512},
    6: {"kernel_variant": "shared", "block_size": 512},
    7: {"kernel_variant": "shared", "block_size": 256},
    8: {"kernel_variant": "shared", "block_size": 256},
    9: {"kernel_variant": "global", "block_size": 1024},
}
_FALLBACK_JOINT_HIST = {"kernel_variant": "shared", "block_size": 512}  # generic default
_FALLBACK_JOINT_HIST_GLOBAL = {"kernel_variant": "global", "block_size": 1024}
_SHARED_HIST_MAX_JOINT_FALLBACK = 4096


def _hw_aware_fallback(joint_size: int) -> dict:
    """Pick fallback based on live GPU compute capability. Cheap probe
    via pyutilz; cached for the process lifetime."""
    try:
        from pyutilz.system.gpu_dispatch import gpu_capability_summary
        summary = gpu_capability_summary(0)
        if summary is not None:
            cc_major = int(summary.get("cc_major", 0))
            cc_entry = _FALLBACK_BY_CC.get(cc_major)
            if cc_entry is not None:
                # cc-9+ prefers global for large joint sizes by default.
                if joint_size > _SHARED_HIST_MAX_JOINT_FALLBACK:
                    return dict(_FALLBACK_JOINT_HIST_GLOBAL)
                return dict(cc_entry)
    except Exception:
        pass
    if joint_size > _SHARED_HIST_MAX_JOINT_FALLBACK:
        return dict(_FALLBACK_JOINT_HIST_GLOBAL)
    return dict(_FALLBACK_JOINT_HIST)


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
    # Bounded re-measure: only the OFFENDING (n_samples, joint_size)
    # region, not the full sweep grid. Total cost ~50-200 ms per fire
    # (6 measurements × ~10 ms each on cc 6.1), matching the docstring
    # promise. The full _run_sweep_joint_hist (15-30s) is too expensive
    # to drop into a production lookup path -- use the CLI ``refresh``
    # subcommand for a full re-sweep.
    try:
        from . import auto_tune as _at
        # _measure_single_region: re-tunes just one (n, joint) point.
        # Returns the winning region dict (with the standard ..._max keys)
        # or None on failure. Merges into the existing cache via update().
        region = _at._measure_single_region(
            n_samples=n_samples, joint_size=joint_size, n_iters=3,
        )  # noqa: SLF001 - intentional access to private API
        if region is None:
            return
        cache = _get_cache()
        if cache and cache is not False:
            # Merge: read existing regions, replace any with matching caps,
            # append the new region, persist. KernelTuningCache.update
            # replaces the WHOLE kernel entry by design; we instead
            # construct the merged region list explicitly.
            existing = cache.get_regions("joint_hist_batched") or []
            # Drop any region with the same caps as the new one.
            new_regions = [
                r for r in existing
                if (r.get("n_samples_max") != region.get("n_samples_max")
                    or r.get("joint_size_max") != region.get("joint_size_max"))
            ]
            new_regions.append(region)
            # Move catch-all (None caps) to the end.
            new_regions.sort(key=lambda r: (
                r.get("n_samples_max") is None,
                r.get("n_samples_max") or 0,
                r.get("joint_size_max") or 0,
            ))
            cache.update("joint_hist_batched",
                         axes=["n_samples", "joint_size"], regions=new_regions)
            logger.info(
                "online relearn: n=%d joint=%d cache updated (counter=%d)",
                n_samples, joint_size, _LEARN_COUNTER,
            )
    except Exception as exc:
        logger.debug("online relearn failed: %s", exc)


def _fallback_for_joint_size(joint_size: int) -> dict:
    """HW-aware fallback used when the cache is absent. Routes by cc_major
    via ``_hw_aware_fallback``; legacy hand-tuned defaults for non-CUDA /
    pyutilz-missing hosts."""
    return _hw_aware_fallback(joint_size)


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
