"""Runtime dispatcher for the mlframe joint-hist kernels.

Thin wrapper that delegates storage + lookup to the generic
``pyutilz.performance.kernel_tuning.cache.KernelTuningCache``. This module
keeps the mlframe-specific entry point ``lookup_joint_hist`` so
``filters/gpu.py:mi_direct_gpu_batched`` doesn't need to know about the
generic backing storage; it also owns the hand-tuned fallbacks used
when the cache is missing AND auto-tune hasn't been triggered yet.
"""
from __future__ import annotations

import logging

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


# Process-lifetime cache of the GPU compute-capability MAJOR version. The
# fallback table is keyed only on ``cc_major``, which is IMMUTABLE for the
# process. ``gpu_capability_summary`` re-probes LIVE free-VRAM on every call by
# shelling out to ``nvidia-smi`` via GPUtil (a ~50ms subprocess spawn); a
# cProfile of ``mi_direct_gpu_batched`` (n=5000, nperm=1024) on a cc 6.1 host
# showed that subprocess at 45% of the GPU-path wall because this fallback ran
# once PER batched-MI call on the kernel-tuning-cache MISS path. We only need the
# static cc_major, so probe ONCE and memoise. ``None`` = "not yet probed";
# ``-1`` = "probed, unavailable" (so we never re-shell after a CPU-only / error
# result). The live-VRAM fields of the summary are intentionally discarded --
# kernel-variant selection is a function of (cc_major, joint_size) only.
_CC_MAJOR_CACHE: "int | None" = None


def _cached_cc_major() -> int:
    """GPU compute-capability major version, probed once per process.

    Returns ``-1`` when no GPU / probe failed (callers then take the generic,
    cc-agnostic fallback). Avoids the per-call ``nvidia-smi`` subprocess that
    ``gpu_capability_summary`` incurs for its live-VRAM fields, which this hot
    dispatch path does not use."""
    global _CC_MAJOR_CACHE  # noqa: PLW0603 - process-lifetime memo by design
    if _CC_MAJOR_CACHE is not None:
        return _CC_MAJOR_CACHE
    cc = -1
    try:
        from pyutilz.system.gpu_dispatch import gpu_capability_summary
        summary = gpu_capability_summary(0)
        if summary is not None:
            cc = int(summary.get("cc_major", 0))
    except Exception:
        cc = -1
    _CC_MAJOR_CACHE = cc
    return cc


def _hw_aware_fallback(joint_size: int) -> dict:
    """Pick fallback based on GPU compute capability. The cc_major probe is
    cached for the process lifetime (see ``_cached_cc_major``); this routine
    adds no per-call subprocess cost."""
    cc_major = _cached_cc_major()
    if cc_major >= 0:
        cc_entry = _FALLBACK_BY_CC.get(cc_major)
        if cc_entry is not None:
            # cc-9+ prefers global for large joint sizes by default.
            if joint_size > _SHARED_HIST_MAX_JOINT_FALLBACK:
                return dict(_FALLBACK_JOINT_HIST_GLOBAL)
            return dict(cc_entry)
    if joint_size > _SHARED_HIST_MAX_JOINT_FALLBACK:
        return dict(_FALLBACK_JOINT_HIST_GLOBAL)
    return dict(_FALLBACK_JOINT_HIST)


# Online-learning counter: every Nth ``lookup_joint_hist`` call we
# OPTIONALLY re-measure the chosen kernel + one alternative and update
# the cache if the alternative is faster. Off by default (gated on
# ``$MLFRAME_KTC_ONLINE_LEARN``) because it adds a one-call overhead
# of a couple milliseconds; users who want continuous self-tuning opt in.
_LEARN_COUNTER = 0
_LEARN_EVERY = 1000  # 0.1% sampling rate at default


def _get_cache():
    """Lazy import + singleton init of the pyutilz cache.

    Delegates to the shared ``filters._kernel_tuning.get_kernel_tuning_cache``
    singleton so this module and the hot-path filters (discretization, gpu)
    share ONE KernelTuningCache instance per process — collapses N
    ``nvidia-smi`` subprocess spawns (one per fresh KernelTuningCache._load)
    into one. Returns the cache instance or ``False`` on import miss to keep
    backward compat with callers that test ``cache is False``.
    """
    from mlframe.feature_selection.filters._kernel_tuning import get_kernel_tuning_cache
    cache = get_kernel_tuning_cache()
    return cache if cache is not None else False


def lookup_joint_hist(n_samples: int, joint_size: int,
                       *, run_auto_tune: bool = False) -> dict:
    """Return ``{"kernel_variant", "block_size"}`` for the given size pair.

    Hits the pyutilz ``KernelTuningCache`` for ``joint_hist_batched``.
    On cache miss + ``run_auto_tune=True`` triggers a one-time sweep
    (~30s) via :mod:`auto_tune`. Returns the hand-tuned fallback if
    pyutilz is unavailable or the kernel hasn't been tuned yet.
    """
    cache = _get_cache()
    fb = _fallback_for_joint_size(joint_size)
    if cache is False or cache is None:
        # pyutilz missing entirely -> source-code fallback.
        return fb

    # get_or_tune orchestrates env -> code-version-checked lookup -> on-miss
    # (locked, once-per-process) sweep -> persist -> re-lookup -> fallback. The
    # tuner only sweeps when run_auto_tune (else a no-op so a miss falls straight
    # to the fallback, preserving the prior gating). code_version (salt=1, matching
    # the @kernel_tuner registration) invalidates stale regions after a kernel edit
    # -- the win over the old bare lookup. Multi-field payload {kernel_variant,
    # block_size} passes through unchanged.
    payload = fb
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version

        from ._auto_tune_sweeps_a import _run_sweep_joint_hist

        tuner = (lambda: _run_sweep_joint_hist(n_iters=2)) if run_auto_tune else (lambda: [])
        result = cache.get_or_tune(
            "joint_hist_batched",
            dims={"n_samples": n_samples, "joint_size": joint_size},
            tuner=tuner, axes=["n_samples", "joint_size"],
            fallback=fb, code_version=compute_code_version(_run_sweep_joint_hist, salt=1),
        )
        if isinstance(result, dict) and "kernel_variant" in result:
            payload = {"kernel_variant": result["kernel_variant"], "block_size": result["block_size"]}
    except Exception as e:
        logger.debug("lookup_joint_hist get_or_tune failed: %s", e)

    _maybe_online_relearn(n_samples, joint_size, payload)
    return payload


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


def lookup_mi_classif_backend(n_samples: int, k: int,
                               *, run_auto_tune: bool = False) -> str:
    """Return ``"njit"`` or ``"cuda"`` for the plug-in MI dispatcher.

    Hits the pyutilz ``KernelTuningCache`` for ``plugin_mi_classif_dispatch``.
    On cache miss + ``run_auto_tune=True`` triggers a one-time sweep
    via :func:`auto_tune.ensure_mi_classif_dispatch_tuning`. Returns
    the measurement-backed hardcoded fallback (75k single / 10k batch
    on GTX 1050 Ti cc 6.1, 2026-05-20) if pyutilz is unavailable or
    the kernel hasn't been tuned yet.

    The fallback values are conservative per-HW measurements; the
    auto-tune sweep refines per-host (e.g. A100 / H100 will have lower
    crossovers due to faster atomics + bus). Sweep runs in ~10-30s
    once per host and persists via the same JSON file used by
    ``joint_hist_batched``.
    """
    cache = _get_cache()
    fb = _fallback_mi_backend(n_samples, k)
    if cache is False or cache is None:
        return fb

    # get_or_tune: code-version-checked lookup -> on-miss (locked) sweep when
    # run_auto_tune, else fallback. salt=1 matches the @kernel_tuner registration,
    # so a kernel edit invalidates stale regions on the dispatch path.
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version

        from ._auto_tune_sweeps_a import _run_sweep_mi_classif_dispatch

        tuner = (lambda: _run_sweep_mi_classif_dispatch(n_iters=2)) if run_auto_tune else (lambda: [])
        result = cache.get_or_tune(
            "plugin_mi_classif_dispatch",
            dims={"n_samples": n_samples, "k": k},
            tuner=tuner, axes=["n_samples", "k"],
            fallback={"backend_choice": fb},
            code_version=compute_code_version(_run_sweep_mi_classif_dispatch, salt=1),
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        if bc in ("njit", "cuda"):
            return bc
    except Exception as e:
        logger.debug("lookup_mi_classif_backend get_or_tune failed: %s", e)

    return fb


def _fallback_mi_backend(n_samples: int, k: int) -> str:
    """Conservative fallback when the per-host cache has no measured verdict.

    Always ``njit``. The earlier GPU-favoring constants (cuda from n>=75k single /
    n>=10k batch, "measured 2026-05-20 on GTX 1050 Ti") came from a SOLO microbenchmark
    that gave the GPU the card to itself. The production FE pipeline instead fires these
    MI calls from many joblib worker threads contending on one GPU, where each call pays
    H2D/D2H + serialised launch/sync -- a ~700ms fixed per-call penalty the solo bench
    never saw, making the real per-call cuda wall 20-70x the solo time (end-to-end A/B on
    the canonical 5-feature/n=100k fit: GPU 318-368s/5.0GB vs njit 115s/1.6GB, identical
    selection). So with no contention-aware measurement on hand, njit is the safe default;
    the concurrency-aware sweep (``_run_sweep_mi_classif_dispatch``) overrides this per host
    where cuda genuinely wins UNDER contention, and ``MLFRAME_MI_BACKEND=cuda`` forces GPU.
    """
    return "njit"


def reset_cache() -> None:
    """Drop the in-memory cache singleton; next lookup re-loads from disk. The
    singleton lives in ``filters._kernel_tuning`` (``_get_cache`` delegates there),
    so reset it at its source. For tests + driver-update hooks."""
    from mlframe.feature_selection.filters import _kernel_tuning as _kt

    with _kt._LOAD_LOCK:
        if _kt._CACHE_SINGLETON not in (None, False):
            try:
                _kt._CACHE_SINGLETON.reset()
            except Exception:
                pass
        _kt._CACHE_SINGLETON = None


__all__ = ["lookup_joint_hist", "lookup_mi_classif_backend", "reset_cache"]
