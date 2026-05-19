"""Runtime dispatcher: pick the best (kernel_variant, block_size) per call.

Public API:
    ``lookup_joint_hist(n_samples, joint_size) -> dict``
        Returns a dict ``{"kernel_variant": "shared"|"global", "block_size": int}``.
        Loads the cache lazily on first call; on cache miss falls back to
        the hand-tuned defaults from the source code so the dispatcher
        never blocks production fits even before the first sweep ran.

The cache is consulted ONCE per process via ``_load_or_build`` (under a
threading.Lock so concurrent callers serialise). After that every lookup
is a Python dict scan over a handful of regions -- O(N_regions) per
call but N_regions <= 20.
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from . import auto_tune, cache_io

logger = logging.getLogger(__name__)


_LOAD_LOCK = threading.Lock()
_LOADED: dict[str, list[dict]] = {}  # kernel_name -> regions list

# Hand-tuned fallbacks used before / instead of the sweep cache. These
# match the constants in ``filters/gpu.py:mi_direct_gpu_batched`` so the
# dispatcher's "first-process / no cache yet" decision is identical to
# the hand-coded one.
_FALLBACK_JOINT_HIST = {
    "kernel_variant": "shared",  # current source-code default
    "block_size": 512,
}
_FALLBACK_JOINT_HIST_GLOBAL = {
    "kernel_variant": "global",
    "block_size": 1024,
}
_SHARED_HIST_MAX_JOINT_FALLBACK = 4096


def _load_or_build(kernel_name: str, *, auto_tune_fn=None) -> Optional[list[dict]]:
    """Idempotent loader: cache -> auto_tune -> None. Under a lock so
    only one thread per process pays the sweep cost."""
    with _LOAD_LOCK:
        if kernel_name in _LOADED:
            return _LOADED[kernel_name]
        cached = cache_io.load() or {}
        kernels = cached.get("kernels", {})
        entry = kernels.get(kernel_name)
        if entry and entry.get("regions"):
            _LOADED[kernel_name] = entry["regions"]
            return _LOADED[kernel_name]
        # Cache miss; trigger auto-tune if a tuner was provided.
        if auto_tune_fn is not None:
            regions = auto_tune_fn()
            if regions:
                _LOADED[kernel_name] = regions
                return regions
        _LOADED[kernel_name] = []  # cache miss, no tuner: store empty so we don't retry
        return None


def lookup_joint_hist(n_samples: int, joint_size: int,
                       *, run_auto_tune: bool = False) -> dict:
    """Return ``{"kernel_variant", "block_size"}`` for the given size pair.

    If ``run_auto_tune`` is True, runs the per-host sweep on cache miss;
    otherwise returns the source-code fallback immediately and a
    background tuner can run later (via ``prewarm_fs_cupy_kernels``).
    """
    regions = _load_or_build(
        "joint_hist_batched",
        auto_tune_fn=auto_tune.ensure_joint_hist_tuning if run_auto_tune else None,
    )

    if not regions:
        # No cache, no auto-tune -> hand-tuned fallback. Mirrors the
        # source-code default in ``mi_direct_gpu_batched``.
        if joint_size > _SHARED_HIST_MAX_JOINT_FALLBACK:
            return dict(_FALLBACK_JOINT_HIST_GLOBAL)
        return dict(_FALLBACK_JOINT_HIST)

    for region in regions:
        n_max = region.get("n_samples_max")
        j_max = region.get("joint_size_max")
        if (n_max is None or n_samples <= n_max) and (j_max is None or joint_size <= j_max):
            return {
                "kernel_variant": region["kernel_variant"],
                "block_size": region["block_size"],
            }

    # Should never happen: the saved catch-all has None bounds. Defensive:
    return dict(_FALLBACK_JOINT_HIST)


def reset_cache() -> None:
    """Drop the in-memory dispatch table; next lookup re-loads from disk
    (or re-runs auto-tune if forced). For tests + on driver-update
    cache-invalidation hooks."""
    with _LOAD_LOCK:
        _LOADED.clear()


__all__ = ["lookup_joint_hist", "reset_cache"]
