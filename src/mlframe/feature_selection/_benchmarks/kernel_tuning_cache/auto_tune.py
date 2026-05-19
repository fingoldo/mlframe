"""Run the kernel sweep on cache miss and persist the result.

Public entry: :func:`ensure_joint_hist_tuning`. Called from
``prewarm_fs_cupy_kernels`` (which is called from production startup)
and from the dispatcher on cache miss.

The auto-tune walks a small set of (n_samples, joint_size, block_size)
points -- intentionally smaller than ``bench_gpu_kernel_sweep`` so the
first-process-per-host cost stays under ~30s. Once the JSON is on disk,
every future process loads it in ~1 ms.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Optional

import numpy as np

from . import cache_io

logger = logging.getLogger(__name__)


# Sweep axes. Kept small for first-run latency; can be widened in a
# subsequent on-demand re-tune.
_N_SAMPLES_AXIS = (200_000, 1_000_000)
_NBINS_AXIS: tuple[tuple[int, int], ...] = ((5, 5), (10, 10), (20, 20))  # joint_size = 25, 100, 400
_BLOCK_SIZE_AXIS = (256, 512, 1024)
_BATCH = 1  # one row per kernel launch; cuts dispatch noise


def _measure_one(kernel, grid_x: int, block_size: int, args: tuple,
                 n_iters: int, shared_mem_bytes: int) -> float:
    """Min-of-N wall in ms after one warm-up call."""
    import cupy as cp
    if shared_mem_bytes > 0:
        kernel((grid_x,), (block_size,), args, shared_mem=shared_mem_bytes)
    else:
        kernel((grid_x,), (block_size,), args)
    cp.cuda.runtime.deviceSynchronize()
    best = float("inf")
    for _ in range(n_iters):
        t0 = time.perf_counter()
        if shared_mem_bytes > 0:
            kernel((grid_x,), (block_size,), args, shared_mem=shared_mem_bytes)
        else:
            kernel((grid_x,), (block_size,), args)
        cp.cuda.runtime.deviceSynchronize()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def _run_sweep_joint_hist(n_iters: int = 5) -> list[dict]:
    """Returns a list of region dicts ready to drop into the cache."""
    import cupy as cp
    # Re-fetch kernel references AFTER ``_ensure_kernels_inited`` rebinds
    # the module attributes. A direct ``from .filters.gpu import
    # compute_joint_hist_batched_cuda`` at module-import time grabs the
    # pre-init ``None`` and never updates -- silent NoneType errors below.
    from mlframe.feature_selection.filters import gpu as _gpu_mod
    _gpu_mod._ensure_kernels_inited()
    compute_joint_hist_batched_cuda = _gpu_mod.compute_joint_hist_batched_cuda
    compute_joint_hist_batched_shared_cuda = _gpu_mod.compute_joint_hist_batched_shared_cuda

    rng = np.random.default_rng(11)
    best_per_combo: dict[tuple[int, int], dict] = {}

    for n_samples, (nbx, nby) in itertools.product(_N_SAMPLES_AXIS, _NBINS_AXIS):
        joint = nbx * nby
        classes_x = rng.integers(0, nbx, size=n_samples).astype(np.int32)
        classes_y = rng.integers(0, nby, size=n_samples).astype(np.int32)
        d_x = cp.asarray(classes_x)
        d_y_perms = cp.asarray(classes_y.reshape(1, -1).copy())
        d_out = cp.zeros((_BATCH, joint), dtype=cp.int32)
        args = (d_x, d_y_perms, d_out,
                np.int32(n_samples), np.int32(nbx), np.int32(nby))

        best_wall = float("inf")
        best_choice = None
        for variant, bs in itertools.product(("shared", "global"), _BLOCK_SIZE_AXIS):
            d_out[:] = 0
            grid_x = (n_samples + bs - 1) // bs
            kernel = (compute_joint_hist_batched_shared_cuda if variant == "shared"
                      else compute_joint_hist_batched_cuda)
            smem = joint * 4 if variant == "shared" else 0
            try:
                wall = _measure_one(kernel, grid_x, bs, args, n_iters=n_iters,
                                    shared_mem_bytes=smem)
            except Exception as e:
                logger.debug("auto_tune skipped (variant=%s bs=%d): %s", variant, bs, e)
                continue
            if wall < best_wall:
                best_wall = wall
                best_choice = {"kernel_variant": variant, "block_size": bs,
                               "wall_ms": round(wall, 4)}
        if best_choice is not None:
            best_per_combo[(n_samples, joint)] = best_choice
            logger.info(
                "auto_tune joint_hist n=%d joint=%d -> %s bs=%d (%.3fms)",
                n_samples, joint, best_choice["kernel_variant"],
                best_choice["block_size"], best_choice["wall_ms"],
            )

    # Build regions sorted ascending by (n_samples_max, joint_size_max).
    regions: list[dict] = []
    for (n_samples, joint), choice in sorted(best_per_combo.items()):
        regions.append({
            "n_samples_max": n_samples,
            "joint_size_max": joint,
            "kernel_variant": choice["kernel_variant"],
            "block_size": choice["block_size"],
            "wall_ms": choice["wall_ms"],
        })
    # Catch-all fallback for above-axis sizes.
    regions.append({
        "n_samples_max": None,
        "joint_size_max": None,
        "kernel_variant": "shared",
        "block_size": 512,
        "wall_ms": None,
    })
    return regions


def ensure_joint_hist_tuning(force: bool = False) -> Optional[list[dict]]:
    """Return cached regions for ``joint_hist_batched``; run the sweep
    + persist if missing. Returns None if CUDA is unavailable."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            return None
    except Exception:
        return None

    if not force:
        cached = cache_io.load()
        if cached is not None:
            kernels = cached.get("kernels", {})
            entry = kernels.get("joint_hist_batched")
            if entry and entry.get("regions"):
                return entry["regions"]

    logger.info("kernel_tuning_cache: joint_hist sweep starting (one-time per host)")
    t0 = time.perf_counter()
    try:
        regions = _run_sweep_joint_hist(n_iters=5)
    except Exception as e:
        logger.warning("kernel_tuning_cache: joint_hist sweep failed: %s", e)
        return None
    logger.info("kernel_tuning_cache: joint_hist sweep done in %.2fs", time.perf_counter() - t0)

    try:
        cache_io.update_kernel("joint_hist_batched", regions)
    except OSError as e:
        logger.warning("kernel_tuning_cache: cache save failed: %s", e)

    return regions


__all__ = ["ensure_joint_hist_tuning"]
