"""Run the kernel sweep on cache miss and persist via the pyutilz cache.

Public entry: :func:`ensure_joint_hist_tuning`. Called from
``prewarm_fs_cupy_kernels`` (production startup) and from the dispatcher
on cache miss + ``run_auto_tune=True``.

The auto-tune walks a small (n_samples, nbins, block_size) grid. Once
the cache JSON is on disk, every future process loads it in ~1 ms via
``pyutilz.performance.kernel_tuning.cache.KernelTuningCache``.
"""
from __future__ import annotations

import itertools
import logging
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# Sweep axes. Kept small for first-run latency.
#
# Multi-axis dispatch (added in WAVE 4): regions emit separate
# ``nbins_x_max`` / ``nbins_y_max`` keys alongside the legacy
# ``joint_size_max`` (= product) so dispatchers that have explicit nbins
# values can match more precisely. The legacy ``joint_size`` lookup still
# works via the catch-all fallback path -- a region with finer-grained
# keys also satisfies the ``joint_size <=`` test by construction
# (joint_size = nbins_x * nbins_y).
_N_SAMPLES_AXIS = (200_000, 1_000_000)
_NBINS_AXIS: tuple[tuple[int, int], ...] = ((5, 5), (10, 10), (20, 20))
_BLOCK_SIZE_AXIS = (256, 512, 1024)


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




def _measure_single_region(
    n_samples: int, joint_size: int, n_iters: int = 3,
) -> Optional[dict]:
    """Re-measure ONE (n_samples, joint_size) point with the same variant
    + block_size axes used by the main sweep. Returns the winning region
    dict (with ``n_samples_max`` / ``joint_size_max`` caps) or None.

    Used by the online-relearn hook in ``dispatch._maybe_online_relearn``
    so that the per-call re-measure cost stays bounded (~50-200 ms,
    matching the docstring promise) instead of running the full grid
    sweep (~15-30 s).
    """
    import cupy as cp
    from mlframe.feature_selection.filters import gpu as _gpu_mod
    _gpu_mod._ensure_kernels_inited()

    # Pick (nbx, nby) closest to the requested joint_size, biased to
    # square shapes (matches the typical MRMR axis).
    nbx_nby = None
    for _nbx, _nby in _NBINS_AXIS:
        if _nbx * _nby == joint_size:
            nbx_nby = (_nbx, _nby)
            break
    if nbx_nby is None:
        # Pick the nearest joint_size from the sweep axes.
        _diffs = [(abs(a * b - joint_size), (a, b)) for a, b in _NBINS_AXIS]
        nbx_nby = min(_diffs)[1]
    nbx, nby = nbx_nby

    rng = np.random.default_rng(11)
    classes_x = rng.integers(0, nbx, size=n_samples).astype(np.int32)
    classes_y = rng.integers(0, nby, size=n_samples).astype(np.int32)
    d_x = cp.asarray(classes_x)
    d_y_perms = cp.asarray(classes_y.reshape(1, -1).copy())
    d_out = cp.zeros((1, nbx * nby), dtype=cp.int32)
    args = (d_x, d_y_perms, d_out,
            np.int32(n_samples), np.int32(nbx), np.int32(nby))

    best_wall = float("inf")
    best_choice = None
    for variant in ("shared", "global"):
        for bs in _BLOCK_SIZE_AXIS:
            d_out[:] = 0
            grid_x = (n_samples + bs - 1) // bs
            kernel = (_gpu_mod.compute_joint_hist_batched_shared_cuda
                      if variant == "shared"
                      else _gpu_mod.compute_joint_hist_batched_cuda)
            smem = nbx * nby * 4 if variant == "shared" else 0
            try:
                wall = _measure_one(kernel, grid_x, bs, args, n_iters=n_iters,
                                    shared_mem_bytes=smem)
            except Exception:
                continue
            if wall < best_wall:
                best_wall = wall
                best_choice = {"kernel_variant": variant, "block_size": bs,
                               "wall_ms": round(wall, 4)}
    if best_choice is None:
        return None
    return {
        "n_samples_max": int(n_samples),
        "joint_size_max": int(nbx * nby),
        "nbins_x_max": int(nbx),
        "nbins_y_max": int(nby),
        **best_choice,
    }


def _shared_cache():
    """Return the shared ``KernelTuningCache`` singleton via
    :mod:`mlframe.feature_selection.filters._kernel_tuning`, or ``None``
    when pyutilz / CUDA are unavailable. Using the shared singleton
    collapses N ``nvidia-smi`` subprocess spawns (one per fresh
    ``KernelTuningCache._load``) into one per process."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        if not is_cuda_available():
            return None
    except ImportError:
        return None
    from mlframe.feature_selection.filters._kernel_tuning import (
        get_kernel_tuning_cache,
    )
    return get_kernel_tuning_cache()












# ============================================================================
# Wave 24 populators (2026-05-20). Nine consumer sites had cache lookups but
# no populator -- every lookup fell through to source-code defaults. This
# block adds one (_run_sweep_X, ensure_X_tuning) pair per consumer site,
# mirroring the canonical patterns above:
#   * block_size sweeps   -> mirror _run_sweep_joint_hist
#   * backend-choice      -> mirror _run_sweep_mi_classif_dispatch
#   * crossover threshold -> mirror _run_sweep_polyeval
# Each helper is best-effort: graceful skip + empty regions on CUDA / cupy /
# optional-lib unavailability; never crash the parent process.
# ============================================================================


def _cuda_available_or_skip(kernel_name: str) -> bool:
    """Return True iff cupy + pyutilz are importable AND CUDA is up. Logs
    a single info line on skip so the CLI shows why a sweep produced 0
    regions. Mirrors the gate pattern used by ``_shared_cache``."""
    try:
        from pyutilz.core.pythonlib import is_cuda_available
    except ImportError:
        logger.info("auto_tune %s skipped: pyutilz unavailable", kernel_name)
        return False
    if not is_cuda_available():
        logger.info("auto_tune %s skipped: CUDA unavailable", kernel_name)
        return False
    try:
        import cupy as _cp  # noqa: F401
    except ImportError:
        logger.info("auto_tune %s skipped: cupy not installed", kernel_name)
        return False
    return True


# ----------------------------------------------------------------------------
# 1. joint_hist_single_perm -- block_size keyed by n_samples
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# 2. joint_hist_multi_pair -- block_size keyed by (n_rows, n_pairs)
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# 3. batch_pair_mi -- backend (njit / cuda / cupy) keyed by (n_samples, n_pairs)
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# 4. cat_fe_perm_kernel -- crossover_n keyed by (n_samples, n_perms)
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# 5. rmse_partial_sum -- block_n keyed by (n_samples, n_cols)
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# 6. unary_elementwise -- min_cells keyed by n_samples
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# 7. rff_matmul -- work_threshold keyed by work (= n * d)
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# 8. knn_hnsw_crossover -- n_threshold keyed by (n_subset, d)
# ----------------------------------------------------------------------------
def _hnswlib_importable() -> bool:
    """Probe ``import hnswlib`` in a subprocess. On some Windows builds
    the wheel segfaults at import time (access violation in the C
    extension); a ``try / except ImportError`` in the parent process
    won't catch that. Run the probe out-of-process so the parent stays
    alive when the wheel is broken."""
    import subprocess, sys
    try:
        r = subprocess.run(
            [sys.executable, "-c", "import hnswlib"],
            capture_output=True, timeout=15,
        )
    except Exception as exc:
        logger.info("auto_tune knn_hnsw_crossover probe failed: %s", exc)
        return False
    return r.returncode == 0






# ----------------------------------------------------------------------------
# 9. discretize_2d_array -- min_cells keyed by arr_size
# ----------------------------------------------------------------------------




__all__ = [
    "ensure_joint_hist_tuning",
    "ensure_mi_classif_dispatch_tuning",
    "ensure_polyeval_tuning",
    "ensure_joint_hist_single_perm_tuning",
    "ensure_joint_hist_multi_pair_tuning",
    "ensure_batch_pair_mi_tuning",
    "ensure_cat_fe_perm_kernel_tuning",
    "ensure_rmse_partial_sum_tuning",
    "ensure_unary_elementwise_tuning",
    "ensure_rff_matmul_tuning",
    "ensure_knn_hnsw_crossover_tuning",
    "ensure_discretize_2d_array_tuning",
]


# ----------------------------------------------------------------------
# Sibling-module re-exports. Sweep + ensure functions live in two
# siblings (group A / group B) so this file stays below the 1k-LOC
# monolith threshold.
# ----------------------------------------------------------------------
from ._auto_tune_sweeps_a import (  # noqa: E402,F401
    _run_sweep_batch_pair_mi, _run_sweep_joint_hist, _run_sweep_joint_hist_multi_pair, _run_sweep_joint_hist_single_perm, _run_sweep_mi_classif_dispatch, _run_sweep_polyeval, ensure_batch_pair_mi_tuning, ensure_joint_hist_multi_pair_tuning, ensure_joint_hist_single_perm_tuning, ensure_joint_hist_tuning, ensure_mi_classif_dispatch_tuning, ensure_polyeval_tuning,
)
from ._auto_tune_sweeps_b import (  # noqa: E402,F401
    _run_sweep_cat_fe_perm_kernel, _run_sweep_discretize_2d_array, _run_sweep_fe_mi_split, _run_sweep_knn_hnsw_crossover, _run_sweep_rff_matmul, _run_sweep_rmse_partial_sum, _run_sweep_unary_elementwise, ensure_cat_fe_perm_kernel_tuning, ensure_discretize_2d_array_tuning, ensure_fe_mi_split_tuning, ensure_knn_hnsw_crossover_tuning, ensure_rff_matmul_tuning, ensure_rmse_partial_sum_tuning, ensure_unary_elementwise_tuning,
)
