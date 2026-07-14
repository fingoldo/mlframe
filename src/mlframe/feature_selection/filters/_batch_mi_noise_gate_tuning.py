"""CPU-vs-GPU backend chooser + kernel_tuning_cache sweep for ``batch_mi_noise_gate_gpu``.

Carved out of ``batch_mi_noise_gate_gpu.py`` to keep that module under the 1k-LOC ceiling. The GPU
kernels themselves stay in the parent module (and its own ``_batch_mi_noise_gate_kernels`` sibling);
this module only holds the sweep grid, the per-host tuned dispatch, and the CLI/registry glue. The
cuda/cupy entry points (``batch_mi_with_noise_gate_cuda`` / ``batch_mi_with_noise_gate_cupy``) are
imported lazily inside the functions below to avoid a circular import with the parent module, which
imports this module's functions at load time for re-export + ``@kernel_tuner`` registration.
"""
from __future__ import annotations

import time
import logging

import numpy as np

from .info_theory import batch_mi_with_noise_gate as _cpu_batch_mi_with_noise_gate
from ._batch_mi_noise_gate_kernels import (
    _CUDA_AVAIL,
    _CUPY_AVAIL,
    _mi_from_counts_cpu,
    _gate_from_mi,
    _build_shuffle_matrix,
)

# Measurement-backed fallback thresholds for the CPU-vs-GPU crossover, keyed on
# n_rows and n_cols (the FE batch shape). The CPU njit-prange kernel wins for
# small/medium batches (tiny joint-hist pass; the H2D copy of the discretized
# frame + per-shuffle bincount launch overhead dominate on GPU). GPU pays off for
# large K at moderate-to-large n. These are the source-code fallback; the live
# dispatch consults the per-host kernel_tuning_cache first (per
# feedback_use_kernel_tuning_cache_for_gpu) so consumer Ampere/Hopper cards learn
# their own crossover instead of inheriting the dev box's threshold.
#
# Dev-box measurement (cupy 13.6.0, cuda_available=True; nbins=10, nperm=3,
# n_classes_y=4), CPU = njit-prange, speedup = CPU/GPU:
#   n=700:  K=64 .16x  K=256 .16x  K=1024 .33-.45x  K=4096 .65-.82x  -> CPU all
#   n=2407: K=64 .26-.43x  K=256 cupy 1.09x  K=1024 ~9-11x  K=4096 ~8-11x -> GPU K>=256
#   n=100000 (LARGE-N, GTX 1050 Ti 4GB): K=256 26.3x  K=1024 32.9x  (bit-identical to
#     the CPU kernel); K=4096 GPU OOMs on the 4GB card (3.3GB alloc) -> CPU fallback.
# So the GPU win starts at n>=~2000 AND K>=256 on this host AND GROWS with n (the O(n*K)
# counting amortises the fixed H2D + launch overhead); below that the launch + H2D
# overhead loses to the tiny CPU joint-hist pass. (Note the CPU njit-prange has a sharp
# slowdown at n=2407,K>=1024 -- per-thread (nbins x K_y) joint buffers x 1024 columns
# thrash cache -- which widens the GPU win there; the cache captures it.)
GPU_MIN_ROWS = 2_000
GPU_MIN_COLS = 256

# LARGE-N ROUTING FIX (2026-06-08). The sweep grid previously topped out at
# n_rows=10000, so EVERY query beyond it (the n=50000/100000/1M FE batches the
# large-n MRMR path actually produces) fell through to the multi-dim grid's
# "catch-all" region -- whose winner is the decision at the LARGEST swept cell
# (n_rows=10000, n_cols=4096). On a card where that corner does not win for GPU
# (e.g. the 4GB 1050 Ti, where K=4096 OOMs -> CPU), the catch-all is CPU and the
# GPU path is DEAD for ALL large n at ALL K -- even though the GPU is measured
# 26-33x FASTER and bit-identical at n=100000, K=256/1024. Extending the grid to
# n_rows=50000/100000 makes the catch-all corner a genuine large-n cell, so on
# capable HW (>=8GB Ampere/Hopper -- e.g. the user's RTX 2070, where the
# (100000, n_cols[-1]) corner fits in memory and GPU wins) the cache routes large
# batches to GPU. On this 4GB box the top-K corner still OOMs -> the corner picks
# CPU (correct locally), but the per-cell large-n bands at GPU-friendly K
# (256/1024) now exist and the fallback heuristic (n>=2000 & K>=256 -> cupy)
# already routes large n to GPU before any sweep lands. The GPU variants gracefully
# skip OOMing sweep cells (sweep_backend_grid try/excepts each cell), so adding
# large-n cells never breaks the sweep on small cards.
_BMING_SWEEP_N_ROWS = [700, 2_407, 10_000, 50_000, 100_000]
_BMING_SWEEP_N_COLS = [64, 256, 1_024, 4_096]
_BMING_SWEEP_NBINS = 10
_BMING_SWEEP_N_CLASSES_Y = 4
_BMING_SWEEP_NPERM = 3
_BMING_SWEEP_MNC = 0.99
_BMING_SALT = 2  # bump on any numerics change / grid change to invalidate stale per-host cache


def _make_batch_mi_noise_gate_inputs(dims: dict):
    """Synthetic (disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
    npermutations, base_seed, min_nonzero_confidence, use_su, dtype) tuple at
    ``dims['n_rows']`` rows x ``dims['n_cols']`` candidate columns, matching the
    CPU/GPU kernel call signature."""
    rng = np.random.default_rng(0)
    n = int(dims["n_rows"])
    K = int(dims["n_cols"])
    nbins = _BMING_SWEEP_NBINS
    disc_2d = rng.integers(0, nbins, size=(n, K)).astype(np.int32)
    factors_nbins = np.full(K, nbins, dtype=np.int64)
    classes_y = rng.integers(0, _BMING_SWEEP_N_CLASSES_Y, size=n).astype(np.int32)
    classes_y_safe = classes_y.copy()
    freqs_y = np.bincount(classes_y, minlength=_BMING_SWEEP_N_CLASSES_Y).astype(np.float64) / max(1, n)
    return (
        disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
        _BMING_SWEEP_NPERM, np.uint64(0), _BMING_SWEEP_MNC, False, np.int32,
    )


def _run_batch_mi_noise_gate_sweep() -> list:
    """Full (n_rows x n_cols) grid sweep -> backend_choice regions: cpu / cuda /
    cupy, fastest BIT-IDENTICAL variant per cell. The GPU variants are exact (they
    only move the integer counting to the GPU; entropy stays on the bit-exact CPU
    path), so equivalence holds at array_equal -- a tight rtol/atol is used for the
    sweep's equivalence harness."""
    from pyutilz.dev.benchmarking import sweep_backend_grid
    from .batch_mi_noise_gate_gpu import batch_mi_with_noise_gate_cuda, batch_mi_with_noise_gate_cupy

    variants = {
        "cpu": lambda *a: _cpu_batch_mi_with_noise_gate(*a),
    }
    if _CUDA_AVAIL:
        variants["cuda"] = lambda *a: batch_mi_with_noise_gate_cuda(*a)
    if _CUPY_AVAIL:
        variants["cupy"] = lambda *a: batch_mi_with_noise_gate_cupy(*a)
    # MEMORY-AWARE grid filter. The CPU reference kernel allocates ~int64 (n, K) intermediates, so the
    # top cell (100k x 4096 ~ 3 GiB) OOMs the HOST on RAM-tight boxes and -- because the per-cell guard
    # does not cover the shared input-gen + reference allocation -- kills the WHOLE sweep, leaving 0
    # regions (so the dispatch is stuck on the fallback heuristic forever). Drop the n_cols that would
    # exceed a fraction of free host RAM at the largest n_row, so the sweep COMPLETES with the runnable
    # cells (partial per-host tuning beats no tuning). The cartesian grid only lets us filter whole
    # columns; that is acceptable -- the dropped large-K-at-large-n cells are exactly the OOM ones, and
    # the fallback (cupy at K>=256) already routes them correctly.
    n_rows = list(_BMING_SWEEP_N_ROWS)
    n_cols = list(_BMING_SWEEP_N_COLS)
    try:
        import psutil
        free = int(psutil.virtual_memory().available)
    except Exception:
        free = 4 * 1024**3
    budget = int(free * 0.4)
    max_n = max(n_rows) if n_rows else 1
    fitting = [k for k in n_cols if max_n * int(k) * 8 * 3 <= budget]
    n_cols = fitting or [min(n_cols)]  # always keep at least the smallest column
    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        variants,
        {"n_rows": n_rows, "n_cols": n_cols},
        _make_batch_mi_noise_gate_inputs,
        reference="cpu",
        repeats=3, equiv_rtol=1e-9, equiv_atol=1e-12,
    )


def _batch_mi_noise_gate_code_version():
    """code_version over the CPU body + GPU bodies + the shared bit-exact reducer;
    re-tunes on any kernel edit."""
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version
        from .batch_mi_noise_gate_gpu import batch_mi_with_noise_gate_cuda, batch_mi_with_noise_gate_cupy

        fns = [_cpu_batch_mi_with_noise_gate, _mi_from_counts_cpu, _gate_from_mi]
        if _CUDA_AVAIL:
            fns.append(batch_mi_with_noise_gate_cuda)
        if _CUPY_AVAIL:
            fns.append(batch_mi_with_noise_gate_cupy)
            fns.append(_build_shuffle_matrix)
        return compute_code_version(*fns, salt=_BMING_SALT)
    except Exception:
        return None


def ensure_batch_mi_noise_gate_tuning(force: bool = False):
    """Force-run the ``batch_mi_noise_gate`` CPU-vs-GPU sweep and persist it per-host.

    This is the CLI / ``refresh-all`` entry point that was MISSING -- without it the noise-gate sweep
    only ever fired ASYNC during the first real fit (a multi-minute grid sweep that thrashes the GPU
    mid-MRMR). Pre-running it via the CLI avoids that first-fit cost. Persists with the SAME
    ``code_version`` + ``axes`` the live dispatch's ``get_or_tune`` uses (see
    ``_batch_mi_noise_gate_backend_choice``), so the tuned regions are a HIT for production. Returns the
    region list (``[]``/``None`` if cupy/CUDA absent or the sweep fails -> caller reports a skip)."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
    except Exception:
        return None
    cache = KernelTuningCache.load_or_create()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("batch_mi_noise_gate")
        if regions:
            return regions
    log = logging.getLogger(__name__)
    log.info("kernel_tuning_cache: batch_mi_noise_gate sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_batch_mi_noise_gate_sweep()
    except Exception as e:
        log.warning("kernel_tuning_cache: batch_mi_noise_gate sweep failed: %s", e)
        return None
    log.info("kernel_tuning_cache: batch_mi_noise_gate sweep done in %.2fs", time.perf_counter() - t0)
    if regions:
        try:
            cache.update(
                "batch_mi_noise_gate", axes=["n_rows", "n_cols"], regions=regions,
                code_version=_batch_mi_noise_gate_code_version(),
            )
        except OSError as e:
            log.warning("kernel_tuning_cache: batch_mi_noise_gate save failed: %s", e)
    return regions


def _batch_mi_noise_gate_fallback_choice(n_rows: int, n_cols: int) -> str:
    """Pre-sweep heuristic: GPU only for large n AND large K (where the O(n*K)
    counting amortises the H2D copy + per-shuffle launch overhead); CPU otherwise.
    Prefers cupy over cuda (single batched bincount per shuffle vs per-column
    block launch)."""
    if n_rows >= GPU_MIN_ROWS and n_cols >= GPU_MIN_COLS:
        if _CUPY_AVAIL:
            return "cupy"
        if _CUDA_AVAIL:
            return "cuda"
    return "cpu"


def _batch_mi_noise_gate_backend_choice(n_rows: int, n_cols: int) -> str:
    """Per-host backend (cpu/cuda/cupy) for this (n_rows, n_cols) via the shared
    get_or_tune orchestrator; measurement-backed threshold fallback."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        result = KernelTuningCache.load_or_create().get_or_tune(
            "batch_mi_noise_gate",
            dims={"n_rows": int(n_rows), "n_cols": int(n_cols)},
            tuner=_run_batch_mi_noise_gate_sweep,
            axes=["n_rows", "n_cols"],
            fallback={"backend_choice": _batch_mi_noise_gate_fallback_choice(n_rows, n_cols)},
            code_version=_batch_mi_noise_gate_code_version(),
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        # Legacy "gpu" region (pre-cupy/cuda split) -> resolve to the available GPU backend.
        if bc == "gpu":
            bc = "cupy" if _CUPY_AVAIL else ("cuda" if _CUDA_AVAIL else "cpu")
        if bc in ("cpu", "cuda", "cupy"):
            return bc
    except Exception as e:
        logging.getLogger(__name__).debug("batch_mi_noise_gate get_or_tune failed: %s", e)
    return _batch_mi_noise_gate_fallback_choice(n_rows, n_cols)
