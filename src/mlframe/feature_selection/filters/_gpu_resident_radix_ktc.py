"""Lever B (2026-06-23): kernel-tuning-cache integration for the radix-select block size (threads/block).

Carved as a sibling of ``_gpu_resident_select.py`` (LOC budget + acyclic import graph, mirroring the G3
``_gpu_resident_k_chunk_ktc`` pattern). The ``radix_select_f32`` / ``radix_select_f64`` quantile-edge
kernels launch one block per candidate column with a hardcoded 512 threads/block. nsys (isolated KTC, F2
100k, one fit) showed radix_select_f32 = 3964ms = the single biggest kernel (~54% of GPU kernel time after
Lever A). A production-shape microbench (n=100k, K=583, R=38) measured the block size is materially
under-tuned on the GTX 1050 Ti: threads/block 256 -> 117ms, 512 -> 78ms, 768 -> 68ms, 1024 -> 65ms (the
n-loop over the column parallelises across threads; the per-rank 256-bucket scan stays serial in tid==0).
512 -> 1024 is a 1.20x kernel speedup AT THIS SHAPE.

Per ``feedback_use_kernel_tuning_cache_for_gpu`` the block size is NOT hardcoded: it is looked up per-host
from the kernel_tuning_cache (sweep keyed on n), so a different card learns its own sweet spot instead of
inheriting the dev box's. Block size NEVER changes the produced order statistics (the histogram is summed
over the same column values regardless of how many threads cooperate) -> the edges / codes / MI / selection
are BIT-IDENTICAL for every thread count; the sweep ranks variants by WALL only. The original 512 stays the
fallback (pre-sweep / no-cupy / lookup failure), so the CPU / no-CUDA path is byte-for-byte unchanged.
"""
from __future__ import annotations

import numpy as np

# Candidate threads/block to sweep. Powers + 768 spanning the GTX 1050 Ti measured curve; a higher-VRAM /
# higher-SM card can learn a different point. 512 = the historical hardcoded default (and the fallback).
_RADIX_THREADS_VARIANTS = [256, 512, 768, 1024]
_RADIX_THREADS_DEFAULT = 512
_RADIX_THREADS_SWEEP_N_SAMPLES = [50_000, 100_000, 300_000, 1_000_000]
_RADIX_THREADS_SALT = 1


def radix_select_threads(n: int) -> int:
    """Per-host radix-select threads/block from the kernel_tuning_cache.

    env-checked cache -> once-per-process sweep -> fallback. Returns the historical
    ``_RADIX_THREADS_DEFAULT`` (512) when no cache entry exists / the lookup fails / no cupy."""
    if _RADIX_THREADS_SPEC is None:
        return _RADIX_THREADS_DEFAULT
    try:
        choice = _RADIX_THREADS_SPEC.choose(n_samples=int(n))
    except Exception:
        return _RADIX_THREADS_DEFAULT
    if isinstance(choice, str) and choice.startswith("th_"):
        try:
            return int(choice[len("th_"):])
        except ValueError:
            return _RADIX_THREADS_DEFAULT
    return _RADIX_THREADS_DEFAULT


def _radix_edges_with_threads(cand, nbins: int, threads: int):
    """Run the radix-select interior-edge extraction with an EXPLICIT threads/block (the per-variant probe
    the sweep times). Output is BIT-IDENTICAL across thread counts (order stats are sum-reductions over the
    same column values), so the sweep ranks by WALL only. Returns the (nbins-1, K) f64 edges (host) so the
    equivalence check is a cheap numeric compare."""
    import cupy as cp

    from . import _gpu_resident_select as _sel

    saved = _sel._RADIX_THREADS_OVERRIDE
    try:
        _sel._RADIX_THREADS_OVERRIDE = int(threads)
        edges = _sel._radix_select_interior_edges(cand, int(nbins))
    finally:
        _sel._RADIX_THREADS_OVERRIDE = saved
    if edges is None:
        return np.empty(0, dtype=np.float64)
    return cp.asnumpy(edges)


def _make_radix_inputs(dims: dict):
    """A resident (n, K) f32 candidate matrix shaped like the F2 production chunk (heavy-tailed a**2/b)."""
    import cupy as cp

    n = int(dims["n_samples"])
    # K kept modest + built directly in f32 so the 1M-sample probe does not OOM the host (a 384-col f64
    # temp at 1M = 2.9GB). The kernel is launched one-block-per-column, so the per-block timing the sweep
    # ranks is K-independent (block size affects the n-loop, the same for every column); a smaller K just
    # bounds the probe's memory while preserving the threads/block ranking.
    K = 96
    rng = np.random.default_rng(0)
    a = rng.uniform(0.1, 1.1, (n, K)).astype(np.float32)
    b = rng.uniform(0.1, 1.1, (n, K)).astype(np.float32)
    cand = (a * a) / b   # heavy-tailed a**2/b, f32 throughout (no f64 temp)
    return (cp.asarray(cand),)


def _run_radix_threads_sweep() -> list:
    """Time each threads/block on the resident radix-edge path; fastest EQUIVALENT per n-region wins.
    All thread counts produce the SAME edges (sum-reduction invariance), so equivalence is trivially met."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        f"th_{t}": (lambda cand, _t=t: _radix_edges_with_threads(cand, 20, _t))
        for t in _RADIX_THREADS_VARIANTS
    }
    return sweep_backend_grid(
        variants,
        {"n_samples": _RADIX_THREADS_SWEEP_N_SAMPLES},
        _make_radix_inputs,
        reference=f"th_{_RADIX_THREADS_DEFAULT}",
        repeats=3, equiv_rtol=1e-6, equiv_atol=1e-6,
    )


def _radix_threads_fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the historical hardcoded 512 threads/block."""
    return f"th_{_RADIX_THREADS_DEFAULT}"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _RADIX_THREADS_SPEC = kernel_tuner(
        kernel_name="gpu_fe_radix_select_threads",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_radix_threads_sweep,
        axes={"n_samples": list(_RADIX_THREADS_SWEEP_N_SAMPLES)},
        fallback=_radix_threads_fallback_choice,
        gpu_capable=True,
        salt=_RADIX_THREADS_SALT,
        cli_label="gpu_fe_radix_select_threads",
    )
except Exception:
    _RADIX_THREADS_SPEC = None
