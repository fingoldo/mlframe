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

# Candidate threads/block to sweep. HW-AWARE (2026-06-23): instead of fixed magic constants the candidate
# SET is now derived from the device occupancy (warp-multiple block sizes that achieve >=2 active blocks/SM
# for THIS kernel's register + shared usage, see _gpu_hw_launch.occupancy_block_candidates) so a different
# card sweeps its own HW-valid points. The seed list below (powers + 768, spanning the GTX 1050 Ti measured
# curve) is intersected with / falls back to the derived set; 512 = the historical hardcoded default + the
# pre-sweep / no-cupy / lookup-failure fallback. Block size NEVER changes the order statistics (sum-reduction
# over the same column values) -> edges/codes/MI/selection bit-identical for every thread count.
_RADIX_THREADS_SEED = [256, 512, 768, 1024]
_RADIX_THREADS_DEFAULT = 512  # historical hardcoded default + pre-sweep / no-cupy / lookup-failure fallback
# Worst-case dynamic shared the radix histogram uses (R<=40 active -> the host gates R*256*4 > device limit
# off; R=38 is the canonical F2 R -> 38*256*4 ~= 38KB). Used only to bound the OCCUPANCY candidate set.
_RADIX_DYN_SMEM_PROBE = 38 * 256 * 4


def _radix_threads_variants() -> list:
    """HW-occupancy-bounded threads/block candidates to sweep. Intersects the seed list with the warp-multiple
    block sizes that hit >=2 active blocks/SM for the radix kernel's register/shared footprint on THIS device
    (so the sweep ranges over HW-valid, occupancy-sane options that port to other cards). Falls back to the
    raw seed list when no HW info / occupancy info is available (cupy missing) so behavior is unchanged there.
    Always includes the default 512 so the reference variant exists."""
    seed = list(_RADIX_THREADS_SEED)
    try:
        from ._gpu_hw_launch import device_props, occupancy_block_candidates
        from ._gpu_resident_select import _get_radix_select_kernel

        if not device_props():
            return seed
        ker = _get_radix_select_kernel(True)
        valid = set(occupancy_block_candidates(
            regs_per_thread=int(getattr(ker, "num_regs", 32)) or 32,
            static_smem=int(getattr(ker, "shared_size_bytes", 0)) or 0,
            dyn_smem=_RADIX_DYN_SMEM_PROBE, min_active_blocks=2,
        ))
        if not valid:
            return seed
        cands = sorted(v for v in seed if v in valid)
        if _RADIX_THREADS_DEFAULT not in cands and _RADIX_THREADS_DEFAULT in valid:
            cands.append(_RADIX_THREADS_DEFAULT)
        # Add the device's occupancy-max block (largest warp-multiple still >=2 blocks/SM) if not present --
        # the seed list may not contain this card's sweet spot. Keeps the sweep small but HW-spanning.
        top = max(valid)
        if top not in cands:
            cands.append(top)
        return sorted(set(cands)) or seed
    except Exception:
        return seed


_RADIX_THREADS_VARIANTS = _radix_threads_variants()
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


# Lever C (2026-06-23): which f32 radix-select WINDOW-MATCH variant -- the base linear scan or the
# binary-search variant -- is the per-host fastest. The base kernel is warp-divergence bound (~42% eff) on
# the per-row linear window scan; the bsearch variant replaces it with a branchless binary search over the
# sorted active-window prefixes (bit-identical order statistics; the sweep ranks by WALL only). "bsearch" is
# the measured-faster default / lookup-failure fallback (proven by the CUDA-event A/B at production shape).
_RADIX_F32_VARIANTS = ("linear", "bsearch")
_RADIX_F32_VARIANT_DEFAULT = "bsearch"
_RADIX_F32_VARIANT_SALT = 1


def radix_select_f32_variant(n: int) -> str:
    """Per-host f32 radix-select window-match variant ("linear"/"bsearch") from the kernel_tuning_cache.
    Returns ``_RADIX_F32_VARIANT_DEFAULT`` ("bsearch") when no cache entry exists / the lookup fails / no
    cupy. Both variants are bit-identical in the produced order statistics (only the window search differs)."""
    if _RADIX_F32_VARIANT_SPEC is None:
        return _RADIX_F32_VARIANT_DEFAULT
    try:
        choice = _RADIX_F32_VARIANT_SPEC.choose(n_samples=int(n))
    except Exception:
        return _RADIX_F32_VARIANT_DEFAULT
    if isinstance(choice, str) and choice in _RADIX_F32_VARIANTS:
        return choice
    return _RADIX_F32_VARIANT_DEFAULT


def _radix_edges_with_f32_variant(cand, nbins: int, variant: str):
    """Run the radix-select interior-edge extraction with an EXPLICIT f32 window-match variant (the
    per-variant probe the sweep times). Output is BIT-IDENTICAL across variants (the binary search only
    changes HOW a row finds its window, not the order statistics), so the sweep ranks by WALL only."""
    import cupy as cp

    from . import _gpu_resident_select as _sel

    saved = _sel._RADIX_F32_VARIANT_OVERRIDE
    try:
        _sel._RADIX_F32_VARIANT_OVERRIDE = str(variant)
        edges = _sel._radix_select_interior_edges(cand, int(nbins))
    finally:
        _sel._RADIX_F32_VARIANT_OVERRIDE = saved
    if edges is None:
        return np.empty(0, dtype=np.float64)
    return cp.asnumpy(edges)


def _run_radix_f32_variant_sweep() -> list:
    """Time each f32 window-match variant on the resident radix-edge path; fastest EQUIVALENT per n-region
    wins. Both variants produce the SAME edges (order-statistic invariance), so equivalence is trivially met."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        v: (lambda cand, _v=v: _radix_edges_with_f32_variant(cand, 20, _v))
        for v in _RADIX_F32_VARIANTS
    }
    return sweep_backend_grid(
        variants,
        {"n_samples": _RADIX_THREADS_SWEEP_N_SAMPLES},
        _make_radix_inputs,
        reference=_RADIX_F32_VARIANT_DEFAULT,
        repeats=3, equiv_rtol=1e-6, equiv_atol=1e-6,
    )


def _radix_f32_variant_fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the measured-faster binary-search variant."""
    return _RADIX_F32_VARIANT_DEFAULT


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


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _RADIX_F32_VARIANT_SPEC = kernel_tuner(
        kernel_name="gpu_fe_radix_select_f32_variant",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_radix_f32_variant_sweep,
        axes={"n_samples": list(_RADIX_THREADS_SWEEP_N_SAMPLES)},
        fallback=_radix_f32_variant_fallback_choice,
        gpu_capable=True,
        salt=_RADIX_F32_VARIANT_SALT,
        cli_label="gpu_fe_radix_select_f32_variant",
    )
except Exception:
    _RADIX_F32_VARIANT_SPEC = None
