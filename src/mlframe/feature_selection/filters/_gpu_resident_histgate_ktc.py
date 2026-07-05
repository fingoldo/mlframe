"""Lever C (2026-06-23): HW-aware + kernel-tuning-cache integration for the MI noise-gate batched histogram
threads/block (the ``_kernel_bs`` shared-privatized joint-histogram in ``_batch_mi_noise_gate_kernels``).

Carved as a sibling of ``batch_mi_noise_gate_gpu.py`` (that module is at ~892 LOC -- adding the sweep there
would push it over the 1k ceiling; mirrors the radix ``_gpu_resident_radix_ktc`` carve). The batched hist
kernel ``_kernel_bs`` (and the global-atomic ``_kernel_b``) launched with a HARDCODED ``threads_per_block=128``
over a grid of ``(K, P)`` blocks -- each block strides over ALL n rows, so the row-loop parallelises across
threads exactly like the radix-select n-loop. 128 threads left the GTX 1050 Ti badly under-occupied (max
2048 threads/SM -> 128 is 1/16th of an SM per block); nsys (isolated KTC, F2 100k, one fit) showed the
MI-gate hist ``_kernel_bs`` = the #2 kernel (~2375ms, 32 launches). More threads/block = more rows binned in
parallel per block.

Per ``feedback_use_kernel_tuning_cache_for_gpu`` + the HW-spec request: the candidate threads/block SET is
DERIVED from device occupancy (warp-multiples that fill the SM, via ``_gpu_hw_launch.occupancy_block_candidates``)
so a different card sweeps its own HW-valid points, and the KTC empirically picks the measured-best among them
(keyed on n_rows). The histogram COUNTS are integer + commutative -> identical for any thread count, so MI and
the gate decision are BIT-IDENTICAL for every block size; the sweep ranks by WALL only. 128 stays the
fallback (pre-sweep / no-cuda / lookup failure) so the CPU / no-CUDA path is byte-for-byte unchanged.
"""
from __future__ import annotations

import numpy as np

_HISTGATE_THREADS_DEFAULT = 128  # historical hardcoded default + pre-sweep / no-cuda / lookup-failure fallback
_HISTGATE_THREADS_SEED = [128, 256, 512, 768, 1024]
_HISTGATE_THREADS_SWEEP_N_ROWS = [50_000, 100_000, 300_000]
_HISTGATE_THREADS_SALT = 2  # bump on probe-shape change (K=96->512) to invalidate the stale-shape cache


def _histgate_threads_variants() -> list:
    """HW-occupancy-bounded threads/block candidates for the batched-hist kernel. Intersects the seed list
    with the warp-multiple block sizes hitting >=2 active blocks/SM on THIS device (the kernel's dynamic
    shared is the small per-column nb_k*K_y*4 histogram, so it is register/thread-limited, not shared-limited
    -- pass a tiny dyn_smem). Falls back to the seed list when no HW info (no cupy)."""
    seed = list(_HISTGATE_THREADS_SEED)
    try:
        from ._gpu_hw_launch import device_props, occupancy_block_candidates

        if not device_props():
            return seed
        valid = set(occupancy_block_candidates(
            regs_per_thread=32,          # numba.cuda hist kernel is light; 32 regs/thread is a safe estimate
            static_smem=0, dyn_smem=4096,  # small per-column histogram (nb_k*K_y*4); register/thread-bound
            min_active_blocks=2,
        ))
        if not valid:
            return seed
        cands = sorted(v for v in seed if v in valid)
        if _HISTGATE_THREADS_DEFAULT not in cands:
            cands.append(_HISTGATE_THREADS_DEFAULT)  # always keep the reference variant
        top = max(valid)
        if top not in cands:
            cands.append(top)
        return sorted(set(cands)) or seed
    except Exception:
        return seed


_HISTGATE_THREADS_VARIANTS = _histgate_threads_variants()


def histgate_threads(n_rows: int) -> int:
    """Per-host batched-hist threads/block from the kernel_tuning_cache (keyed on n_rows). env-checked cache
    -> once-per-process sweep -> fallback. Returns the historical 128 when no cache entry / lookup fails / no
    cuda. Block size NEVER changes the integer counts -> MI + gate decision bit-identical."""
    if _HISTGATE_THREADS_SPEC is None:
        return _HISTGATE_THREADS_DEFAULT
    try:
        choice = _HISTGATE_THREADS_SPEC.choose(n_rows=int(n_rows))
    except Exception:
        return _HISTGATE_THREADS_DEFAULT
    if isinstance(choice, str) and choice.startswith("th_"):
        try:
            return int(choice[len("th_") :])
        except ValueError:
            return _HISTGATE_THREADS_DEFAULT
    return _HISTGATE_THREADS_DEFAULT


def _hist_counts_with_threads(disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
                              npermutations, base_seed, min_nonzero_confidence, use_su, dtype, threads):
    """Run the resident noise-gate with an EXPLICIT threads/block (the per-variant probe the sweep times).
    Output (the per-column gated MI vector) is BIT-IDENTICAL across thread counts (integer commutative
    counts), so the sweep ranks by WALL only."""
    from .batch_mi_noise_gate_gpu import batch_mi_with_noise_gate_cuda_resident

    return batch_mi_with_noise_gate_cuda_resident(
        disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
        classes_y_safe=classes_y_safe, freqs_y=freqs_y, npermutations=int(npermutations),
        base_seed=np.uint64(base_seed), min_nonzero_confidence=float(min_nonzero_confidence),
        use_su=False, threads_per_block=int(threads),
    )


def _make_histgate_inputs(dims: dict):
    """Synthetic resident-gate inputs at ``dims['n_rows']`` rows. K is PRODUCTION-SCALE (a candidate chunk is
    ~500-600 columns): the threads/block ranking is K-SENSITIVE here because the per-launch H2D wall (which a
    tiny K lets dominate and MASK the kernel win) shrinks relative to the (K, P)-block kernel time as K grows.
    A K=512 probe makes the kernel the wall (as in production) so the sweep ranks the kernel, not the H2D.

    CORRECTION (2026-06-23, contention-robust re-sweep + CUDA-event back-to-back A/B at THIS K=512 probe shape):
    the earlier "1024 is 2.7x faster than 128 at K=583" claim was a measurement ARTIFACT (different probe shape /
    contention-distorted A/B). The definitive interleaved-min CUDA-event A/B on the GTX 1050 Ti at K=512 gives,
    per single launch: n=50k 128=58ms vs 1024=88ms; n=100k 128=110ms vs 1024=174ms (1.57x SLOWER at 1024);
    n=300k 128=800ms vs 1024=535ms (1024 wins only here). So 128 IS the true-fastest at <=100k and the
    contention-robust sweep correctly keeps th_128 there + th_1024 at >=300k -- there is NO ~1.28s MI-gate win to
    materialise; forcing 1024 at 100k would REGRESS the kernel ~1.5x. The KTC pick now matches the measured floor."""
    rng = np.random.default_rng(0)
    n = int(dims["n_rows"])
    K = 512
    nbins = 20
    n_classes_y = 4
    disc_2d = rng.integers(0, nbins, size=(n, K)).astype(np.int32)
    factors_nbins = np.full(K, nbins, dtype=np.int64)
    classes_y = rng.integers(0, n_classes_y, size=n).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=n_classes_y).astype(np.float64) / max(1, n)
    return (disc_2d, factors_nbins, classes_y, classes_y.copy(), freqs_y, 10, np.uint64(0), 0.95, False, np.int32)


def _run_histgate_threads_sweep() -> list:
    """Time each threads/block on the resident noise-gate path; fastest EQUIVALENT per n-region wins. All
    thread counts produce the SAME gated-MI vector (integer commutative counts), so equivalence is met."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {f"th_{t}": (lambda *a, _t=t: _hist_counts_with_threads(*a, _t)) for t in _HISTGATE_THREADS_VARIANTS}
    return sweep_backend_grid(
        variants,
        {"n_rows": _HISTGATE_THREADS_SWEEP_N_ROWS},
        _make_histgate_inputs,
        reference=f"th_{_HISTGATE_THREADS_DEFAULT}",
        repeats=3, equiv_rtol=1e-9, equiv_atol=1e-12,
    )


def _histgate_threads_fallback_choice(n_rows: int) -> str:
    """Pre-sweep fallback: the historical hardcoded 128 threads/block."""
    return f"th_{_HISTGATE_THREADS_DEFAULT}"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _HISTGATE_THREADS_SPEC = kernel_tuner(
        kernel_name="gpu_fe_histgate_threads",
        variant_fns=(),  # GPU-only numba.cuda path; covered by salt
        tuner=_run_histgate_threads_sweep,
        axes={"n_rows": list(_HISTGATE_THREADS_SWEEP_N_ROWS)},
        fallback=_histgate_threads_fallback_choice,
        gpu_capable=True,
        salt=_HISTGATE_THREADS_SALT,
        cli_label="gpu_fe_histgate_threads",
    )
except Exception:
    _HISTGATE_THREADS_SPEC = None
