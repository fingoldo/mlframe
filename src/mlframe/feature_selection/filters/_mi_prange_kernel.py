"""Inner-loop permutation-test njit kernel (serial + prange-parallel) + its per-host dispatch.

Split out from ``permutation.py`` to keep that file below the 1k-line monolith threshold (CLAUDE.md:
"Monolith split via re-export"). Behaviour preserved bit-for-bit; every moved symbol is re-exported from
``permutation`` so existing ``from .permutation import parallel_mi_prange`` (and the other moved names)
imports continue to work.
"""
from __future__ import annotations

import logging
from typing import Optional, cast

import numpy as np
from numba import njit, prange
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

from .info_theory import compute_relevance_score

logger = logging.getLogger(__name__)


@njit(parallel=True, nogil=True, cache=True)
def parallel_mi_prange(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    base_seed: np.uint64,
    dtype: type = np.int32,
    use_su: bool = False,  # SU normalization toggle threaded from mi_direct.
) -> tuple:
    """Inner-loop parallel permutation test.

    Runs ``npermutations`` shuffles in a numba ``prange``. Each iteration owns a private ``classes_y`` copy and a private LCG seeded with
    ``base_seed * 2654435761 + i`` (Knuth's multiplicative hash). The seeding scheme is **independent of n_workers**, so the ``(nfailed, nchecked)`` output is
    bit-exact across ``n_workers in {1, 2, 4, 8}`` for the same ``base_seed`` (verified by ``test_phase1_reproducibility``).

    Differences from ``parallel_mi`` (joblib-process worker):
    * No early termination on ``nfailed >= max_failed`` - every permutation in the budget runs because ``prange`` iterations are independent. For short budgets
      (npermutations < 30) the early-exit win was negligible anyway.
    * No global ``np.random.shuffle``; manual Fisher-Yates with a per-iteration LCG so the parallel race that legacy code hit under multi-thread numba is gone
      by construction.
    """
    if npermutations == 0:
        return 0, 0

    n = len(classes_y)
    nfailed_arr = np.zeros(npermutations, dtype=np.int64)

    for i in prange(npermutations):
        # Per-iteration LCG state. Knuth multiplicative hash + fold of i gives a deterministic, n_workers-independent stream.
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(i + 1)

        local = classes_y.copy()
        # Fisher-Yates shuffle with the per-iter LCG.
        for j in range(n - 1, 0, -1):
            # PCG-like step.
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]
            local[j] = local[k]
            local[k] = tmp

        mi_perm = compute_relevance_score(
            use_su, classes_x, freqs_x, local, freqs_y, dtype=dtype,
        )
        if mi_perm >= original_mi:
            nfailed_arr[i] = 1

    return int(nfailed_arr.sum()), npermutations


@njit(parallel=False, nogil=True, cache=True)
def _parallel_mi_prange_serial(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    original_mi: float,
    base_seed: np.uint64,
    dtype: type = np.int32,
    use_su: bool = False,
) -> tuple:
    """Serial twin of :func:`parallel_mi_prange` -- identical body, ``range`` instead of ``prange``.

    ``parallel=True`` pays a fixed per-call thread-pool dispatch cost that a small ``n * npermutations``
    budget (a Fisher-Yates shuffle + one MI evaluation per permutation) cannot amortize. Unlike the
    trivial elementwise kernels this codebase's other njit(parallel=True) fixes targeted, EACH iteration
    here does real O(n) work, so the crossover depends on both ``n`` and ``npermutations`` together, not
    either alone -- see ``_mi_prange_dispatch``'s threshold for the measured n*npermutations crossover.
    Each iteration owns a private ``classes_y`` copy and a per-iteration LCG seeded purely from
    ``(base_seed, i)`` (independent of iteration order/thread), so output is bit-identical to the
    parallel kernel regardless of which one runs."""
    if npermutations == 0:
        return 0, 0

    n = len(classes_y)
    nfailed_arr = np.zeros(npermutations, dtype=np.int64)

    for i in range(npermutations):
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(i + 1)

        local = classes_y.copy()
        for j in range(n - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            k = int(state >> np.uint64(33)) % (j + 1)
            tmp = local[j]
            local[j] = local[k]
            local[k] = tmp

        mi_perm = compute_relevance_score(
            use_su, classes_x, freqs_x, local, freqs_y, dtype=dtype,
        )
        if mi_perm >= original_mi:
            nfailed_arr[i] = 1

    return int(nfailed_arr.sum()), npermutations


# Per-host serial/parallel crossover via the canonical kernel_tuning_cache (NO hardcoded threshold --
# feedback_use_kernel_tuning_cache_for_gpu / feedback_fastest_default_with_dispatch). Dev-box measurement
# (same-process A/B, warm, best-of-30) found n*npermutations<=15000 always a parallel LOSS (1.4x-8.6x
# slower) and n*npermutations>=60000 always a parallel WIN (1.35-4x faster) -- the fallback threshold
# below (used only pre-sweep / on tuner failure) is deliberately close to the loss side for that reason,
# but the REAL per-host decision comes from the tuner's measured sweep.
_MI_PRANGE_SWEEP_N = [500, 15_000, 200_000]
_MI_PRANGE_SWEEP_NPERM = [10, 100]
_MI_PRANGE_SALT = 1
_MI_PRANGE_NBINS = 10


def _make_mi_prange_inputs(dims: dict) -> tuple:
    """(classes_x, freqs_x, classes_y, freqs_y, npermutations, original_mi, base_seed, dtype, use_su)
    at the sweep's (n, npermutations) cell."""
    n, npermutations = int(dims["n"]), int(dims["npermutations"])
    rng = np.random.default_rng(0)
    classes_x = rng.integers(0, _MI_PRANGE_NBINS, n).astype(np.int32)
    freqs_x = np.bincount(classes_x, minlength=_MI_PRANGE_NBINS).astype(np.int32)
    classes_y = rng.integers(0, 2, n).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=2).astype(np.int32)
    return (classes_x, freqs_x, classes_y, freqs_y, npermutations, 0.01, np.uint64(42), np.int32, False)


def _run_mi_prange_sweep() -> list:
    """Serial-vs-parallel wall-clock sweep over the (n, npermutations) grid -> kernel_tuning_cache regions."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        "serial": lambda *a: _parallel_mi_prange_serial(*a),
        "parallel": lambda *a: parallel_mi_prange(*a),
    }
    return cast(list, sweep_backend_grid(
        variants,
        {"n": _MI_PRANGE_SWEEP_N, "npermutations": _MI_PRANGE_SWEEP_NPERM},
        _make_mi_prange_inputs,
        reference="serial", repeats=5, equiv_atol=0.0, equiv_rtol=0.0,
    ))


def _mi_prange_fallback_choice(n: int, npermutations: int) -> str:
    """Pre-sweep / tuner-failure fallback: parallel above the dev-box-measured n*npermutations
    crossover (see the module comment above for the confirmed-loss/confirmed-win bracket)."""
    return "parallel" if int(n) * int(npermutations) >= 30_000 else "serial"


_MI_PRANGE_PARALLELISM_SPEC = kernel_tuner(
    kernel_name="mi_prange_kernel_parallelism",
    variant_fns=(_parallel_mi_prange_serial, parallel_mi_prange),
    tuner=_run_mi_prange_sweep,
    axes={"n": _MI_PRANGE_SWEEP_N, "npermutations": _MI_PRANGE_SWEEP_NPERM},
    fallback=_mi_prange_fallback_choice,
    gpu_capable=False,
    salt=_MI_PRANGE_SALT,
    cli_label="mi_prange_kernel_parallelism",
)


def _mi_prange_dispatch(
    classes_x: np.ndarray,
    freqs_x: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: Optional[np.ndarray],
    npermutations: int,
    original_mi: float,
    base_seed: np.uint64,
    dtype: type = np.int32,
    use_su: bool = False,
) -> tuple[int, int]:
    """Dispatch to the serial or parallel ``parallel_mi_prange`` variant via the per-host
    kernel_tuning_cache (``_MI_PRANGE_PARALLELISM_SPEC``). Bit-identical output either way (see
    ``_parallel_mi_prange_serial``'s docstring)."""
    n = len(classes_y)
    try:
        choice = _MI_PRANGE_PARALLELISM_SPEC.choose(n=n, npermutations=int(npermutations))
    except Exception as e:
        logger.debug("mi_prange_kernel_parallelism choose() failed, using the size-based fallback: %s", e)
        choice = _mi_prange_fallback_choice(n, int(npermutations))
    fn = parallel_mi_prange if choice == "parallel" else _parallel_mi_prange_serial
    nfailed, n_checked = fn(
        classes_x=classes_x, freqs_x=freqs_x, classes_y=classes_y, freqs_y=freqs_y,
        npermutations=npermutations, original_mi=original_mi, base_seed=base_seed,
        dtype=dtype, use_su=use_su,
    )
    return int(nfailed), int(n_checked)
