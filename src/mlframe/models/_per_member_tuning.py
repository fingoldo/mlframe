"""Per-host numpy-vs-numba calibration sweep for the 2-D ``_per_member_mae_std`` path.

Mirrors the ``joint_hist_batched`` sweep pattern (measure the live machine, persist
the winning regions to the per-host pyutilz ``kernel_tuning_cache``), but for a
single CPU kernel with two variants instead of the FS GPU-RawKernel package.

It runs **automatically on the first cache miss** (see ``_per_member_use_numba_2d``
in ``_ensembling_base``): the first time the kernel is dispatched on a machine with
no cached entry, the numpy and numba backends are benchmarked across an
``elements_per_member`` grid, each size's wall-time AND max-abs-diff-vs-reference
are logged, and the chosen regions are written to
``~/.pyutilz/kernel_tuning/<hw_fingerprint>.json``. Later calls read the cache.

Two design points enforced here:

* **Bounded sweep size.** The grid is capped (``max_elements``, default derived
  from the *observed* triggering size up to a hard ceiling) so a kernel that only
  ever runs on thousands of rows is never tuned on billions. Because numba's edge
  over the numpy broadcast grows monotonically with size, the ``numba`` region
  above the measured crossover is a catch-all (no upper cap) -- larger-than-swept
  sizes still route correctly without having been benchmarked.

* **Correctness gate on max abs diff.** Speed is only trusted when the faster
  variant AGREES with the reference (numpy). Each size's max abs diff is computed,
  logged, and recorded; if a variant diverges beyond tolerance it is NOT selected
  regardless of speed (this is the guard that keeps e.g. a differently-defined
  kernel from being silently picked just because it is faster).

cupy is intentionally absent: it was measured and lost at every size for this
CPU-resident axis-1 reduction (H2D transfer ~6x slower than numba).

Disable the on-first-call benchmark with ``MLFRAME_PER_MEMBER_AUTOTUNE=0`` (the
dispatcher then uses its measurement-backed fallback); force a re-tune with
``ensure_per_member_tuning(force=True)``.
"""
from __future__ import annotations

import logging
import timeit

import numpy as np

from ._ensembling_base import _per_member_mae_std_njit, _HAS_NUMBA_PER_MEMBER, _PER_MEMBER_KERNEL_NAME

logger = logging.getLogger("mlframe.models.ensembling")

# Base element grid spanning the sub-ms tie zone up through where numba wins
# decisively; filtered by the resolved max_elements. K kept small (hardest case
# for numba -- its edge only grows with K) so the crossover is conservative.
_SWEEP_ELEMENTS = (2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 300_000, 1_000_000)
_SWEEP_K = 4
_SWEEP_CEILING = 2_000_000  # never benchmark above this many elements (cost + relevance guard)
# Variants must agree this closely for the faster one to be trusted (two-pass
# reductions differ only by float reassociation ~1e-14; a real semantic gap is
# orders larger and must force the reference path).
_EQUIV_RTOL = 1e-6
_EQUIV_ATOL = 1e-9

_AUTOTUNE_ATTEMPTED = False  # process-scoped guard: sweep at most once per process


def _numpy_2d(arr, med):
    diffs = np.abs(arr - med)
    return diffs.mean(axis=1), np.sqrt(np.var(diffs, axis=1))


def _resolve_max_elements(observed_elements, max_elements) -> int:
    """Bound the sweep to relevant sizes: explicit ``max_elements`` if given,
    else ~2x the observed triggering size, always within [2k, ceiling]."""
    if max_elements is not None:
        return max(2_000, min(int(max_elements), _SWEEP_CEILING))
    if observed_elements:
        return max(20_000, min(int(observed_elements) * 2, _SWEEP_CEILING))
    return 300_000  # no usage info -> modest default


def run_per_member_sweep(observed_elements: int | None = None, max_elements: int | None = None,
                         repeats: int = 25) -> list[dict]:
    """Benchmark numpy vs numba across a bounded element grid on THIS host; return
    an ordered region list for ``kernel_tuning_cache`` (numpy below the measured
    crossover, numba at/above it). Logs per-size wall-time + max-abs-diff and
    refuses any variant that diverges from the numpy reference."""
    if not _HAS_NUMBA_PER_MEMBER:
        logger.info("per_member sweep: numba unavailable -> numpy everywhere")
        return [{"elements_per_member_max": None, "backend_choice": "numpy"}]
    cap = _resolve_max_elements(observed_elements, max_elements)
    grid = sorted({e for e in _SWEEP_ELEMENTS if e <= cap}
                  | ({int(observed_elements)} if observed_elements and observed_elements <= _SWEEP_CEILING else set()))
    if not grid:
        grid = [_SWEEP_ELEMENTS[0]]
    rng = np.random.default_rng(0)
    # Prewarm the numba per-signature JIT / disk-cache once for the (2-D, float64)
    # signature before any timing, so even the first swept size never pays compile.
    _per_member_mae_std_njit(rng.standard_normal((2, 16)), rng.standard_normal(16))
    crossover = None
    worst_diff = 0.0
    for n in grid:
        arr = rng.standard_normal((_SWEEP_K, n))
        med = rng.standard_normal(n)
        out_np = _numpy_2d(arr, med)
        out_nb = _per_member_mae_std_njit(arr, med)
        diff = float(max(np.abs(out_np[0] - out_nb[0]).max(), np.abs(out_np[1] - out_nb[1]).max()))
        worst_diff = max(worst_diff, diff)
        equivalent = (np.allclose(out_np[0], out_nb[0], rtol=_EQUIV_RTOL, atol=_EQUIV_ATOL)
                      and np.allclose(out_np[1], out_nb[1], rtol=_EQUIV_RTOL, atol=_EQUIV_ATOL))
        # Per-size prewarm: an extra steady-state pass of BOTH variants on this
        # exact array so the timed loop measures steady state (warm CPU caches,
        # warm numba code path), not the cold first iteration.
        _numpy_2d(arr, med)
        _per_member_mae_std_njit(arr, med)
        t_np = timeit.timeit(lambda: _numpy_2d(arr, med), number=repeats) / repeats
        t_nb = timeit.timeit(lambda: _per_member_mae_std_njit(arr, med), number=repeats) / repeats
        if not equivalent:
            # Faster-but-different -> never trust it; force the reference path.
            logger.warning("per_member sweep n=%d K=%d: numba DIVERGES from numpy "
                           "(maxdiff=%.2e > tol) -> forcing numpy regardless of speed", n, _SWEEP_K, diff)
            winner = "numpy"
        else:
            winner = "numba" if t_nb < t_np else "numpy"
            logger.info("per_member sweep n=%d K=%d: numpy=%.3fms numba=%.3fms maxdiff=%.2e -> %s",
                        n, _SWEEP_K, t_np * 1e3, t_nb * 1e3, diff, winner)
        if winner == "numba" and crossover is None:
            crossover = n
    if crossover is None:  # numba never won (or always diverged) on this HW
        return [{"elements_per_member_max": None, "backend_choice": "numpy", "max_abs_diff": worst_diff}]
    if crossover <= grid[0]:  # numba won from the smallest swept size
        return [{"elements_per_member_max": None, "backend_choice": "numba", "max_abs_diff": worst_diff}]
    return [
        {"elements_per_member_max": crossover - 1, "backend_choice": "numpy", "max_abs_diff": worst_diff},
        {"elements_per_member_max": None, "backend_choice": "numba", "max_abs_diff": worst_diff},
    ]


def ensure_per_member_tuning(observed_elements: int | None = None, observed_groups: int | None = None,
                             force: bool = False, max_elements: int | None = None, repeats: int = 25) -> None:
    """Populate the per-host cache for the 2-D backend if absent (or ``force``).

    ``observed_elements`` (the size that triggered the miss) bounds the sweep grid
    so calibration stays on realistic sizes. Best-effort + idempotent: skips if
    pyutilz is missing, an entry already exists, or a sweep was already attempted
    this process (so repeated misses for different sizes don't re-benchmark)."""
    global _AUTOTUNE_ATTEMPTED
    try:
        from pyutilz.system.kernel_tuning_cache import KernelTuningCache, cache_path
    except Exception as e:
        logger.debug("kernel_tuning_cache unavailable; skip per_member tuning: %s", e)
        return
    cache = KernelTuningCache()
    if not force:
        if _AUTOTUNE_ATTEMPTED or cache.has(_PER_MEMBER_KERNEL_NAME):
            return
    _AUTOTUNE_ATTEMPTED = True
    logger.debug("per_member: no kernel_tuning_cache entry for %r on this host; "
                 "auto-tuning numpy-vs-numba now (observed elements=%s)",
                 _PER_MEMBER_KERNEL_NAME, observed_elements)
    try:
        regions = run_per_member_sweep(observed_elements=observed_elements,
                                       max_elements=max_elements, repeats=repeats)
        cache.update(_PER_MEMBER_KERNEL_NAME,
                     axes=["elements_per_member", "n_groups"], regions=regions)
        logger.info("per_member auto-tune winners persisted to %s: %s", cache_path(), regions)
    except Exception as e:  # never let calibration break a training run
        logger.debug("per_member auto-tune failed (using fallback): %s", e)


__all__ = ["run_per_member_sweep", "ensure_per_member_tuning"]
