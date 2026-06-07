"""Per-host numpy-vs-numba calibration sweep for the 2-D ``_per_member_mae_std`` path.

Mirrors the ``joint_hist_batched`` sweep pattern (measure the live machine, persist
the winning regions to the per-host pyutilz ``kernel_tuning_cache``), but for a
single CPU kernel with two variants instead of the FS GPU-RawKernel package.

It runs **automatically on the first cache miss** (see ``_per_member_use_numba``
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

from .base import _per_member_mae_std_njit, _HAS_NUMBA_PER_MEMBER, _PER_MEMBER_KERNEL_NAME

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
# Bump when _SWEEP_K or the _EQUIV_* tolerances change -- sweep semantics the
# source hash can't see -- so cached tunings re-validate on the next call.
# 2: 3-D njit std fixed to per-column form + ndim threaded into the cache key.
# 3: sweep now measures ndim=2 AND ndim=3 separately -> ndim_eq-tagged regions.
_PER_MEMBER_SALT = 3

_AUTOTUNE_ATTEMPTED = False  # process-scoped guard: sweep at most once per process


def per_member_code_version() -> str | None:
    """code_version for the per_member dispatch: hashes the two variant bodies
    (_numpy_2d + the njit kernel) + _PER_MEMBER_SALT. A kernel edit (or a salt
    bump) invalidates stale cached tunings deterministically. None if pyutilz
    code-versioning is unavailable (cache then falls back to no-version-check)."""
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version
        return compute_code_version(_numpy_2d, _per_member_mae_std_njit, salt=_PER_MEMBER_SALT)
    except Exception:
        return None


def _numpy_2d(arr, med):
    diffs = np.abs(arr - med)
    return diffs.mean(axis=1), np.sqrt(np.var(diffs, axis=1))


def _numpy_3d(arr, med):
    """3-D (K, N, C) numpy reference matching the njit 3-D branch: per-column MAE
    & std (std anchored at each column's own mean), then averaged across C."""
    diffs = np.abs(arr - med[np.newaxis, :, :])
    mae_per_col = diffs.mean(axis=1)
    std_per_col = np.sqrt(diffs.var(axis=1))
    return mae_per_col.mean(axis=1), std_per_col.mean(axis=1)


_SWEEP_C = 3  # 3-D column count for the ndim=3 sweep (small -- hardest case for numba)


def _resolve_max_elements(observed_elements, max_elements) -> int:
    """Bound the sweep to relevant sizes: explicit ``max_elements`` if given,
    else ~2x the observed triggering size, always within [2k, ceiling]."""
    if max_elements is not None:
        return max(2_000, min(int(max_elements), _SWEEP_CEILING))
    if observed_elements:
        return max(20_000, min(int(observed_elements) * 2, _SWEEP_CEILING))
    return 300_000  # no usage info -> modest default


def _measure_per_member_crossover(ndim: int, grid: list, repeats: int, rng) -> tuple:
    """Measure numpy-vs-numba per-member MAE/std across the element grid for one
    ``ndim`` (2 or 3); return ``(crossover_elements_or_None, worst_abs_diff)``. A
    variant that diverges from the numpy reference is never trusted (forces numpy).
    n_groups is held at _SWEEP_K (the parallelism axis; the crossover is dominated
    by elements_per_member)."""
    ref = _numpy_2d if ndim == 2 else _numpy_3d
    crossover = None
    worst_diff = 0.0
    for e in grid:
        if ndim == 2:
            arr = rng.standard_normal((_SWEEP_K, e))
            med = rng.standard_normal(e)
        else:
            n = max(1, e // _SWEEP_C)
            arr = rng.standard_normal((_SWEEP_K, n, _SWEEP_C))
            med = rng.standard_normal((n, _SWEEP_C))
        out_np = ref(arr, med)
        out_nb = _per_member_mae_std_njit(arr, med)
        diff = float(max(np.abs(out_np[0] - out_nb[0]).max(), np.abs(out_np[1] - out_nb[1]).max()))
        worst_diff = max(worst_diff, diff)
        equivalent = (np.allclose(out_np[0], out_nb[0], rtol=_EQUIV_RTOL, atol=_EQUIV_ATOL)
                      and np.allclose(out_np[1], out_nb[1], rtol=_EQUIV_RTOL, atol=_EQUIV_ATOL))
        # Steady-state prewarm of BOTH variants on this exact array before timing.
        ref(arr, med)
        _per_member_mae_std_njit(arr, med)
        t_np = timeit.timeit(lambda: ref(arr, med), number=repeats) / repeats
        t_nb = timeit.timeit(lambda: _per_member_mae_std_njit(arr, med), number=repeats) / repeats
        if not equivalent:
            logger.warning("per_member sweep ndim=%d e=%d K=%d: numba DIVERGES from numpy "
                           "(maxdiff=%.2e > tol) -> forcing numpy regardless of speed", ndim, e, _SWEEP_K, diff)
            winner = "numpy"
        else:
            winner = "numba" if t_nb < t_np else "numpy"
            logger.info("per_member sweep ndim=%d e=%d K=%d: numpy=%.3fms numba=%.3fms maxdiff=%.2e -> %s",
                        ndim, e, _SWEEP_K, t_np * 1e3, t_nb * 1e3, diff, winner)
        if winner == "numba" and crossover is None:
            crossover = e
    return crossover, worst_diff


def run_per_member_sweep(observed_elements: int | None = None, max_elements: int | None = None,
                         repeats: int = 25) -> list[dict]:
    """Benchmark numpy vs numba across a bounded element grid on THIS host, for BOTH
    ndim=2 and ndim=3 (separate crossovers -- the 3-D per-column reduction has a
    different cost profile than the 2-D one). Returns ``ndim_eq``-tagged regions for
    ``kernel_tuning_cache`` (numpy below the measured crossover, numba at/above it).
    n_groups is fixed at _SWEEP_K (carried in the dispatch key but not swept; the
    crossover is dominated by elements_per_member)."""
    if not _HAS_NUMBA_PER_MEMBER:
        logger.info("per_member sweep: numba unavailable -> numpy everywhere")
        return [{"elements_per_member_max": None, "backend_choice": "numpy"}]
    cap = _resolve_max_elements(observed_elements, max_elements)
    grid = sorted({e for e in _SWEEP_ELEMENTS if e <= cap}
                  | ({int(observed_elements)} if observed_elements and observed_elements <= _SWEEP_CEILING else set()))
    if not grid:
        grid = [_SWEEP_ELEMENTS[0]]
    rng = np.random.default_rng(0)
    # Prewarm the numba JIT / disk-cache for BOTH the 2-D and 3-D float64 signatures.
    _per_member_mae_std_njit(rng.standard_normal((2, 16)), rng.standard_normal(16))
    _per_member_mae_std_njit(rng.standard_normal((2, 8, _SWEEP_C)), rng.standard_normal((8, _SWEEP_C)))
    regions: list[dict] = []
    for ndim in (2, 3):
        crossover, worst_diff = _measure_per_member_crossover(ndim, grid, repeats, rng)
        if crossover is None:  # numba never won (or always diverged) for this ndim
            regions.append({"ndim_eq": ndim, "elements_per_member_max": None,
                            "backend_choice": "numpy", "max_abs_diff": worst_diff})
        elif crossover <= grid[0]:  # numba won from the smallest swept size
            regions.append({"ndim_eq": ndim, "elements_per_member_max": None,
                            "backend_choice": "numba", "max_abs_diff": worst_diff})
        else:
            regions.append({"ndim_eq": ndim, "elements_per_member_max": crossover - 1,
                            "backend_choice": "numpy", "max_abs_diff": worst_diff})
            regions.append({"ndim_eq": ndim, "elements_per_member_max": None,
                            "backend_choice": "numba", "max_abs_diff": worst_diff})
    return regions


def ensure_per_member_tuning(observed_elements: int | None = None, observed_groups: int | None = None,
                             force: bool = False, max_elements: int | None = None, repeats: int = 25) -> None:
    """Populate the per-host cache for the 2-D backend if absent (or ``force``).

    ``observed_elements`` (the size that triggered the miss) bounds the sweep grid
    so calibration stays on realistic sizes. Best-effort + idempotent: skips if
    pyutilz is missing, an entry already exists, or a sweep was already attempted
    this process (so repeated misses for different sizes don't re-benchmark)."""
    global _AUTOTUNE_ATTEMPTED
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache, cache_path
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
                     axes=["elements_per_member", "n_groups", "ndim"], regions=regions)
        logger.info("per_member auto-tune winners persisted to %s: %s", cache_path(), regions)
    except Exception as e:  # never let calibration break a training run
        logger.debug("per_member auto-tune failed (using fallback): %s", e)


# Register with the kernel-tuner registry so ``mlframe-tune-kernels`` /
# ``retune_all`` can discover + batch-tune this kernel on a quiet machine. CPU
# only (no GPU variant -- a CPU-resident axis-1 reduction; cupy was measured and
# lost under both residencies on this HW). The sweep measures ndim=2 AND ndim=3
# separately -> ``ndim_eq``-tagged regions, so the two get their own crossovers
# (the 3-D njit was fixed to per-column std, bit-identical to numpy). ``n_groups``
# is carried in the dispatch key but held at _SWEEP_K in the sweep (the parallelism
# axis; the crossover is dominated by elements_per_member).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

kernel_tuner(
    kernel_name=_PER_MEMBER_KERNEL_NAME,
    variant_fns=(_numpy_2d,),  # reference; the njit variant is covered by salt
    tuner=run_per_member_sweep,
    axes={"elements_per_member": list(_SWEEP_ELEMENTS), "n_groups": [_SWEEP_K], "ndim": [2, 3]},
    fallback={"backend_choice": "numpy"},
    salt=_PER_MEMBER_SALT,
    env_key="MLFRAME_PER_MEMBER_BACKEND",
    gpu_capable=False,
    cli_label="per_member_mae_std",
)


__all__ = ["run_per_member_sweep", "ensure_per_member_tuning", "per_member_code_version"]
