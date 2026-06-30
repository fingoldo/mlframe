"""KTC crossover for the resident-GPU ORDER-2 pooled-max joint-MI permutation-null floor (2026-06-30).

Gates ``_permutation_null_pair_resident.pooled_pair_permutation_null_joint_mi_floor_cupy`` (resident GPU
batched joint-histogram + MI + per-shuffle max over the candidate-pair pool) against the host
``_permutation_null.pooled_pair_permutation_null_joint_mi_floor`` (the CPU ``batch_pair_mi_prange`` njit loop)
it replaces. Per ``feedback_use_kernel_tuning_cache_for_gpu`` the engage/skip decision is NOT a hardcoded
threshold: a ``kernel_tuner`` sweeps both paths over an (n, n_pairs) grid and records, per region, which is
faster. The resident path engages ONLY where MEASURED faster; otherwise the caller stays on the exact CPU njit
floor (the default + fallback).

MEASURED (GTX 1050 Ti, 6 SMs, 4GB; 2026-06-30, K=25): the device path is SLOWER than the CPU njit
``batch_pair_mi_prange`` floor at every measured shape -- n=50k/36-pair 0.64x, n=100k/66-pair 0.63x,
n=200k/120-pair 0.66x, and 0.26-0.34x at the wide pools (n>=200k, pairs>=435) where the
``(perm x pair x n)`` flat-index + bincount working set tiles into many launches and under-occupies the small
card while the njit prange kernel saturates the CPU. So on THIS card the sweep keeps every region on CPU, which
is correct -- the CPU floor is already sub-3s at the production shapes. A big-VRAM / many-SM card (more SMs to
occupy, fewer bincount-tile launches) is where the crossover may flip; the sweep learns that per host.

The two paths are SELECTION-EQUIVALENT (same pooled-MAX construction, same plug-in joint MI of the same integer
contingency table to FP round-off, host-owned ``np.quantile``); the device RNG draws a different but reproducible
shuffle stream (a RANDOM null), so the floor matches the host floor closely enough that the gate decision
``pair_mi >= floor`` is unchanged (verified in ``test_pair_maxt_perm_null_resident.py``). The sweep's equivalence
tol is loosened accordingly and it ranks by WALL.

CPU/no-cupy host: the sweep never runs, ``.choose()`` returns "njit", and the caller takes the exact host path.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

# (n, n_pairs) grid. n_pairs spans the narrow production floor (just above fe_pair_maxt_min_pairs=30) up to wide
# embedding/screening pools where the resident batched histogram might amortise its launch on a big card. Default
# fallback keeps the un-tuned host path (njit) -- the resident path is OPT-IN-BY-MEASUREMENT.
_PAIR_PERMNULL_SWEEP_N = [50_000, 100_000, 200_000, 500_000]
_PAIR_PERMNULL_SWEEP_NPAIRS = [36, 120, 435]
_PAIR_PERMNULL_SALT = 1


def pair_permnull_use_resident(n: int, n_pairs: int) -> bool:
    """Per-host engage decision for the resident-GPU order-2 maxT permutation-null floor, from the KTC cache.

    Returns ``True`` (use the resident GPU path) only on a measured-faster cache hit; ``False`` on a miss /
    no-cupy / lookup failure (caller stays on the exact host njit floor). Each axis snaps to the nearest swept
    bucket. STRICT GPU mode (``MLFRAME_FE_GPU_STRICT=1``, diagnostic, default OFF) forces the resident path: the
    two paths are selection-equivalent (the device floor preserves the gate decision)."""
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled

        if fe_gpu_strict_enabled():
            return True
    except Exception:
        pass
    if _PAIR_PERMNULL_SPEC is None:
        return False
    pb = min(_PAIR_PERMNULL_SWEEP_NPAIRS, key=lambda b: abs(b - int(n_pairs)))
    try:
        choice = _PAIR_PERMNULL_SPEC.choose(n_samples=int(n), n_pairs=int(pb))
    except Exception:
        return False
    return choice == "resident"


def _make_pair_permnull_inputs(dims: dict):
    """An (n, n_pairs) host workload shaped like an order-2 maxT floor call: a candidate-pair pool over a discrete
    X matrix and a discrete target -- the SAME arguments both kernels receive, so the crossover measured is the
    per-shuffle batched joint-histogram + MI + pool-max work the floor routes."""
    n = int(dims["n_samples"])
    n_pairs_target = int(dims["n_pairs"])
    nbins_x = 8
    n_classes_y = 4
    # Smallest n_features whose C(nf, 2) covers the target pair count.
    nf = 2
    while nf * (nf - 1) // 2 < n_pairs_target:
        nf += 1
    rng = np.random.default_rng(0)
    fd = rng.integers(0, nbins_x, size=(n, nf)).astype(np.int32)
    nb = np.full(nf, nbins_x, dtype=np.int32)
    y = rng.integers(0, n_classes_y, size=n).astype(np.int64)
    fy = np.bincount(y, minlength=n_classes_y).astype(np.float64) / n
    pairs = list(combinations(range(nf), 2))[:n_pairs_target]
    pa = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
    pb = np.fromiter((p[1] for p in pairs), dtype=np.int64, count=len(pairs))
    return (fd, nb, pa, pb, y, fy)


def _pair_permnull_njit(fd, nb, pa, pb, y, fy):
    from ._permutation_null import pooled_pair_permutation_null_joint_mi_floor

    return pooled_pair_permutation_null_joint_mi_floor(
        factors_data=fd, nbins=nb, pair_a=pa, pair_b=pb, classes_y=y, freqs_y=fy,
        n_permutations=25, quantile=0.95, random_seed=0,
    )


def _pair_permnull_resident(fd, nb, pa, pb, y, fy):
    from ._permutation_null_pair_resident import pooled_pair_permutation_null_joint_mi_floor_cupy

    return pooled_pair_permutation_null_joint_mi_floor_cupy(
        factors_data=fd, pair_a=pa, pair_b=pb, nbins=nb, classes_y=y, freqs_y=fy,
        n_permutations=25, quantile=0.95, random_seed=0,
    )


def _run_pair_permnull_sweep() -> list:
    """Time host njit vs the resident GPU floor across the (n, n_pairs) grid; faster EQUIVALENT wins per region.
    Both are 0.95-quantiles of a pool-max permutation null; the device RNG draws a different stream, so the
    equivalence tol is loosened (the floor values agree to a few-percent band that does not flip the gate) and
    the sweep ranks by WALL."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"njit": _pair_permnull_njit, "resident": _pair_permnull_resident}
    return sweep_backend_grid(
        variants,
        {"n_samples": _PAIR_PERMNULL_SWEEP_N, "n_pairs": _PAIR_PERMNULL_SWEEP_NPAIRS},
        _make_pair_permnull_inputs,
        reference="njit",
        repeats=3, equiv_rtol=3e-1, equiv_atol=1e-3,
    )


def _pair_permnull_fallback_choice(n_samples: int, n_pairs: int = 36) -> str:
    """Pre-sweep fallback: the host njit path (the resident path engages only when MEASURED faster)."""
    return "njit"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _PAIR_PERMNULL_SPEC = kernel_tuner(
        kernel_name="fe_pair_maxt_permnull_floor_resident_crossover",
        variant_fns=(),  # GPU resident path covered by salt; njit is the reference
        tuner=_run_pair_permnull_sweep,
        axes={"n_samples": list(_PAIR_PERMNULL_SWEEP_N), "n_pairs": list(_PAIR_PERMNULL_SWEEP_NPAIRS)},
        fallback=_pair_permnull_fallback_choice,
        gpu_capable=True,
        salt=_PAIR_PERMNULL_SALT,
        cli_label="fe_pair_maxt_permnull_floor_resident_crossover",
    )
except Exception:
    _PAIR_PERMNULL_SPEC = None


__all__ = ["pair_permnull_use_resident"]
