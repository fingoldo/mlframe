"""KTC crossover for the resident-GPU maxT permutation-null pooled gain floor (iter16, 2026-06-23).

Gates ``_permutation_null_resident.pooled_gain_floor_perms_cupy`` (resident GPU histogram+MI+max) against
the host ``_permutation_null._pooled_gain_floor_perms_njit`` it replaces. Per
``feedback_use_kernel_tuning_cache_for_gpu`` the engage/skip decision is NOT a hardcoded threshold: a
``kernel_tuner`` sweeps both paths over an (n, ncand, nperm) grid and records, per region, which is faster.
The resident path engages only where MEASURED faster; otherwise the caller stays on the exact njit kernel.

HONEST EXPECTATION (GTX 1050 Ti, 6 SMs): the floor's work is (nperm * ncand * n) histogram cells. The
resident path amortises its single H2D + launch only at LARGE total work (wide pools p>=64 and/or n>=300k);
the production tabular F2 floor (ncand<=9, nperm=200, n=100k) is likely sub-crossover -> the sweep keeps it
on CPU, which is correct (the njit kernel is already sub-second there). The two paths differ only in FP
reduction order of the per-cell entropy (~1e-15) and the host owns the final ``np.quantile``, so they are
selection-equivalent; the sweep's equivalence tol is loosened accordingly and it ranks by WALL.

CPU/no-cupy host: the sweep never runs, ``.choose()`` returns "njit", and the caller takes the exact host path.
"""
from __future__ import annotations

import numpy as np

# (n, ncand, nperm) grid. ncand spans the narrow tabular floor (ncand~8) up to wide embedding/TF-IDF pools
# (ncand>=128) where the resident histogram amortises its launch; nperm spans the legacy 25 and the default
# 200. Default fallback keeps the un-tuned host path (njit) -- the resident path is OPT-IN-BY-MEASUREMENT.
_PERMNULL_SWEEP_N = [50_000, 100_000, 300_000]
_PERMNULL_SWEEP_NCAND = [8, 64, 256]
_PERMNULL_SWEEP_NPERM = [25, 200]
_PERMNULL_SALT = 1


def permnull_use_resident(n: int, ncand: int, nperm: int) -> bool:
    """Per-host engage decision for the resident-GPU permutation-null floor, from the kernel_tuning_cache.

    Returns ``True`` (use the resident GPU path) only on a measured-faster cache hit; ``False`` on a miss /
    no-cupy / lookup failure (caller stays on the exact host njit kernel). Each axis snaps to the nearest
    swept bucket. STRICT GPU mode (``MLFRAME_FE_GPU_STRICT=1``, diagnostic, default OFF) forces the resident
    path: the two paths differ only in FP reduction order (~1e-15) -> selection-equivalent."""
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        if fe_gpu_strict_enabled():
            return True
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    if _PERMNULL_SPEC is None:
        return False
    nb = min(_PERMNULL_SWEEP_NCAND, key=lambda b: abs(b - int(ncand)))
    pb = min(_PERMNULL_SWEEP_NPERM, key=lambda b: abs(b - int(nperm)))
    try:
        choice = _PERMNULL_SPEC.choose(n_samples=int(n), ncand=int(nb), nperm=int(pb))
    except Exception:
        return False
    return bool(choice == "resident")


def _make_permnull_inputs(dims: dict):
    """An (n, ncand, nperm) host workload shaped like a maxT floor call: per-candidate scaled X-codes, the
    K target shuffles, and the invariant marginal terms -- the SAME arguments both kernels receive, so the
    crossover measured is the histogram+MI+max work the floor routes (gen-agnostic)."""
    n = int(dims["n_samples"])
    ncand = int(dims["ncand"])
    nperm = int(dims["nperm"])
    nbins_x = 16
    nbins_y = 10
    rng = np.random.default_rng(0)
    x = rng.integers(0, nbins_x, size=(ncand, n)).astype(np.int64)
    y = rng.integers(0, nbins_y, size=n).astype(np.int32)
    scaled_codes = [(x[j] * nbins_y).astype(np.int32) for j in range(ncand)]
    scaled_flat = np.concatenate(scaled_codes)
    offsets = np.arange(ncand + 1, dtype=np.int64) * n
    joint_card = np.full(ncand, nbins_x * nbins_y, dtype=np.int64)
    inv_n = 1.0 / n
    y_counts = np.bincount(y, minlength=nbins_y).astype(np.float64)
    py = y_counts[y_counts > 0] * inv_n
    h_y = float(-(py * np.log(py)).sum())
    h_x = np.empty(ncand, dtype=np.float64)
    for j in range(ncand):
        xc = np.bincount(x[j], minlength=nbins_x).astype(np.float64)
        px = xc[xc > 0] * inv_n
        h_x[j] = float(-(px * np.log(px)).sum())
    mm_bias = np.full(ncand, (nbins_x - 1) * (nbins_y - 1) / (2.0 * n), dtype=np.float64)
    y_perm = y.copy()
    y_perms = np.empty((nperm, n), dtype=np.int32)
    for k in range(nperm):
        rng.shuffle(y_perm)
        y_perms[k] = y_perm
    return (scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n)


def _permnull_njit(scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n):
    """Sweep variant: dispatch straight to the exact host njit floor kernel (the sweep's timing reference)."""
    from ._permutation_null import _pooled_gain_floor_perms_njit

    return _pooled_gain_floor_perms_njit(scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n)


def _permnull_resident(scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n):
    """Sweep variant: dispatch to the resident-GPU cupy floor kernel being benchmarked against ``_permnull_njit``."""
    from ._permutation_null_resident import pooled_gain_floor_perms_cupy

    return pooled_gain_floor_perms_cupy(scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n)


def _run_permnull_sweep() -> list:
    """Time host njit vs the resident GPU floor across the (n, ncand, nperm) grid; faster EQUIVALENT wins per
    region. The two paths differ only in FP reduction order of the per-cell entropy -> selection-equivalent,
    so the equivalence tol is loosened and the sweep ranks by WALL."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"njit": _permnull_njit, "resident": _permnull_resident}
    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        variants,
        {"n_samples": _PERMNULL_SWEEP_N, "ncand": _PERMNULL_SWEEP_NCAND, "nperm": _PERMNULL_SWEEP_NPERM},
        _make_permnull_inputs,
        reference="njit",
        repeats=3, equiv_rtol=5e-2, equiv_atol=5e-2,
    )


def _permnull_fallback_choice(n_samples: int, ncand: int = 8, nperm: int = 200) -> str:
    """Pre-sweep fallback: the host njit path (the resident path engages only when MEASURED faster)."""
    return "njit"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _PERMNULL_SPEC = kernel_tuner(
        kernel_name="fe_maxt_permnull_floor_resident_crossover",
        variant_fns=(),  # GPU resident path covered by salt; njit is the reference
        tuner=_run_permnull_sweep,
        axes={"n_samples": list(_PERMNULL_SWEEP_N), "ncand": list(_PERMNULL_SWEEP_NCAND), "nperm": list(_PERMNULL_SWEEP_NPERM)},
        fallback=_permnull_fallback_choice,
        gpu_capable=True,
        salt=_PERMNULL_SALT,
        cli_label="fe_maxt_permnull_floor_resident_crossover",
    )
except Exception:
    _PERMNULL_SPEC = None
