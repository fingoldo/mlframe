"""Kernel-tuning-cache integration for the cat-FE permutation kernel.

The per-host backend-crossover sweep (cpu_serial / cpu_parallel / cupy) + the
``@kernel_tuner`` registration of ``cat_fe_perm_kernel``. Split out of
``_cat_confirm_permutation`` so the parent stays under the 1k-LOC ceiling. The
permutation kernels themselves stay in the parent; this module imports them and
exposes ``_CAT_PERM_SPEC``, which the parent re-imports at its module bottom.
"""

from __future__ import annotations

import numpy as np

from ._cat_confirm_permutation import (
    _count_nfailed_joint_indep_cupy,
    _count_nfailed_joint_indep_prange,
    _count_nfailed_joint_indep_serial,
)

# Module-level cupy availability probe -- gates the crossover sweep + @kernel_tuner
# registration (the per-call dispatch still re-checks importability inside the parent's
# ``_perm_kernel_dispatch_use_gpu``).
try:
    import cupy as _cp_probe  # noqa: F401
    _CUPY_AVAIL = True
except Exception:
    _CUPY_AVAIL = False


# Crossover threshold for CPU vs GPU on the permutation kernel.
# Calibrated empirically via ``profiling/bench_perm_kernel_gpu.py``
# (GTX-class consumer GPU, 6-core numba CPU):
#   100k x 3       CPU 4 ms  vs GPU 36 ms    -> CPU wins
#   100k x 50      CPU 49 ms vs GPU 241 ms   -> CPU wins
#   1M x 3         CPU 56 ms vs GPU 42 ms    -> GPU wins 1.3x
#   1M x 50        CPU 612 ms vs GPU 419 ms  -> GPU wins 1.5x
#   1M x 100       CPU 1135 ms vs GPU 834 ms -> GPU wins 1.4x
#   5M x 10        CPU 1018 ms vs GPU 373 ms -> GPU wins 2.7x
# Crossover sits at ~N=1_000_000 regardless of n_perms. Below 1M the
# per-call GPU launch + transfer cost (~30-40 ms) dominates the
# tiny CPU compute; above 1M, GPU bincount parallelism amortises the
# transfer and consistently wins.
_GPU_PERM_KERNEL_THRESHOLD_N: int = 1_000_000

# The threshold above remains the source-code fallback, applied verbatim when the
# per-host cache has no entry yet (mirrors ``signal/dtw.py``). 2-D axes, BOTH swept
# (full grid, not a fixed-n_perms 1-D crossover): ``n_samples`` (dominant) x ``n_perms``.
_PERM_SWEEP_N_PERMS_GRID = [10, 50, 200]  # full n_perms axis (was a single fixed 50)
_PERM_SWEEP_N_SAMPLES = [100_000, 300_000, 1_000_000, 3_000_000, 5_000_000]
_PERM_SWEEP_K_PAIR = 10
_PERM_SWEEP_K_X = 5
_PERM_SWEEP_K_Y = 3
_PERM_SALT = 2  # serial-njit variant added + full 2-D (n_samples x n_perms) grid


def _make_perm_kernel_inputs(dims: dict):
    """Synthetic args for the perm-kernel grid sweep at ``dims['n_samples']`` rows.
    ``n_perms`` (a kernel argument) is carried as the last positional so it varies
    per cell; the variant wrappers append only ``base_seed`` / ``dtype``."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    n_perms = int(dims["n_perms"])
    classes_pair = rng.integers(0, _PERM_SWEEP_K_PAIR, size=n).astype(np.int64)
    freqs_pair = np.bincount(classes_pair, minlength=_PERM_SWEEP_K_PAIR).astype(np.float64) / max(1, n)
    classes_x1 = rng.integers(0, _PERM_SWEEP_K_X, size=n).astype(np.int64)
    freqs_x1 = np.bincount(classes_x1, minlength=_PERM_SWEEP_K_X).astype(np.float64) / max(1, n)
    classes_x2 = rng.integers(0, _PERM_SWEEP_K_X, size=n).astype(np.int64)
    freqs_x2 = np.bincount(classes_x2, minlength=_PERM_SWEEP_K_X).astype(np.float64) / max(1, n)
    classes_y = rng.integers(0, _PERM_SWEEP_K_Y, size=n).astype(np.int64)
    freqs_y = np.bincount(classes_y, minlength=_PERM_SWEEP_K_Y).astype(np.float64) / max(1, n)
    ii_obs = 0.0  # count all perms -> identical work on all backends, equiv-comparable
    return (classes_pair, freqs_pair, classes_x1, freqs_x1, classes_x2, freqs_x2,
            classes_y, freqs_y, ii_obs, n_perms)


def _run_perm_kernel_sweep() -> list:
    """Full (n_samples x n_perms) grid sweep -> backend_choice regions: cpu_serial /
    cpu_parallel / cupy, fastest EQUIVALENT per cell. Both axes are swept (n_perms is
    a kernel arg carried in the inputs). Inputs are host-resident, so no residency
    axis. cupy included only when available. With ii_obs=0 every perm counts on all
    backends, so the integer ``n_failed`` outputs match exactly."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    _seed = 7

    def _cpu_serial(*a):
        return _count_nfailed_joint_indep_serial(*a, _seed, np.int32)

    def _cpu_parallel(*a):
        return _count_nfailed_joint_indep_prange(*a, _seed, np.int32)

    variants = {"cpu_serial": _cpu_serial, "cpu_parallel": _cpu_parallel}
    if _CUPY_AVAIL:
        def _gpu(*a):
            return _count_nfailed_joint_indep_cupy(*a, base_seed=_seed)
        variants["cupy"] = _gpu

    return sweep_backend_grid(
        variants,
        {"n_samples": _PERM_SWEEP_N_SAMPLES, "n_perms": _PERM_SWEEP_N_PERMS_GRID},
        _make_perm_kernel_inputs,
        reference="cpu_serial",
        repeats=5, equiv_rtol=1e-3, equiv_atol=1e-3,
    )


def _perm_kernel_fallback_choice(n_samples: int, n_perms: int) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): cupy above the old
    ``_GPU_PERM_KERNEL_THRESHOLD_N``; on CPU, parallel above a modest row count, else
    serial."""
    if _CUPY_AVAIL and n_samples >= _GPU_PERM_KERNEL_THRESHOLD_N:
        return "cupy"
    return "cpu_parallel" if n_samples >= 300_000 else "cpu_serial"


# Register with the @kernel_tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune the cat-FE permutation kernel. GPU-capable (cupy backend).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner  # noqa: E402

_CAT_PERM_SPEC = kernel_tuner(
    kernel_name="cat_fe_perm_kernel",
    variant_fns=(_count_nfailed_joint_indep_serial, _count_nfailed_joint_indep_prange),  # CPU bodies; cupy via salt
    tuner=_run_perm_kernel_sweep,
    axes={"n_samples": list(_PERM_SWEEP_N_SAMPLES), "n_perms": list(_PERM_SWEEP_N_PERMS_GRID)},
    fallback=_perm_kernel_fallback_choice,  # callable (n_samples, n_perms) -> str
    gpu_capable=True,
    salt=_PERM_SALT,
    cli_label="cat_fe_perm_kernel",
)
