"""Per-host, residency-aware numpy-vs-cupy backend selection for the elementwise
unary-transform path in ``check_prospective_fe_pairs`` (see
``_feature_engineering_pairs``).

Carved into a sibling module (the pairs file is already >1k LOC) and consumed via
``unary_elementwise_backend_choice``. A full ``sweep_backend_grid`` sweep over
``n_samples`` AND data residency (DRAM vs VRAM) is wired through
``KernelTuningCache.get_or_tune``; the old fixed 500_000-cell breakeven is the
measurement-backed fallback.

Why residency is a real axis here (measured, not assumed -- GTX 1050 Ti,
2026-06-05): the H2D transfer is what made cupy lose on DRAM-resident input, so
the optimal backend FLIPS with where the data lives --
  * DRAM-resident: numpy wins up to ~1M cells, cupy only at 10M (transfer amortised);
  * VRAM-resident: cupy wins at EVERY size (no transfer to pay).
So the sweep measures both residencies and the dispatch picks by the live input's
memory. The benchmarked op is a single-arg elementwise ufunc (``cos``) on a 1-D
float array -- representative of the production set (``sin``/``exp``/``sqrt``/...):
all are bandwidth-bound maps with the same breakeven.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.feature_engineering")

try:
    import cupy as _cp  # type: ignore

    _HAS_CUPY = True
except Exception:
    _cp = None  # type: ignore
    _HAS_CUPY = False

# Source-code default GPU breakeven (cells) for DRAM-resident input, used until a
# per-host kernel_tuning_cache entry exists.
_UNARY_DEFAULT_MIN_CELLS = 500_000
_UNARY_SWEEP_N = [50_000, 200_000, 800_000, 3_200_000, 12_800_000]
_UNARY_SALT = 3  # residency-aware sweep + cupy variant now includes output D2H


def _is_device(x) -> bool:
    """True if ``x`` is a GPU-resident array (cupy / CUDA-array-interface)."""
    return hasattr(x, "__cuda_array_interface__")


def _unary_numpy(vals):
    """Reference CPU elementwise unary (``cos``). Pays D2H if given device input."""
    if _is_device(vals):
        import cupy as cp

        vals = cp.asnumpy(vals)
    return np.cos(vals)


def _unary_cupy(vals):
    """cupy elementwise unary (``cos``). Pays H2D if given a host array (no-op if
    already VRAM-resident) AND the output D2H back to host. The production call
    site (_feature_engineering_pairs) consumes a HOST array, so the swept cost
    MUST include the round trip -- returning a device array here would time cupy
    too favourably and bias the crossover toward GPU. Returns host numpy."""
    import cupy as cp

    return cp.asnumpy(cp.cos(cp.asarray(vals)))


def _make_unary_inputs(dims: dict):
    """A 1-D float32 host array of length ``n_samples`` -- the operand a unary ufunc takes."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(dims["n_samples"])).astype(np.float32),)


def _run_unary_sweep() -> list:
    """Full n_samples grid x residency sweep (numpy vs cupy, DRAM- and
    VRAM-resident) -> ``location_eq`` backend_choice regions. cupy is included
    only when importable; float32 elementwise maps agree to a loosened tol."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"numpy": _unary_numpy}
    if _HAS_CUPY:
        variants["cupy"] = _unary_cupy
    return sweep_backend_grid(
        variants,
        {"n_samples": _UNARY_SWEEP_N},
        _make_unary_inputs,
        reference="numpy",
        residencies=("host", "device") if _HAS_CUPY else ("host",),
        repeats=5,
        equiv_rtol=1e-4,
        equiv_atol=1e-5,
    )


def _unary_fallback_choice(n_samples: int, location: str) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): VRAM-resident ->
    cupy (no transfer to pay); DRAM-resident -> cupy only above the 500k breakeven."""
    if not _HAS_CUPY:
        return "numpy"
    if location == "device":
        return "cupy"
    return "cupy" if n_samples >= _UNARY_DEFAULT_MIN_CELLS else "numpy"


def unary_elementwise_backend_choice(n_samples: int, location: str = "host") -> str:
    """Per-host, residency-aware backend ("numpy"/"cupy") for an elementwise unary
    on ``n_samples`` cells whose input lives in ``location`` ("host"/"device"), via
    the spec's one-call choose() (env -> code-version-checked cache -> on-miss sweep
    -> the _unary_fallback_choice heuristic; memoized per dims). The caller MUST
    still gate a "cupy" result on live CUDA + per-op compatibility before GPU."""
    return _UNARY_SPEC.choose(n_samples=int(n_samples), location=location)


# Register with the kernel-tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune unary_elementwise (GPU-capable; residency-aware).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_UNARY_SPEC = kernel_tuner(
    kernel_name="unary_elementwise",
    variant_fns=(_unary_numpy, _unary_cupy),  # both always-defined -> auto-invalidate
    tuner=_run_unary_sweep,
    axes={"n_samples": list(_UNARY_SWEEP_N), "location": ["host", "device"]},
    fallback=_unary_fallback_choice,  # callable (n_samples, location) -> str
    gpu_capable=True,
    salt=_UNARY_SALT,
    cli_label="unary_elementwise",
)

# Public alias for the sweep (registry + external retune callers).
run_unary_sweep = _run_unary_sweep
__all__ = ["unary_elementwise_backend_choice", "run_unary_sweep"]
