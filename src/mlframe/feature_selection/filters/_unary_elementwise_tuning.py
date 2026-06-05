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
from functools import lru_cache

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
_UNARY_SALT = 2  # residency-aware sweep + 2-variant code_version


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
    """cupy elementwise unary (``cos``). Pays H2D if given a host array; a no-op
    when the input is already VRAM-resident. Returns a device array (the small
    output transfer is the caller's concern, negligible for this bandwidth map)."""
    import cupy as cp

    return cp.cos(cp.asarray(vals))


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


def _unary_code_version():
    """code_version over BOTH variant bodies (both always-defined) -> a numpy OR
    cupy kernel edit auto-invalidates the cached tuning."""
    from pyutilz.dev.code_versioning import compute_code_version

    return compute_code_version(_unary_numpy, _unary_cupy, salt=_UNARY_SALT)


def _unary_fallback_choice(n_samples: int, location: str) -> str:
    """Pre-sweep heuristic: VRAM-resident -> cupy (no transfer to pay);
    DRAM-resident -> cupy only above the old 500k-cell breakeven."""
    if not _HAS_CUPY:
        return "numpy"
    if location == "device":
        return "cupy"
    return "cupy" if n_samples >= _UNARY_DEFAULT_MIN_CELLS else "numpy"


@lru_cache(maxsize=256)
def unary_elementwise_backend_choice(n_samples: int, location: str = "host") -> str:
    """Per-host, residency-aware backend ("numpy"/"cupy") for an elementwise unary
    on ``n_samples`` cells whose input lives in ``location`` ("host"/"device").

    env -> per-host cache (code-version checked, keyed by n_samples + location) ->
    measurement-backed fallback, via the shared get_or_tune orchestrator. Memoized
    (the dispatch is hot). The caller MUST still gate a "cupy" result on live CUDA
    + per-op cupy-compatibility before routing to GPU."""
    n_samples = int(n_samples)
    try:
        from pyutilz.system.kernel_tuning_cache import KernelTuningCache

        result = KernelTuningCache().get_or_tune(
            "unary_elementwise",
            dims={"n_samples": n_samples, "location": location},
            tuner=_run_unary_sweep,
            axes=["n_samples", "location"],
            fallback={"backend_choice": _unary_fallback_choice(n_samples, location)},
            code_version=_unary_code_version(),
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        if bc in ("numpy", "cupy"):
            return bc
    except Exception as e:
        logger.debug("unary_elementwise get_or_tune failed: %s", e)
    return _unary_fallback_choice(n_samples, location)


# Register with the kernel-tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune unary_elementwise (GPU-capable; residency-aware).
from pyutilz.system.kernel_tuner import kernel_tuner

kernel_tuner(
    kernel_name="unary_elementwise",
    variant_fns=(_unary_numpy, _unary_cupy),  # both always-defined -> auto-invalidate
    tuner=_run_unary_sweep,
    axes={"n_samples": list(_UNARY_SWEEP_N), "location": ["host", "device"]},
    fallback={"backend_choice": "numpy"},
    gpu_capable=True,
    salt=_UNARY_SALT,
    cli_label="unary_elementwise",
)

# Public alias for the sweep (registry + external retune callers).
run_unary_sweep = _run_unary_sweep
__all__ = ["unary_elementwise_backend_choice", "run_unary_sweep"]
