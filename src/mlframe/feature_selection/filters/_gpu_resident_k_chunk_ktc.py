"""G3 (2026-06-22): kernel-tuning-cache integration for the GPU FE K-chunk VRAM fraction.

Carved out of ``_gpu_resident_fe.py`` (LOC budget). Holds the per-host VRAM-fraction RESOLVER, the
per-fraction timing PROBE, the discrete-fraction SWEEP, and the ``kernel_tuner`` registration. The parent
``_gpu_resident_fe._gpu_k_chunk_vram_fraction`` lazy-imports :func:`gpu_k_chunk_vram_fraction` from here so
the parent stays under the <1k-LOC ceiling and the import graph stays acyclic.

WHY a tunable fraction: the K-chunk width that bounds the resident candidate-MI working set was governed by
a hardcoded ``0.25 * free_VRAM``. A WIDER fraction = fewer/larger chunks = fewer kernel launches + reductions
(the nsys-measured launch/sync overhead that dominates the per-pair x per-chunk loop), bounded by the card
not thrashing. Per ``feedback_use_kernel_tuning_cache_for_gpu`` the live fraction is looked up per-host from
the kernel_tuning_cache so a high-VRAM card learns a wider fraction instead of leaving it on the table --
never exceeding what that host measured as safe. Chunk width is per-column-INDEPENDENT, so the candidate MI
is selection-equivalent regardless of the chunk boundary (the sweep ranks fractions by WALL only).

GPU-only: on a CPU-only / no-cupy host the sweep never runs and ``.choose()`` returns the fallback, so
``_gpu_k_chunk`` keeps the conservative 0.25 default.
"""
from __future__ import annotations

import numpy as np

from ._gpu_resident_fe import (
    _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT,
    _GPU_K_CHUNK_VRAM_FRACTIONS,
)

_GPU_K_CHUNK_SWEEP_N_SAMPLES = [50_000, 100_000, 300_000, 1_000_000]
_GPU_K_CHUNK_SALT = 1


def gpu_k_chunk_vram_fraction(n: int) -> float:
    """Per-host VRAM-budget fraction for ``_gpu_k_chunk`` from the kernel_tuning_cache.

    env override -> code-version-checked cache -> once-per-process sweep -> fallback. Returns the
    conservative ``_GPU_K_CHUNK_VRAM_FRACTION_DEFAULT`` when no cache entry exists / the lookup fails."""
    if _GPU_K_CHUNK_SPEC is None:
        return _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT
    try:
        choice = _GPU_K_CHUNK_SPEC.choose(n_samples=int(n))
    except Exception:
        return _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT
    if isinstance(choice, str) and choice.startswith("frac_"):
        try:
            return float(choice[len("frac_") :])
        except ValueError:
            return _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT
    return _GPU_K_CHUNK_VRAM_FRACTION_DEFAULT


def gpu_resident_pair_candidate_mi_vram_fraction(
    a: np.ndarray, b: np.ndarray, y_codes: np.ndarray, *, nbins: int = 20, vram_fraction: float,
) -> tuple[list[str], np.ndarray]:
    """Variant of ``gpu_resident_pair_candidate_mi`` whose K-chunk width uses an EXPLICIT VRAM fraction --
    the per-fraction probe the sweep times to learn the fastest safe width. Output is selection-equivalent
    to the default-fraction path (chunk width is per-column-independent), so the sweep ranks by WALL only."""
    import cupy as cp

    from . import hermite_fe as _hf
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident
    from ._gpu_resident_fe import (
        _COMBOS, _candidate_names, _fused_generate_block, _gpu_k_chunk, _unary_stack_cm,
    )

    a_gpu = cp.asarray(a, dtype=cp.float64)
    b_gpu = cp.asarray(b, dtype=cp.float64)
    n = int(a_gpu.shape[0])
    y_i64 = np.ascontiguousarray(y_codes, dtype=np.int64)
    ua_cm = _unary_stack_cm(cp, a_gpu)
    ub_cm = _unary_stack_cm(cp, b_gpu)
    y_gpu = cp.asarray(y_i64)
    _ymm = cp.asnumpy(cp.stack((cp.min(y_gpu), cp.max(y_gpu))))
    _ymin = int(_ymm[0]); _ncls = int(_ymm[1]) - _ymin + 1
    k_chunk = _gpu_k_chunk(n, vram_fraction=vram_fraction)
    mi_parts: list[np.ndarray] = []
    for start in range(0, len(_COMBOS), k_chunk):
        block = _COMBOS[start : start + k_chunk]
        cand = _fused_generate_block(ua_cm, ub_cm, block)
        mi_parts.append(
            np.asarray(_plugin_mi_classif_batch_cuda_resident(cand, y_gpu, nbins, y_min=_ymin, n_classes=_ncls, relax_binning=True), dtype=np.float64)
        )
        del cand
    return _candidate_names(), np.concatenate(mi_parts) if mi_parts else np.empty(0)


def _make_gpu_k_chunk_inputs(dims: dict):
    n = int(dims["n_samples"])
    rng = np.random.default_rng(0)
    a = rng.normal(size=n).astype(np.float64)
    b = rng.normal(size=n).astype(np.float64)
    y = rng.integers(0, 4, size=n).astype(np.int64)
    return (a, b, y)


def _run_gpu_k_chunk_sweep() -> list:
    """Time each VRAM fraction on the resident candidate-MI path; fastest EQUIVALENT per n-region wins.
    All fractions produce the SAME candidate MI (chunk-invariant), so equivalence is trivially met."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        f"frac_{f}": (lambda a, b, y, _f=f: gpu_resident_pair_candidate_mi_vram_fraction(a, b, y, vram_fraction=_f)[1]) for f in _GPU_K_CHUNK_VRAM_FRACTIONS
    }
    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        variants,
        {"n_samples": _GPU_K_CHUNK_SWEEP_N_SAMPLES},
        _make_gpu_k_chunk_inputs,
        reference=f"frac_{_GPU_K_CHUNK_VRAM_FRACTION_DEFAULT}",
        repeats=3, equiv_rtol=1e-6, equiv_atol=1e-6,
    )


def _gpu_k_chunk_fallback_choice(n_samples: int) -> str:
    """Pre-sweep fallback: the conservative source-code default fraction."""
    return f"frac_{_GPU_K_CHUNK_VRAM_FRACTION_DEFAULT}"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _GPU_K_CHUNK_SPEC = kernel_tuner(
        kernel_name="gpu_fe_k_chunk_vram_fraction",
        variant_fns=(),  # GPU-only cupy path; covered by salt
        tuner=_run_gpu_k_chunk_sweep,
        axes={"n_samples": list(_GPU_K_CHUNK_SWEEP_N_SAMPLES)},
        fallback=_gpu_k_chunk_fallback_choice,
        gpu_capable=True,
        salt=_GPU_K_CHUNK_SALT,
        cli_label="gpu_fe_k_chunk_vram_fraction",
    )
except Exception:
    _GPU_K_CHUNK_SPEC = None
