"""KTC crossover for the batched conditional-MI CPU<->CUDA dispatch (2026-06-23).

MANDATE-1 (repo rule ``feedback_use_kernel_tuning_cache_for_gpu``: never hardcode CUDA thresholds): the
``_cmi_cuda._should_use_cuda`` gate previously fell back, on every cache miss, to a HARDCODED magic crossover
``(n*p) >= 1_000_000 and p >= 64``. That constant is a dev-box guess: it is wrong on any other card (a
stronger GPU wins at a smaller (n,p); a weaker one needs a larger one). This module replaces the magic number
with a per-host MEASURED crossover -- a ``kernel_tuner`` spec that sweeps the REAL conditional-MI shapes
(CPU ``_cpu_cmi_loop`` vs CUDA ``conditional_mi_batched_cuda``) across an n x p grid and records, per
(n_samples, p) region, which backend is faster. ``_should_use_cuda`` consults the cache first; the hardcoded
heuristic survives ONLY as the un-tuned (pre-sweep / no-cupy / lookup-failure) default, exactly per the rule.

The two backends are NUMERICALLY EQUIVALENT (the CUDA kernel reproduces the CPU ``conditional_mi`` reduction
to ~1e-9; see ``_cmi_cuda`` header + ``test_cmi_cuda_kernel.py``), so the sweep's equivalence check passes and
it ranks the two purely by WALL. The winning variant name (``"cuda"`` / ``"cpu"``) per region IS the gate.

GPU-only: on a CPU-only / no-cupy host the sweep never runs and ``.choose()`` returns ``"cpu"``, so the
dispatch stays on the exact CPU loop -- byte-for-byte the legacy no-GPU behavior.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import numpy as np

# Production (n, p) grid the cmi gate must rank. n spans the MRMR fit sizes; p (candidate count per greedy
# round) spans the small-round (p~16) to wide-screen (p>=256) regimes where the crossover lives. The CUDA
# kernel amortises its launch only once p is sizable, so p is a real axis (not just n).
_CMI_SWEEP_N_SAMPLES = [50_000, 100_000, 300_000, 1_000_000]
_CMI_SWEEP_P = [16, 64, 128, 256]
_CMI_SALT = 1

# Un-tuned default crossover (the historical hardcoded heuristic): CUDA from ~n*p >= 1e6 with p >= 64. Used
# ONLY when the cache has no entry / cupy missing / lookup fails. KEPT as the safe bootstrap per the repo rule.
_CMI_DEFAULT_NP_PRODUCT = 1_000_000
_CMI_DEFAULT_MIN_P = 64


def cmi_use_cuda(n: int, p: int) -> bool | None:
    """Per-host CPU<->CUDA crossover decision for the batched conditional MI, from the kernel_tuning_cache.

    Returns ``True`` (use CUDA) / ``False`` (use CPU) on a cache hit, or ``None`` when no measured entry
    exists yet (pre-sweep / no-cupy / lookup failure) so the caller applies its hardcoded bootstrap fallback.
    The ``p`` axis is snapped to the nearest swept bucket so an arbitrary candidate count maps to a measured
    region. STRICT GPU mode (``MLFRAME_FE_GPU_STRICT=1``, diagnostic, default OFF) forces CUDA: the CPU/CUDA
    backends are numerically equivalent (~1e-9), so this is selection-equivalent."""
    try:
        from mlframe.feature_selection.filters._fe_gpu_strict import fe_gpu_strict_enabled
        # Pass THIS call's own (n, p) -- 2026-07-11 fix. Calling with no args made the per-call work floor a
        # no-op here (fe_gpu_strict_enabled's floor is a no-op whenever n/p are omitted), silently defeating
        # the caller's own p<64 decline at _cmi_cuda.py's _should_use_cuda: a small-p late-round call that
        # correctly declined STRICT there still fell through to THIS function and got re-approved for CUDA
        # unconditionally, since this was the only other place in the fallback chain that also checks STRICT.
        if fe_gpu_strict_enabled(n=n, p=p):
            return True
    except Exception as _strict_exc:  # nosec B110 - optional dependency import guard
        logger.debug("_cmi_fallback_choice: fe_gpu_strict_enabled probe failed (%s); falling through to the KTC lookup.", _strict_exc)
    if _CMI_SPEC is None:
        return None
    p_bucket = min(_CMI_SWEEP_P, key=lambda b: abs(b - int(p)))
    try:
        choice = _CMI_SPEC.choose(n_samples=int(n), p=int(p_bucket))
    except Exception as _choose_exc:
        logger.debug("_cmi_fallback_choice: KTC .choose(n_samples=%d, p=%d) failed (%s); no crossover verdict this call.", int(n), int(p_bucket), _choose_exc)
        return None
    if choice == "cuda":
        return True
    if choice == "cpu":
        return False
    return None


def cmi_default_use_cuda(n: int, p: int) -> bool:
    """Un-tuned hardcoded bootstrap crossover (the legacy heuristic), used only on cache miss."""
    return (int(n) * int(p)) >= _CMI_DEFAULT_NP_PRODUCT and int(p) >= _CMI_DEFAULT_MIN_P


def _make_cmi_inputs(dims: dict):
    """A discretized (n, P+2) int32 factor matrix: P candidate columns + a y column + a z column, each
    binned to a modest cardinality matching the MRMR greedy-round shapes the gate routes."""
    n = int(dims["n_samples"])
    p = int(dims["p"])
    nb = 16  # per-axis cardinality (joint_size = nb^3 = 4096, fits cc6.x 48KB shared)
    rng = np.random.default_rng(0)
    factors = rng.integers(0, nb, size=(n, p + 2)).astype(np.int32)
    cand_indices = np.arange(p, dtype=np.int64)
    y_index = p
    z_index = p + 1
    factors_nbins = np.full(p + 2, nb, dtype=np.int64)
    return (factors, cand_indices, y_index, z_index, factors_nbins)


def _cmi_variant(factors, cand_indices, y_index, z_index, factors_nbins, *, backend: str):
    """Run the batched conditional MI through an EXPLICIT backend ('cuda'/'cpu') -- the per-variant probe the
    sweep times. Both produce ~equal MI (the CUDA kernel reproduces the CPU reduction to ~1e-9), so the sweep's
    equivalence check passes and ranks by WALL only."""
    from ._cmi_cuda import conditional_mi_batched_dispatch

    return conditional_mi_batched_dispatch(
        factors, cand_indices, y_index, z_index, factors_nbins, force=backend,
    )


def _run_cmi_sweep() -> list:
    """Time CPU vs CUDA on the real conditional-MI dispatch across the n x p grid; faster EQUIVALENT wins per
    region. The kernel reproduces the CPU value to ~1e-9 so the equivalence tol is loose accordingly."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        "cuda": (lambda *a, **k: _cmi_variant(*a, backend="cuda")),
        "cpu": (lambda *a, **k: _cmi_variant(*a, backend="cpu")),
    }
    return list(sweep_backend_grid(
        variants,
        {"n_samples": _CMI_SWEEP_N_SAMPLES, "p": _CMI_SWEEP_P},
        _make_cmi_inputs,
        reference="cpu",
        repeats=3, equiv_rtol=1e-5, equiv_atol=1e-6,
    ))


def _cmi_fallback_choice(n_samples: int, p: int = 64) -> str:
    """Pre-sweep fallback: the historical hardcoded heuristic crossover."""
    return "cuda" if cmi_default_use_cuda(n_samples, p) else "cpu"


try:
    from pyutilz.performance.kernel_tuning.registry import TunerSpec, kernel_tuner

    _CMI_SPEC: "TunerSpec | None" = kernel_tuner(
        kernel_name="cmi_batched_cpu_cuda_crossover",
        variant_fns=(),  # CUDA path covered by salt; CPU is the reference
        tuner=_run_cmi_sweep,
        axes={"n_samples": list(_CMI_SWEEP_N_SAMPLES), "p": list(_CMI_SWEEP_P)},
        fallback=_cmi_fallback_choice,
        gpu_capable=True,
        salt=_CMI_SALT,
        cli_label="cmi_batched_cpu_cuda_crossover",
    )
except Exception as _spec_exc:
    logger.debug("info_theory._cmi_cuda_ktc: kernel_tuner spec construction failed (%s); CMI CPU/CUDA crossover stays on the hardcoded bootstrap fallback.", _spec_exc)
    _CMI_SPEC = None
