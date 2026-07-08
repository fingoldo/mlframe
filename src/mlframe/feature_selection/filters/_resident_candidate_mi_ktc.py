"""MANDATE-1/2 KTC crossover for the resident-GPU FE-gate candidate-MI path (2026-06-23).

Gates ``_resident_candidate_mi.best_existing_op_mi_resident`` (resident GPU candidate-gen + plug-in MI) against
the host ``_plugin_mi_classif_batch_njit`` it replaces. Per ``feedback_use_kernel_tuning_cache_for_gpu`` the
engage/skip decision is NOT a hardcoded k threshold: a ``kernel_tuner`` sweeps both paths over an (n, k) grid
and records, per region, which is faster. The resident path engages only where MEASURED faster; otherwise the
caller stays on the exact njit batch-MI.

HONEST EXPECTATION (GTX 1050 Ti, 6 SMs): the resident plug-in-MI crossover is ~k>=100 @ n=100k (the F2 gate
calls are k<=18-dominated), so on THIS card the sweep keeps small-k on CPU (the per-launch+sync overhead loses
sub-crossover) and selects the resident path for large-k / on stronger GPUs / larger p. The two paths are
selection-equivalent (percentile-edge vs rank binning, the approved FE-PAIR trade), so the sweep's equivalence
tol is loosened accordingly and it ranks by WALL.

CPU/no-cupy host: the sweep never runs, ``.choose()`` returns "njit", and the caller takes the exact host path.
"""
from __future__ import annotations

import numpy as np

# (n, k) grid. k spans the F2 gate distribution (small-k k~16) up to the wide orth-univariate / large-screen
# regimes (k>=128) where the resident gen+MI amortises its launch. The default fallback keeps the un-tuned
# host path (njit) -- the resident path is OPT-IN-BY-MEASUREMENT only, never the un-tuned default.
_RESCAND_SWEEP_N_SAMPLES = [50_000, 100_000, 300_000]
_RESCAND_SWEEP_K = [16, 64, 128, 256]
_RESCAND_SALT = 1


def rescand_use_resident(n: int, k: int) -> bool:
    """Per-host engage decision for the resident-GPU candidate-MI path, from the kernel_tuning_cache.

    Returns ``True`` (use the resident GPU gen+MI) only on a measured-faster cache hit; ``False`` on a miss /
    no-cupy / lookup failure (caller stays on the exact host njit batch-MI). ``k`` snaps to the nearest swept
    bucket. STRICT GPU mode (``MLFRAME_FE_GPU_STRICT=1``, diagnostic, default OFF) forces the resident path:
    the two binning schemes are selection-equivalent (the approved FE-PAIR trade)."""
    try:
        from ._fe_gpu_strict import fe_gpu_strict_enabled
        if fe_gpu_strict_enabled():
            return True
    except Exception:  # nosec B110 - optional dependency import guard
        pass
    if _RESCAND_SPEC is None:
        return False
    k_bucket = min(_RESCAND_SWEEP_K, key=lambda b: abs(b - int(k)))
    try:
        choice = _RESCAND_SPEC.choose(n_samples=int(n), k=int(k_bucket))
    except Exception:
        return False
    return bool(choice == "resident")


def _make_rescand_inputs(dims: dict):
    """An (n, k) host float64 candidate matrix + an int64 y, shaped like a gate MI call. The two probe
    variants score the SAME matrix (njit on host vs the resident GPU plug-in MI on its device copy), so the
    crossover measured is gen-agnostic launch+MI vs njit -- the dominant cost the gate routes."""
    n = int(dims["n_samples"])
    k = int(dims["k"])
    rng = np.random.default_rng(0)
    a = rng.uniform(0.1, 1.1, (n, k)).astype(np.float64)
    b = rng.uniform(0.1, 1.1, (n, k)).astype(np.float64)
    mat = (a * a) / b  # heavy-tailed, the kind of engineered candidate column the gate scores
    y = rng.integers(0, 4, size=n).astype(np.int64)
    return (np.ascontiguousarray(mat), y)


def _rescand_njit(mat, y):
    """Sweep probe variant: score ``mat`` on the host exact njit batch plug-in MI (the path being gated against)."""
    from .hermite_fe import _plugin_mi_classif_batch_njit

    return _plugin_mi_classif_batch_njit(mat, y, 20)


def _rescand_resident(mat, y):
    """Sweep probe variant: upload ``mat``/``y`` to the GPU and score via the resident plug-in MI kernel being gated."""
    import cupy as cp

    from . import hermite_fe as _hf
    from ._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    mat_gpu = cp.asarray(mat, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.int64)
    return _plugin_mi_classif_batch_cuda_resident(mat_gpu, y_gpu, 20)


def _run_rescand_sweep() -> list:
    """Time host njit MI vs the resident GPU plug-in MI across the (n, k) grid; faster EQUIVALENT wins per
    region. The two binning schemes (rank vs percentile-edge) differ only at ties -> selection-equivalent, so
    the equivalence tol is loosened (the approved FE-PAIR trade) and the sweep ranks by WALL."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {"njit": _rescand_njit, "resident": _rescand_resident}
    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        variants,
        {"n_samples": _RESCAND_SWEEP_N_SAMPLES, "k": _RESCAND_SWEEP_K},
        _make_rescand_inputs,
        reference="njit",
        repeats=3, equiv_rtol=5e-2, equiv_atol=5e-2,
    )


def _rescand_fallback_choice(n_samples: int, k: int = 16) -> str:
    """Pre-sweep fallback: the host njit path (the resident path engages only when MEASURED faster)."""
    return "njit"


try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _RESCAND_SPEC = kernel_tuner(
        kernel_name="fe_gate_resident_candidate_mi_crossover",
        variant_fns=(),  # GPU resident path covered by salt; njit is the reference
        tuner=_run_rescand_sweep,
        axes={"n_samples": list(_RESCAND_SWEEP_N_SAMPLES), "k": list(_RESCAND_SWEEP_K)},
        fallback=_rescand_fallback_choice,
        gpu_capable=True,
        salt=_RESCAND_SALT,
        cli_label="fe_gate_resident_candidate_mi_crossover",
    )
except Exception:
    _RESCAND_SPEC = None
